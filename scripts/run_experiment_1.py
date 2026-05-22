"""run_experiment_1.py

Sandbox-friendly runner for the FIRST entry of the registry in
`scripts/run_experiments.py` — `G0_classical_baseline`. The full orchestrator
relies on `torch` and `torch_geometric`; this minimal runner only depends on
NumPy, PennyLane (its autograd-aware NumPy wrapper), scikit-learn, and W&B,
which is what is available in this environment. The training pipeline is the
same in shape: load QG12500 graph data, generate two augmented views per
graph, train a permutation-invariant encoder under a margin contrastive loss,
then linear-probe the frozen embeddings on a held-out test set and report
accuracy and AUC.

Backbone substitution: the registry's GAT-GNN backbone is replaced here with a
Deep-Sets / Particle-Flow-Network-style mean-pool MLP (Zaheer et al. 2017,
Komiske, Metodiev, Thaler 2019). This is a strict subset of GAT (no learned
edge attention), so the result is a *lower bound* on what the full GAT
baseline would deliver — the goal of this script is to demonstrate the
end-to-end pipeline, not to reproduce the 73.28% headline number.

We also print, for reference, the canonical QC1 quantum circuit that the
next-up experiments (H1, H1F, H7, H8, H9, H10, H11) all use, so the user can
inspect both the classical model that was actually trained and the quantum
circuit that the hybrid variants would attach to it.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_qg_subset(n_train: int, n_val: int, n_test: int, seed: int):
    """Stratified subsample of the QG12500 particle-cloud dataset."""
    x = np.load(REPO_ROOT / "data" / "qg_graph" / "x10_sorted_12500.npy")
    y = np.load(REPO_ROOT / "data" / "qg_graph" / "y10_sorted_12500.npy").astype(np.int64)
    rng = np.random.default_rng(seed)

    half = (n_train + n_val + n_test) // 2
    idx0 = np.where(y == 0)[0]; rng.shuffle(idx0)
    idx1 = np.where(y == 1)[0]; rng.shuffle(idx1)
    sel = np.concatenate([idx0[:half], idx1[:half]])
    rng.shuffle(sel)

    sel_train = sel[:n_train]
    sel_val = sel[n_train:n_train + n_val]
    sel_test = sel[n_train + n_val:n_train + n_val + n_test]

    return (
        (x[sel_train].astype(np.float32), y[sel_train]),
        (x[sel_val].astype(np.float32), y[sel_val]),
        (x[sel_test].astype(np.float32), y[sel_test]),
    )


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two augmentations used: gaussian feature jitter (sigma 2%) + per-particle
    dropout with p=0.1 implemented by zeroing rows. Both preserve the
    permutation-invariance of the encoder."""
    out = x + rng.normal(0, 0.02, x.shape).astype(np.float32)
    if x.ndim == 3:
        keep = (rng.random(x.shape[:2]) > 0.1).astype(np.float32)[..., None]
        out = out * keep
    return out


# ---------------------------------------------------------------------------
# Encoder (Deep-Sets / PFN-style)
# ---------------------------------------------------------------------------

def init_params(input_dim: int, hidden_dim: int, embedding_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    scale_in  = np.sqrt(2.0 / input_dim)
    scale_h   = np.sqrt(2.0 / hidden_dim)
    return {
        "W1": pnp.array(rng.normal(0, scale_in, (input_dim, hidden_dim)).astype(np.float64), requires_grad=True),
        "b1": pnp.array(np.zeros(hidden_dim),                               requires_grad=True),
        "W2": pnp.array(rng.normal(0, scale_h, (hidden_dim, hidden_dim)).astype(np.float64), requires_grad=True),
        "b2": pnp.array(np.zeros(hidden_dim),                               requires_grad=True),
        "W3": pnp.array(rng.normal(0, scale_h, (hidden_dim, embedding_dim)).astype(np.float64), requires_grad=True),
        "b3": pnp.array(np.zeros(embedding_dim),                             requires_grad=True),
    }


def encoder_forward(params, x):
    """x: (B, M, F) -> (B, embedding_dim) embeddings on the unit sphere.

    Architecture: per-particle Linear -> tanh -> Linear -> tanh -> mean pool ->
    Linear -> L2 normalize. The mean pool makes the encoder permutation
    invariant; the per-particle nonlinearity gives it more capacity than a
    plain mean-of-features baseline."""
    B, M, F = x.shape
    h = x.reshape(B * M, F)
    h = pnp.tanh(h @ params["W1"] + params["b1"])
    h = pnp.tanh(h @ params["W2"] + params["b2"])
    h = h.reshape(B, M, -1).mean(axis=1)         # (B, hidden_dim)
    z = h @ params["W3"] + params["b3"]          # (B, embedding_dim)
    norm = pnp.sqrt(pnp.sum(z * z, axis=1, keepdims=True) + 1e-9)
    return z / norm


def contrastive_pair_loss(params, x1, x2, labels, margin=1.0):
    """Margin contrastive loss (Hadsell, Chopra, LeCun 2006). For positive
    pairs (label=1) the loss is squared distance; for negatives it is the
    hinge `max(margin - d, 0)^2`."""
    z1 = encoder_forward(params, x1)
    z2 = encoder_forward(params, x2)
    d = pnp.sqrt(pnp.sum((z1 - z2) ** 2, axis=1) + 1e-9)
    pos = labels * d ** 2
    neg = (1 - labels) * pnp.maximum(margin - d, 0) ** 2
    return pnp.mean(pos + neg)


# ---------------------------------------------------------------------------
# Adam optimiser (numpy)
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = np.asarray(grads[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = pnp.array(
                np.asarray(params[k]) - self.lr * m_hat / (np.sqrt(v_hat) + self.eps),
                requires_grad=True,
            )
        return params


# ---------------------------------------------------------------------------
# Reference QC1 circuit (used by H1, H1F, H7, H8, H9, H10, H11)
# ---------------------------------------------------------------------------

def build_qc1(n_qubits: int = 8, n_layers: int = 3):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        # Angle embedding via RY on each qubit.
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        # Basic entangler: per-layer RX rotations + chain CNOTs.
        for layer in range(weights.shape[0]):
            for i in range(n_qubits):
                qml.RX(weights[layer][i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ---------------------------------------------------------------------------
# Linear probe (downstream evaluation)
# ---------------------------------------------------------------------------

def linear_probe(encoder_params, train_x, train_y, test_x, test_y):
    z_tr = np.asarray(encoder_forward(encoder_params, pnp.array(train_x)))
    z_te = np.asarray(encoder_forward(encoder_params, pnp.array(test_x)))
    clf = LogisticRegression(max_iter=500, n_jobs=1).fit(z_tr, train_y)
    p_te = clf.predict_proba(z_te)[:, 1]
    return accuracy_score(test_y, (p_te > 0.5).astype(int)), roc_auc_score(test_y, p_te)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    # Stage 0 (smoke) hyperparameters from `STAGES["stage0_smoke"]`.
    cfg = dict(
        name="G0_classical_baseline",
        backbone="deep_sets_pfn_substitute",
        quantum_head="none",
        loss="pairs",
        margin=1.0,
        embedding_dim=16,
        hidden_dim=32,
        batch_size=64,
        lr=1e-3,
        epochs=5,
        n_train=400,
        n_val=50,
        n_test=50,
        seed=SEED,
        stage="stage0_smoke",
        n_qubits_reference=8,
        n_quantum_layers_reference=3,
        notes="torch unavailable in sandbox -> Deep-Sets/PFN (Komiske 2019) "
              "stand-in for GAT; substantively the same pipeline at lower "
              "encoder capacity.",
    )

    # ----- W&B -----
    import wandb
    os.environ.setdefault("WANDB_DIR", str(REPO_ROOT / "wandb"))
    run = wandb.init(
        project="qssl-experiments",
        name=f"{cfg['name']}__{cfg['stage']}__seed{cfg['seed']}",
        config=cfg,
        mode="offline",   # no creds in sandbox; logs are kept locally
        reinit=True,
    )

    # ----- Data -----
    print(">>> Loading QG12500 ...")
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_qg_subset(
        cfg["n_train"], cfg["n_val"], cfg["n_test"], cfg["seed"]
    )
    F = x_tr.shape[-1]
    print(f"    train {x_tr.shape}  val {x_va.shape}  test {x_te.shape}  features={F}")
    print(f"    class balance (train) -> 0:{int((y_tr==0).sum())}, 1:{int((y_tr==1).sum())}")

    # ----- Model -----
    params = init_params(F, cfg["hidden_dim"], cfg["embedding_dim"], cfg["seed"])
    n_params = int(sum(np.asarray(v).size for v in params.values()))

    print()
    print("=" * 78)
    print("MODEL — Deep-Sets / PFN encoder used for G0_classical_baseline")
    print("=" * 78)
    print(f"  Input shape per graph: ({x_tr.shape[1]} particles, {F} features)")
    print(f"  Layer 1 (per-particle):   Linear({F} -> {cfg['hidden_dim']}) + tanh")
    print(f"                            params: W1 {params['W1'].shape}, b1 {params['b1'].shape}")
    print(f"  Layer 2 (per-particle):   Linear({cfg['hidden_dim']} -> {cfg['hidden_dim']}) + tanh")
    print(f"                            params: W2 {params['W2'].shape}, b2 {params['b2'].shape}")
    print(f"  Pool:                     mean over particles (permutation invariant)")
    print(f"  Layer 3 (graph-level):    Linear({cfg['hidden_dim']} -> {cfg['embedding_dim']})")
    print(f"                            params: W3 {params['W3'].shape}, b3 {params['b3'].shape}")
    print(f"  Output:                   L2-normalised embedding in R^{cfg['embedding_dim']}")
    print(f"  Total trainable params:   {n_params:,}")
    print(f"  Quantum head:             none  (G0 is the classical baseline)")
    print()

    # ----- Reference quantum circuit -----
    print("=" * 78)
    print(f"REFERENCE QUANTUM CIRCUIT — QC1 ({cfg['n_qubits_reference']} qubits, "
          f"{cfg['n_quantum_layers_reference']} layers)")
    print("    (used by experiments H1/H1F/H7/H8/H9/H10/H11; not used in G0)")
    print("=" * 78)
    qc1 = build_qc1(cfg["n_qubits_reference"], cfg["n_quantum_layers_reference"])
    sample_inputs = pnp.linspace(0.1, 0.9, cfg["n_qubits_reference"])
    sample_weights = pnp.array(np.zeros((cfg["n_quantum_layers_reference"],
                                         cfg["n_qubits_reference"])) + 0.1)
    print(qml.draw(qc1, level="device")(sample_inputs, sample_weights))
    print()

    # ----- Training loop -----
    grad_fn = qml.grad(contrastive_pair_loss, argnum=0)
    opt = Adam(params, lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"])

    print("=" * 78)
    print("TRAINING")
    print("=" * 78)
    history = []
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        perm = rng.permutation(cfg["n_train"])
        losses = []
        for s in range(0, cfg["n_train"], cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            x_batch = x_tr[idx]
            # Half positive (same graph augmented twice) + half negative
            # (different graph augmented once).
            B = len(idx)
            half = B // 2
            x1_pos = augment(x_batch[:half], rng)
            x2_pos = augment(x_batch[:half], rng)
            shuf = rng.permutation(half) ^ 1   # different element pairing
            x1_neg = augment(x_batch[half:half + half], rng)
            x2_neg = augment(x_batch[half:half + half][shuf % half], rng)
            x1 = np.concatenate([x1_pos, x1_neg], axis=0).astype(np.float32)
            x2 = np.concatenate([x2_pos, x2_neg], axis=0).astype(np.float32)
            labels = np.concatenate([np.ones(half), np.zeros(half)]).astype(np.float32)

            grads = grad_fn(params, pnp.array(x1), pnp.array(x2),
                            pnp.array(labels), cfg["margin"])
            params = opt.step(params, grads)
            losses.append(float(contrastive_pair_loss(
                params, pnp.array(x1), pnp.array(x2), pnp.array(labels), cfg["margin"]
            )))
        train_loss = float(np.mean(losses))
        wall = time.time() - t0
        history.append((epoch + 1, train_loss, wall))
        wandb.log({"train/loss": train_loss, "epoch": epoch + 1, "wall_clock_s": wall},
                  step=epoch + 1)
        print(f"  epoch {epoch+1}/{cfg['epochs']}: train_loss={train_loss:.5f}  "
              f"({wall:.2f}s)")

    # ----- Evaluation -----
    print()
    print("=" * 78)
    print("EVALUATION (linear probe on frozen embeddings)")
    print("=" * 78)
    acc, auc = linear_probe(params, x_tr, y_tr, x_te, y_te)
    print(f"  test accuracy: {acc:.4f}")
    print(f"  test AUC:      {auc:.4f}")
    wandb.log({"eval/test_accuracy": acc, "eval/test_auc": auc, "params/total": n_params})

    # ----- Persist a JSON record next to the W&B offline run -----
    out = {
        "config": cfg,
        "n_params": n_params,
        "train_history": [{"epoch": e, "loss": l, "wall_s": w} for e, l, w in history],
        "test_accuracy": float(acc),
        "test_auc": float(auc),
    }
    out_path = REPO_ROOT / "experiment_1_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved record to {out_path}")
    run.finish()


if __name__ == "__main__":
    main()
