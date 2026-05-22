"""
run_experiments.py — Staged orchestrator for the next round of QML contrastive
learning experiments on the Quark-Gluon dataset.

Why this script exists
----------------------
`benchmarking.md` shows the project's current best is a classical GAT-GNN at 73.28% / AUC
0.7984 and the best hybrid is QC3 at 67.02% / AUC 0.7285. `ablations.md` shows that the
within-config noise band of the classical GNN spans 16.2 percentage points (std 5.4),
which is wider than the apparent classical-vs-hybrid gap. `experiments.md` lays out the
research plan. This script turns that plan into a runnable orchestrator: it screens many
candidates on small data, promotes the survivors to medium data, and only the finalists
are run on the full dataset. Every run is fully seeded and logged to W&B.

Stages
------
Stage 0 — smoke   : 500 graphs,  5 epochs, 1 seed.       (~2-5 min/run on a single GPU)
Stage 1 — screen  : 2,500 graphs, 20 epochs, 3 seeds.    Eliminate ~80% of candidates.
Stage 2 — bench   : 12,500 graphs, 50 epochs, 5 seeds.   Headline numbers + ablation grid.
Stage 3 — scale   : 100k / 933k graphs.                  Only finalists; not implemented here.

Selection rule between stages: a candidate's mean test AUC must be >= (best_so_far.mean
- promotion_slack). promotion_slack defaults to 0.01 at Stage 0->1 and 0.005 at Stage 1->2.
This keeps configurations that are within noise of the leader so they get a fair five-seed
shot at Stage 2.

Limitations honored by this script
----------------------------------
- We do not have access to real quantum hardware; all runs use PennyLane simulators.
- `default.qubit` is the noiseless baseline; `default.mixed` is used for noise tracks B
  with a depolarizing + amplitude/phase damping model. Track C (real-hardware calibration)
  is left as a TODO since it requires a current calibration JSON.
- 8-qubit / few-layer ansatze are the practical ceiling on a single-GPU dev box; deeper
  circuits become prohibitively slow under the parameter-shift rule.
- Barren-plateau mitigation is via small-rotation-init and local cost functions; layer-wise
  training is left for a future iteration.
- Only graph encoders are wired up here. Image encoders (CNN/ResNet) are not in scope for
  this round because the ablation showed they collapse on QG without further engineering.

Experiment registry — what we run and why (with literature)
----------------------------------------------------------
G0  : Classical GAT-GNN baseline. (anchor; matches `rnnye83c` in benchmarking)
G1  : GAT-GNN + classical MLP projection head, parameter-matched to QC1.
        Why: parameter-matched control to test whether any quantum head AUC gain is
        attributable to the *quantum* part rather than to extra capacity. (Sec. 8 of
        experiments.md, falsification test #1.)
H1  : GAT-GNN + QC1 (angle + basic entangler), pair loss.   [reproduces benchmarking row]
H1F : GAT-GNN + QC1, *quantum-fidelity* loss.
        Why: the existing data confounds circuit and loss. QC2 is the only fidelity-loss
        row, so we cannot tell if fidelity is helping or hurting independently of QC2.
        (Ablations sec. 3 — fills the missing 2x2 cell.)
H2  : GAT-GNN + QC2 (angle + entangler), *pair* loss.
        Why: same — fills the other missing 2x2 cell.
H3  : GAT-GNN + QC3 (amplitude + ring entangler), pair loss. [reproduces benchmarking row]
H4  : GAT-GNN + QC4 (data re-uploading), pair loss.
        Why: Pérez-Salinas et al. 2020 show universal approximation with re-uploading;
        not yet tested in QMLHEP.
H5  : GAT-GNN + QC6 (Hardware-Efficient Ansatz), pair loss.
        Why: standard NISQ ansatz; baseline against problem-inspired circuits, also our
        best candidate to study barren-plateau onset as `n_qubits` and depth grow.
H7  : GAT-GNN + QC1 + NT-Xent loss.
        Why: NT-Xent is implemented in qssl/loss/losses.py but never benchmarked. Chen et
        al. 2020 (SimCLR) show NT-Xent is more sample-efficient than pairwise contrastive.
H8  : GAT-GNN + QC1 + Barlow Twins loss.
        Why: Zbontar et al. 2021 — removes the negative-pair requirement and is robust to
        small batch sizes. Hybrid models are slow per-step, so smaller batches help.
H9  : GAT-GNN + QC1 + VICReg loss.
        Why: Bardes, Ponce, LeCun 2022 — variance/invariance/covariance terms that
        explicitly penalise the embedding-collapse failure mode we observed in CNN runs.
H10 : GAT-GNN + QC1 + SWAP-test fidelity loss.
        Why: Lloyd, Schuld et al. 2020 — quantum metric learning. Connects directly to
        QC2's existing fidelity setup but uses a circuit-level SWAP test rather than an
        L2 surrogate.
H11 : GAT-GNN + QC1, noisy simulator (Track B).
        Why: any AUC gap that vanishes under realistic noise is not a real advantage.

Five seeds per surviving config at Stage 2; three seeds at Stage 1; one at Stage 0.

Hyperparameters that are fixed across the registry (so they can be ablated separately
later): batch size 128, Adam lr 1e-3 with 5-epoch cosine warmup, embedding dim 64, GAT
hidden dims (32, 32, 32), dropout 0.5, margin 1.0, NT-Xent temperature 0.3, n_qubits 8,
n_quantum_layers 3.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Make qssl importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# We import torch lazily so that --dry-run works without a GPU/torch install.
def _lazy_imports():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import wandb
    import pennylane as qml
    from torch_geometric.loader import DataLoader as PygDataLoader
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    from qssl.models.qgnn import GNN, HybridQuantumGNN
    from qssl.loss.losses import ContrastiveLoss

    return dict(
        torch=torch, nn=nn, F=F, wandb=wandb, qml=qml,
        PygDataLoader=PygDataLoader,
        LogisticRegression=LogisticRegression,
        roc_auc_score=roc_auc_score, accuracy_score=accuracy_score,
        GNN=GNN, HybridQuantumGNN=HybridQuantumGNN, ContrastiveLoss=ContrastiveLoss,
    )


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Pin every RNG we touch so that two runs with the same config are identical."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Pennylane's default.qubit accepts a seed via the wires arg in newer versions;
    # older versions are deterministic by default.


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """One row of the experiment grid. Anything that affects training lives here so it
    is logged to W&B and pinned to disk."""

    name: str
    backbone: str = "gat_gnn"          # gat_gnn | mlp_control
    quantum_head: str = "none"         # none | qc1 | qc2 | qc3 | qc4 | qc6 | mlp_match
    loss: str = "pairs"                # pairs | ntxent | barlow_twins | vicreg | swap_fidelity | quantum_fidelity
    n_qubits: int = 8
    n_quantum_layers: int = 3
    embedding_dim: int = 64
    hidden_dims: Tuple[int, ...] = (32, 32, 32)
    batch_size: int = 128
    lr: float = 1e-3
    margin: float = 1.0
    temperature: float = 0.3
    dropout: float = 0.5
    noise_track: str = "A"             # A noiseless | B mixed-state-with-noise
    use_local_cost: bool = False       # barren-plateau mitigation knob
    small_init_scale: float = 0.1      # rotation init std; small => avoid plateau region
    notes: str = ""

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class StageConfig:
    name: str
    n_train: int
    n_val: int
    n_test: int
    epochs: int
    seeds: Tuple[int, ...]
    promotion_slack: float       # how far below the leader still gets promoted

STAGES: Dict[str, StageConfig] = {
    "stage0_smoke":  StageConfig("stage0_smoke",   400, 50,  50,  5,  (42,),               0.10),
    "stage1_screen": StageConfig("stage1_screen", 2000, 250, 250, 20, (0, 42, 1337),       0.01),
    "stage2_bench":  StageConfig("stage2_bench", 10000, 1250, 1250, 50, (0, 1, 13, 42, 123), 0.005),
}


# -----------------------------------------------------------------------------
# Quantum heads
# -----------------------------------------------------------------------------

def make_quantum_head(name: str, n_qubits: int, n_layers: int, init_scale: float,
                      noise_track: str = "A"):
    """Return a torch.nn.Module that takes a (B, n_qubits) tensor and returns
    (B, n_qubits) expectation values. We rebuild the circuit per call so the
    parameter shapes are fresh for each experiment."""
    import pennylane as qml
    import torch
    import torch.nn as nn

    dev_name = "default.qubit" if noise_track == "A" else "default.mixed"
    dev = qml.device(dev_name, wires=n_qubits)
    # backprop traces the circuit as PyTorch matrix ops — ~100x faster than
    # parameter-shift on GPU.  default.mixed doesn't support backprop, so
    # noisy-track experiments fall back to parameter-shift.
    diff_method = "backprop" if noise_track == "A" else "parameter-shift"

    if name == "qc1":
        # Angle embedding + basic entangler (matches qssl/models/qc.py QC1).
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for layer in range(weights.shape[0]):
                for i in range(n_qubits):
                    qml.RX(weights[layer][i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        weight_shape = (n_layers, n_qubits)

    elif name == "qc2":
        # Same backbone as qc1 but pure inputs (no trainable weights) — use as a fixed
        # feature map; we add a thin trainable layer to keep gradient flow alive.
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for layer in range(weights.shape[0]):
                for i in range(n_qubits):
                    qml.RZ(weights[layer][i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        weight_shape = (n_layers, n_qubits)

    elif name == "qc3":
        # Amplitude embedding + ring entangler.
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0)
            for layer in range(weights.shape[0]):
                for i in range(n_qubits):
                    qml.RY(weights[layer][i], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        weight_shape = (n_layers, n_qubits)

    elif name == "qc4":
        # Data re-uploading (Pérez-Salinas et al. 2020).
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for layer in range(weights.shape[0]):
                # Re-upload data each layer scaled by trainable weights.
                for i in range(n_qubits):
                    qml.RY(inputs[i] * weights[layer, i, 0] + weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        weight_shape = (n_layers, n_qubits, 3)

    elif name == "qc6":
        # Hardware-efficient ansatz — Ry/Rz rotations with brick-wall CNOTs.
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for layer in range(weights.shape[0]):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # Brick-wall CNOTs: even pairs on even layers, odd pairs on odd layers.
                start = layer % 2
                for i in range(start, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        weight_shape = (n_layers, n_qubits, 2)

    else:
        raise ValueError(f"unknown quantum head: {name}")

    init = torch.randn(*weight_shape) * init_scale

    # PennyLane ≥0.38 expects init_method to modify the parameter tensor in-place
    # (like a torch.nn.init function).  A lambda that *returns* a clone is silently
    # ignored — so we wrap with a proper closure that copies the desired init values.
    _init_tensor = init.clone()
    def _init_fn(tensor):
        with torch.no_grad():
            tensor.copy_(_init_tensor)

    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes={"weights": list(weight_shape)},
                                init_method=_init_fn)

    # PennyLane ≥0.36 passes the full batch tensor to the circuit, but the circuits
    # here index features as inputs[i] (not inputs[:, i]).  Wrapping with a per-sample
    # loop is slower but correct across all PL/torch version combinations.
    class _PerSampleLayer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
        def forward(self, x):
            return torch.stack([self.layer(x[i]) for i in range(x.shape[0])])

    return _PerSampleLayer(qlayer)


def make_classical_match_head(input_dim: int, n_qubits: int, n_quantum_layers: int):
    """Parameter-matched classical control: an MLP whose number of trainable parameters
    roughly matches a QC1-style head (n_layers * n_qubits parameters)."""
    import torch.nn as nn
    target_params = n_quantum_layers * n_qubits
    # An MLP `n_qubits -> h -> n_qubits` has h*n_qubits + h + n_qubits*h + n_qubits params
    # ~ 2*h*n_qubits + h + n_qubits ; pick h so total matches target_params.
    h = max(1, target_params // max(2 * n_qubits + 1, 1))
    return nn.Sequential(nn.Linear(n_qubits, h), nn.ReLU(), nn.Linear(h, n_qubits))


# -----------------------------------------------------------------------------
# Encoder / projection head
# -----------------------------------------------------------------------------

try:
    _torch_nn_Module = __import__('torch').nn.Module
except Exception:
    class _torch_nn_Module:  # type: ignore[no-redef]
        """Stub used when torch is not installed; lets the registry import cleanly
        in numpy-only environments. Instantiating GraphEncoder would still fail —
        only registry+stages dataclasses are usable without torch."""
        def __init__(self, *a, **kw): raise RuntimeError("torch not installed")


class GraphEncoder(_torch_nn_Module):
    """Wraps the existing GAT-based GNN with an optional quantum/classical projection
    head. The pooled vector is L2-normalized before going into the contrastive loss so
    that all losses operate on a unit hypersphere."""

    def __init__(self, cfg: ExperimentConfig, in_features: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from qssl.models.qgnn import GNN
        super().__init__()
        self.cfg = cfg
        # Reuse the existing GAT-GNN; output_dims = embedding_dim.
        self.backbone = GNN(
            input_dims=in_features,
            hidden_dims=list(cfg.hidden_dims),
            output_dims=cfg.embedding_dim,
            activ_fn=F.leaky_relu,
        )

        # Project to n_qubits before the quantum/classical head.
        self.pre_head = nn.Linear(cfg.embedding_dim, cfg.n_qubits)

        if cfg.quantum_head == "none":
            self.head = nn.Identity()
            self.post_head_dim = cfg.n_qubits
        elif cfg.quantum_head == "mlp_match":
            self.head = make_classical_match_head(cfg.n_qubits, cfg.n_qubits, cfg.n_quantum_layers)
            self.post_head_dim = cfg.n_qubits
        else:
            self.head = make_quantum_head(
                cfg.quantum_head, cfg.n_qubits, cfg.n_quantum_layers,
                cfg.small_init_scale, cfg.noise_track,
            )
            self.post_head_dim = cfg.n_qubits

    def forward(self, x, edge_index, batch):
        import torch
        import torch.nn.functional as F
        h = self.backbone(x, edge_index, batch)        # (B, embedding_dim)
        h = torch.tanh(self.pre_head(h))               # bound in [-1,1] for AngleEmbedding
        # Quantum heads expect per-row inputs — TorchLayer handles batching for us.
        h = self.head(h) if not isinstance(self.head, type(F.relu)) else h
        return F.normalize(h, dim=-1)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def make_loss_fn(cfg: ExperimentConfig):
    import torch
    import torch.nn.functional as F
    from qssl.loss.losses import ContrastiveLoss

    if cfg.loss == "pairs":
        return ContrastiveLoss(margin=cfg.margin, mode="pairs")

    if cfg.loss == "ntxent":
        return ContrastiveLoss(mode="ntxent", batch_size=cfg.batch_size,
                               temperature=cfg.temperature)

    if cfg.loss == "barlow_twins":
        # Zbontar et al. 2021. Lambda_off-diag default = 5e-3.
        lambd = 5e-3
        def loss_fn(z1, z2):
            B, D = z1.shape
            z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
            z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
            c = (z1.T @ z2) / B
            on_diag = (torch.diagonal(c) - 1).pow(2).sum()
            off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
            return on_diag + lambd * off_diag
        return loss_fn

    if cfg.loss == "vicreg":
        # Bardes, Ponce, LeCun 2022. Defaults: 25 / 25 / 1.
        lam_var, lam_inv, lam_cov = 25.0, 25.0, 1.0
        def loss_fn(z1, z2):
            inv = F.mse_loss(z1, z2)
            std_z1 = torch.sqrt(z1.var(0) + 1e-4)
            std_z2 = torch.sqrt(z2.var(0) + 1e-4)
            var = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
            B, D = z1.shape
            z1c = z1 - z1.mean(0); z2c = z2 - z2.mean(0)
            cov_z1 = (z1c.T @ z1c) / (B - 1)
            cov_z2 = (z2c.T @ z2c) / (B - 1)
            off1 = cov_z1.flatten()[:-1].view(D - 1, D + 1)[:, 1:].pow(2).sum() / D
            off2 = cov_z2.flatten()[:-1].view(D - 1, D + 1)[:, 1:].pow(2).sum() / D
            return lam_inv * inv + lam_var * var + lam_cov * (off1 + off2)
        return loss_fn

    if cfg.loss == "swap_fidelity":
        # Surrogate SWAP-test fidelity for unit-vector embeddings:
        # F = 0.5 * (1 + |<z1, z2>|^2). Loss is 1 - F. Pulls similar pairs together
        # and (because of label) pushes dissimilar apart.
        def loss_fn(z1, z2, label=None):
            inner = (z1 * z2).sum(-1)
            fidelity = 0.5 * (1 + inner.pow(2))
            return (1 - fidelity).mean()
        return loss_fn

    if cfg.loss == "quantum_fidelity":
        # MSE between embeddings — matches the existing TF
        # `Losses.quantum_fidelity_loss`.
        def loss_fn(z1, z2, label=None):
            return F.mse_loss(z1, z2)
        return loss_fn

    raise ValueError(f"unknown loss: {cfg.loss}")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_qg_graph_pairs(stage: StageConfig, seed: int):
    """Load the QG12500 particle-cloud graphs and build contrastive pairs.

    The repo ships pre-processed tensors at data/qg_graph/. To keep this script
    self-contained, we expect:
        data/qg_graph/x10_sorted_12500.npy    (N, M, F)  features per particle
        data/qg_graph/y10_sorted_12500.npy    (N,)       binary labels
    Edges are k-NN in (eta, phi) per graph. This is a simplified loader; for the
    full preprocessing pipeline see qssl/data/data_loader.py.
    """
    import torch
    from torch_geometric.data import Data

    root = REPO_ROOT / "data" / "qg_graph"
    x_path = root / "x10_sorted_12500.npy"
    y_path = root / "y10_sorted_12500.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {x_path} / {y_path}. Provide preprocessed graphs or call "
            f"qssl.data.data_loader.GraphMaker first."
        )

    x = np.load(x_path)               # (N, M, F)
    y = np.load(y_path).astype(np.int64)
    rng = np.random.default_rng(seed)

    N = x.shape[0]
    n_total = stage.n_train + stage.n_val + stage.n_test
    if n_total > N:
        raise RuntimeError(f"stage requests {n_total} graphs but only {N} are available")

    # Stratified subsample so each split has a class balance close to the parent.
    idx_by_class = [np.where(y == c)[0] for c in (0, 1)]
    for arr in idx_by_class:
        rng.shuffle(arr)
    half = n_total // 2
    sel = np.concatenate([idx_by_class[0][:half], idx_by_class[1][:half]])
    rng.shuffle(sel)

    sel_train = sel[:stage.n_train]
    sel_val = sel[stage.n_train:stage.n_train + stage.n_val]
    sel_test = sel[stage.n_train + stage.n_val:]

    def to_graph_list(indices):
        out = []
        for idx in indices:
            feats = torch.from_numpy(x[idx]).float()              # (M, F)
            # k-NN edges by first two features (eta, phi) — simplified.
            eta_phi = feats[:, :2]
            d = torch.cdist(eta_phi, eta_phi)
            d.fill_diagonal_(float("inf"))
            knn = d.topk(k=min(4, feats.size(0) - 1), largest=False).indices  # (M, k)
            src = torch.arange(feats.size(0)).unsqueeze(1).expand_as(knn).reshape(-1)
            dst = knn.reshape(-1)
            edge_index = torch.stack([src, dst], dim=0)
            out.append(Data(x=feats, edge_index=edge_index, y=torch.tensor(int(y[idx]))))
        return out

    return to_graph_list(sel_train), to_graph_list(sel_val), to_graph_list(sel_test)


def augment_graph(data, rng: np.random.Generator):
    """Cheap, fast augmentation: random per-feature noise + random node dropout."""
    import torch
    out = data.clone()
    noise = torch.from_numpy(rng.normal(0, 0.02, out.x.shape).astype(np.float32))
    out.x = out.x + noise
    if out.x.size(0) > 4 and rng.random() < 0.5:
        keep = rng.choice(out.x.size(0), size=int(out.x.size(0) * 0.9), replace=False)
        keep = torch.from_numpy(np.sort(keep))
        out.x = out.x[keep]
        # Re-map edges to surviving nodes.
        node_map = {int(k): i for i, k in enumerate(keep.tolist())}
        ei = out.edge_index.tolist()
        new_ei = [[node_map[s], node_map[d]] for s, d in zip(ei[0], ei[1])
                  if s in node_map and d in node_map]
        if not new_ei:
            new_ei = [[0, 0]]
        out.edge_index = torch.tensor(new_ei, dtype=torch.long).t().contiguous()
    return out


def build_pair_loader(graphs, batch_size: int, seed: int, shuffle: bool):
    """Yields (batch_a, batch_b, labels) where pairs come from the same graph with
    different augmentations (positive) — we then form negatives by shuffling within
    the batch, so labels are 1 along the diagonal and 0 elsewhere when forming
    full pair tensors. Here we keep a simpler convention: every pair in the batch
    is positive (label 1), and the loss objects derive negatives from the batch
    itself (NT-Xent / Barlow Twins / VICReg) or from `form_pairs` (pair loss)."""
    import torch
    from torch_geometric.loader import DataLoader as PygDataLoader
    rng = np.random.default_rng(seed)

    class _Pairs:
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            d = self.base[i]
            return augment_graph(d, rng), augment_graph(d, rng), int(d.y)

    # PyG DataLoader doesn't natively pair, so we batch lists of (a, b, label).
    def collate(samples):
        from torch_geometric.data import Batch
        a_list = [s[0] for s in samples]
        b_list = [s[1] for s in samples]
        labels = torch.tensor([s[2] for s in samples], dtype=torch.long)
        return Batch.from_data_list(a_list), Batch.from_data_list(b_list), labels

    return torch.utils.data.DataLoader(
        _Pairs(graphs), batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
        drop_last=True,
    )


def build_eval_loader(graphs, batch_size: int):
    import torch
    from torch_geometric.data import Batch

    def collate(samples):
        labels = torch.tensor([int(s.y) for s in samples], dtype=torch.long)
        return Batch.from_data_list(samples), labels

    return torch.utils.data.DataLoader(
        graphs, batch_size=batch_size, shuffle=False, collate_fn=collate,
    )


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def linear_probe_auc(model, train_graphs, test_graphs, device):
    """Freeze encoder, fit a logistic regression on embeddings, return (acc, auc)."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    model.eval()
    def embed(loader):
        zs, ys = [], []
        with torch.no_grad():
            for batch, y in loader:
                batch = batch.to(device)
                z = model(batch.x.float(), batch.edge_index, batch.batch)
                zs.append(z.cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(zs), np.concatenate(ys)

    z_tr, y_tr = embed(build_eval_loader(train_graphs, 256))
    z_te, y_te = embed(build_eval_loader(test_graphs, 256))
    clf = LogisticRegression(max_iter=200).fit(z_tr, y_tr)
    p_te = clf.predict_proba(z_te)[:, 1]
    return accuracy_score(y_te, (p_te > 0.5).astype(int)), roc_auc_score(y_te, p_te)


def train_one(cfg: ExperimentConfig, stage: StageConfig, seed: int,
              wandb_project: str, wandb_mode: str = "online") -> Dict:
    """Run a single (config, seed) pair through one stage. Returns metrics dict."""
    import torch
    import wandb
    from torch_geometric.data import Batch

    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- W&B init --------------------------------------------------------------
    run_name = f"{cfg.name}__{stage.name}__seed{seed}"
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config={**cfg.to_dict(), "stage": stage.name, "seed": seed,
                "n_train": stage.n_train, "epochs": stage.epochs},
        mode=wandb_mode, reinit=True,
    )

    # -- data ------------------------------------------------------------------
    train_g, val_g, test_g = load_qg_graph_pairs(stage, seed)
    in_features = train_g[0].x.size(1)
    train_loader = build_pair_loader(train_g, cfg.batch_size, seed, shuffle=True)

    # -- model + loss ----------------------------------------------------------
    model = GraphEncoder(cfg, in_features).to(device)
    loss_fn = make_loss_fn(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage.epochs)

    n_params = count_parameters(model)
    n_quantum = count_parameters(model.head) if cfg.quantum_head not in ("none", "mlp_match") else 0
    wandb.log({"params/total": n_params, "params/quantum": n_quantum}, step=0)

    # -- training loop ---------------------------------------------------------
    t0 = time.time()
    for epoch in range(stage.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch_a, batch_b, labels in train_loader:
            batch_a = batch_a.to(device); batch_b = batch_b.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            z_a = model(batch_a.x.float(), batch_a.edge_index, batch_a.batch)
            z_b = model(batch_b.x.float(), batch_b.edge_index, batch_b.batch)
            try:
                loss = loss_fn(z_a, z_b)
            except TypeError:  # pair loss expects label arg
                loss = loss_fn(z_a, z_b, labels)
            loss.backward()
            # Gradient clipping — guards against barren-plateau-style gradient blowups.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()
        train_loss = running / max(1, n_batches)
        wandb.log({"train/loss": train_loss, "epoch": epoch + 1}, step=epoch + 1)

    # -- evaluation ------------------------------------------------------------
    acc, auc = linear_probe_auc(model, train_g, test_g, device)
    wall = time.time() - t0
    wandb.log({"eval/test_accuracy": acc, "eval/test_auc": auc, "wall_clock_s": wall})
    run.finish()

    return {"name": cfg.name, "seed": seed, "stage": stage.name,
            "test_accuracy": acc, "test_auc": auc, "params": n_params,
            "wall_clock_s": wall}


# -----------------------------------------------------------------------------
# Stage runner with selection
# -----------------------------------------------------------------------------

def run_stage(configs: List[ExperimentConfig], stage: StageConfig,
              wandb_project: str, mode: str = "online") -> Dict[str, Dict]:
    """Run every config across stage.seeds, aggregate by config name, return summary."""
    raw: Dict[str, List[Dict]] = {}
    for cfg in configs:
        raw.setdefault(cfg.name, [])
        for seed in stage.seeds:
            try:
                m = train_one(cfg, stage, seed, wandb_project, mode)
                raw[cfg.name].append(m)
            except Exception as e:
                print(f"[stage={stage.name}] {cfg.name} seed={seed} FAILED: {e}")
                raw[cfg.name].append({"name": cfg.name, "seed": seed, "error": str(e)})

    # Aggregate.
    summary: Dict[str, Dict] = {}
    for name, runs in raw.items():
        good = [r for r in runs if "error" not in r]
        if not good:
            summary[name] = {"mean_auc": float("nan"), "std_auc": float("nan"),
                             "mean_acc": float("nan"), "n_seeds": 0,
                             "errors": [r.get("error") for r in runs]}
            continue
        aucs = np.array([r["test_auc"] for r in good])
        accs = np.array([r["test_accuracy"] for r in good])
        summary[name] = {
            "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std(ddof=0) if len(aucs) > 1 else 0.0),
            "mean_acc": float(accs.mean()), "std_acc": float(accs.std(ddof=0) if len(accs) > 1 else 0.0),
            "n_seeds": len(good), "raw": good,
        }
    return summary


def select_finalists(summary: Dict[str, Dict], slack: float, k_min: int = 1) -> List[str]:
    """Promote any config whose mean AUC >= leader_mean - slack. Always promote at
    least k_min names so we never drop the entire batch."""
    valid = [(name, s["mean_auc"]) for name, s in summary.items()
             if not np.isnan(s["mean_auc"])]
    if not valid:
        return []
    leader = max(v for _, v in valid)
    promoted = [n for n, v in valid if v >= leader - slack]
    if len(promoted) < k_min:
        promoted = [n for n, _ in sorted(valid, key=lambda kv: -kv[1])[:k_min]]
    return promoted


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

def default_registry() -> List[ExperimentConfig]:
    """Return the default list of experiments. Edit this to focus the sweep."""
    base = dict(n_qubits=8, n_quantum_layers=3, embedding_dim=64,
                hidden_dims=(32, 32, 32), batch_size=128, lr=1e-3,
                margin=1.0, temperature=0.3, dropout=0.5, noise_track="A",
                small_init_scale=0.1)
    return [
        ExperimentConfig(name="G0_classical_baseline", quantum_head="none",
                         loss="pairs", **base, notes="anchor"),
        ExperimentConfig(name="G1_param_matched_mlp",  quantum_head="mlp_match",
                         loss="pairs", **base, notes="parameter-matched classical control"),
        ExperimentConfig(name="H1_qc1_pairs",          quantum_head="qc1",
                         loss="pairs", **base, notes="reproduces benchmarking"),
        ExperimentConfig(name="H1F_qc1_fidelity",      quantum_head="qc1",
                         loss="quantum_fidelity", **base, notes="fills 2x2 cell"),
        ExperimentConfig(name="H2_qc2_pairs",          quantum_head="qc2",
                         loss="pairs", **base, notes="fills 2x2 cell"),
        ExperimentConfig(name="H3_qc3_pairs",          quantum_head="qc3",
                         loss="pairs", **base, notes="reproduces benchmarking"),
        ExperimentConfig(name="H4_qc4_reuploading",    quantum_head="qc4",
                         loss="pairs", **base, notes="data re-uploading"),
        ExperimentConfig(name="H5_qc6_HEA",            quantum_head="qc6",
                         loss="pairs", **base, notes="hardware-efficient ansatz"),
        ExperimentConfig(name="H7_qc1_ntxent",         quantum_head="qc1",
                         loss="ntxent", **base, notes="SimCLR-style"),
        ExperimentConfig(name="H8_qc1_barlow",         quantum_head="qc1",
                         loss="barlow_twins", **base, notes="no-negatives"),
        ExperimentConfig(name="H9_qc1_vicreg",         quantum_head="qc1",
                         loss="vicreg", **base, notes="anti-collapse"),
        ExperimentConfig(name="H10_qc1_swap_fidelity", quantum_head="qc1",
                         loss="swap_fidelity", **base, notes="quantum metric learning"),
        ExperimentConfig(name="H11_qc1_noisyB",        quantum_head="qc1",
                         loss="pairs",
                         **{**base, "noise_track": "B"},
                         notes="depolarising + amplitude/phase damping"),
    ]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="qssl-experiments",
                        help="W&B project name")
    parser.add_argument("--mode", default="online",
                        choices=["online", "offline", "disabled"],
                        help="W&B mode")
    parser.add_argument("--max-stage", default="stage2_bench",
                        choices=list(STAGES.keys()),
                        help="last stage to run")
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these experiment names (defaults to all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without training")
    parser.add_argument("--out", default=str(REPO_ROOT / "experiment_results.json"),
                        help="where to write the aggregated summary")
    args = parser.parse_args()

    configs = default_registry()
    if args.only:
        configs = [c for c in configs if c.name in set(args.only)]
        if not configs:
            print(f"No experiments match {args.only}")
            sys.exit(2)

    if args.dry_run:
        print("Stages:")
        for s in STAGES.values():
            print(f"  {s.name}: n_train={s.n_train} epochs={s.epochs} seeds={s.seeds}")
        print(f"\n{len(configs)} configs:")
        for c in configs:
            print(f"  {c.name:30s}  head={c.quantum_head:10s} loss={c.loss:18s} "
                  f"noise={c.noise_track}  notes={c.notes}")
        return

    aggregated: Dict[str, Dict[str, Dict]] = {}
    surviving = configs
    for stage_name in ["stage0_smoke", "stage1_screen", "stage2_bench"]:
        stage = STAGES[stage_name]
        print(f"\n========== {stage.name} :: {len(surviving)} configs ==========")
        summary = run_stage(surviving, stage, args.project, args.mode)
        aggregated[stage.name] = summary

        # Print and select.
        for name, s in sorted(summary.items(), key=lambda kv: -(kv[1].get("mean_auc") or 0)):
            print(f"  {name:30s}  AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
                  f"acc={s['mean_acc']:.4f}  n={s['n_seeds']}")
        if stage_name == args.max_stage:
            break
        promoted = select_finalists(summary, slack=stage.promotion_slack, k_min=3)
        print(f"  -> promoting to next stage: {promoted}")
        surviving = [c for c in surviving if c.name in set(promoted)]
        if not surviving:
            print("No survivors — stopping.")
            break

    Path(args.out).write_text(json.dumps(aggregated, indent=2))
    print(f"\nWrote summary to {args.out}")


if __name__ == "__main__":
    main()
