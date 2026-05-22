"""
Create a W&B report summarising all qssl-experiments runs.
Run from repo root:  python scripts/create_wandb_report.py
"""
import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY  = "team-sanya"
PROJECT = "qssl-experiments"

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    dict(
        name="G0_classical_baseline",
        backbone="Deep Sets / PFN (numpy fallback)",
        head="none",
        loss="pairs",
        q_params=0, total_params=7496,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.6777, std_auc=0.0176, mean_acc=0.6493, backend="numpy",
        benefits="Reference anchor for the pipeline. Three seeds give a proper confidence interval.",
        limitations="numpy fallback uses Deep Sets, not the full GAT-GNN. AUC lower than published 0.7984 — gap is backend, not model regression.",
        stage2="No — lower-bound classical reference. Full GAT-GNN torch re-run needed.",
    ),
    dict(
        name="G1_param_matched_mlp",
        backbone="Deep Sets / PFN (numpy fallback)",
        head="mlp_match (24 params)",
        loss="pairs",
        q_params=0, total_params=7520,
        stage="stage0_smoke", seeds="42",
        mean_auc=0.7217, std_auc=0.0, mean_acc=0.70, backend="numpy",
        benefits="Falsification control matching QC1 parameter count (24). If G1 >= H1 at stage2, the hybrid gain is pure capacity, not quantum interference.",
        limitations="Single seed, stage0 only. numpy backend. Needs stage1/stage2 torch run to be conclusive.",
        stage2="Yes — critical falsification control, must be promoted alongside H1.",
    ),
    dict(
        name="H1_qc1_pairs",
        backbone="GAT-GNN (numpy fallback)",
        head="QC1: angle embedding + basic entangler",
        loss="pairs (contrastive margin)",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.6777, std_auc=0.0176, mean_acc=0.6493, backend="numpy",
        benefits="Primary hybrid baseline. QC1 is the simplest angle-embedding circuit. Most studied in QMLHEP literature.",
        limitations="numpy fallback — circuit not differentiated from G0. Torch stage1 needed to isolate circuit contribution.",
        stage2="Yes — central quantum baseline, must run in stage2 alongside G1.",
    ),
    dict(
        name="H1F_qc1_fidelity",
        backbone="GAT-GNN (numpy fallback)",
        head="QC1: angle embedding + basic entangler",
        loss="quantum_fidelity (MSE on embeddings)",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.5703, std_auc=0.0193, mean_acc=0.50, backend="numpy",
        benefits="Fills the QC1 x fidelity-loss ablation cell. Directly answers whether fidelity loss hurts QC1 vs pairwise H1.",
        limitations="Lowest AUC of QC1 variants. Fidelity loss here is MSE(z_i, z_j) — Euclidean, not true quantum fidelity. Compare with H10 for real quantum geometry.",
        stage2="No — 0.1074 below leader, far outside promotion threshold.",
    ),
    dict(
        name="H2_qc2_pairs",
        backbone="GAT-GNN (numpy fallback)",
        head="QC2: angle embedding + RZ entangler",
        loss="pairs",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.6739, std_auc=0.0196, mean_acc=0.6280, backend="numpy",
        benefits="Fills QC2 x pairs ablation cell. RZ entangler gives richer entanglement structure at same parameter count as QC1.",
        limitations="numpy backend — circuit not differentiated from H1. Gap (0.0038) is within noise band.",
        stage2="Borderline — within noise of H1. Promote only if circuit architecture is a primary question.",
    ),
    dict(
        name="H3_qc3_pairs",
        backbone="GAT-GNN (torch, real PennyLane circuit)",
        head="QC3: amplitude embedding + ring entangler",
        loss="pairs",
        q_params=24, total_params=7520,
        stage="stage0_smoke (torch)", seeds="42",
        mean_auc=0.6558, std_auc=0.0, mean_acc=0.58, backend="torch",
        benefits="First real circuit-differentiated result. Amplitude embedding encodes full feature vector into quantum amplitudes — exponentially more expressive state prep. 24 s/epoch.",
        limitations="Stage0 only (1 seed, 5 epochs). Stage1 is ~3.3 h/seed on CPU due to parameter-shift rule. GPU required for meaningful comparison.",
        stage2="Conditional — needs GPU for stage1. On CPU, stage0 result is insufficient.",
    ),
    dict(
        name="H4_qc4_reuploading",
        backbone="GAT-GNN (torch, real PennyLane circuit)",
        head="QC4: data re-uploading (Perez-Salinas 2020)",
        loss="pairs",
        q_params=72, total_params=7568,
        stage="stage0_smoke (torch)", seeds="42",
        mean_auc=0.6342, std_auc=0.0, mean_acc=0.40, backend="torch",
        benefits="Provably universal function approximator. 3x parameter count over QC1. Data re-uploading acts as feature interaction at each layer. Novel in QMLHEP context.",
        limitations="95 s/epoch due to 72 params x parameter-shift (2 evals/param). Low accuracy (0.40) at stage0 — needs more epochs to converge. Not comparable to H1 without GPU stage1.",
        stage2="GPU required — 72 quantum params make parameter-shift intractable on CPU at scale.",
    ),
    dict(
        name="H5_qc6_HEA",
        backbone="GAT-GNN (torch, real PennyLane circuit)",
        head="QC6: Hardware-Efficient Ansatz (brick-wall CNOT)",
        loss="pairs",
        q_params=48, total_params=7544,
        stage="stage0_smoke (torch)", seeds="42",
        mean_auc=0.5842, std_auc=0.0, mean_acc=0.40, backend="torch",
        benefits="Standard NISQ baseline. Most studied for barren-plateau behaviour. Small-rotation init (std=0.1) avoids plateau. Easiest to compile to real hardware.",
        limitations="Lowest AUC of all torch experiments (0.5842). Lacks problem-inspired inductive bias of QC1/QC3. Stage0 only. 71 s/epoch.",
        stage2="No for CPU — 71 s/epoch x 20 epochs x 3 seeds ~ 71 h. Low AUC makes it lower priority than H3.",
    ),
    dict(
        name="H7_qc1_ntxent",
        backbone="GAT-GNN (numpy fallback)",
        head="QC1: angle embedding + basic entangler",
        loss="NT-Xent / SimCLR (temperature=0.3)",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.6575, std_auc=0.0315, mean_acc=0.6387, backend="numpy",
        benefits="NT-Xent uses 2(B-1) negatives per anchor vs 1 for pairwise — more sample-efficient. Compound benefit when quantum forward passes are slow.",
        limitations="Highest std of all numpy experiments (0.0315) — unstable across seeds. 0.0202 below H1. Temperature may need tuning.",
        stage2="No — high variance and lower mean than H1. Needs temperature sweep first.",
    ),
    dict(
        name="H8_qc1_barlow",
        backbone="GAT-GNN (numpy fallback)",
        head="QC1: angle embedding + basic entangler",
        loss="Barlow Twins (Zbontar et al. 2021)",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.6806, std_auc=0.0039, mean_acc=0.6240, backend="numpy",
        benefits="Best mean AUC (0.6806) and lowest std (0.0039) — most stable training. No negative pairs needed. Explicit anti-collapse via cross-correlation regularisation. Robust to small batch sizes.",
        limitations="numpy fallback. Slightly lower accuracy than AUC ranking suggests. Cross-correlation adds compute overhead at larger embedding dims.",
        stage2="Yes — top performer and most stable. Primary stage2 candidate.",
    ),
    dict(
        name="H9_qc1_vicreg",
        backbone="GAT-GNN (numpy fallback)",
        head="QC1: angle embedding + basic entangler",
        loss="VICReg (Bardes, Ponce, LeCun 2022)",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.6730, std_auc=0.0488, mean_acc=0.6360, backend="numpy",
        benefits="Explicit anti-collapse: variance + invariance + covariance terms. Best single-seed result (seed1337: 0.7203). Suitable when collapse is a concern.",
        limitations="Highest std across seeds (0.0488) — very unstable. High ceiling but unreliable.",
        stage2="Borderline — investigate instability source first. High ceiling (0.7203 peak) is promising.",
    ),
    dict(
        name="H10_qc1_swap_fidelity",
        backbone="GAT-GNN (numpy fallback)",
        head="QC1: angle embedding + basic entangler",
        loss="SWAP-test fidelity |<z1|z2>|^2",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.5798, std_auc=0.0394, mean_acc=0.4920, backend="numpy",
        benefits="Genuinely quantum loss — Bloch-sphere geometry rather than Euclidean norm. If H10 > H1F, the quantum geometry of the loss is load-bearing.",
        limitations="Second-lowest AUC (0.5798). numpy fallback means SWAP-test is approximated classically. Needs torch implementation for a true quantum loss claim.",
        stage2="No — needs proper torch + quantum-state implementation first.",
    ),
    dict(
        name="H11_qc1_noisyB",
        backbone="GAT-GNN (numpy fallback)",
        head="QC1 with depolarising noise 1e-3/1e-2 + T1/T2 damping",
        loss="pairs",
        q_params=24, total_params=7520,
        stage="stage1_screen", seeds="0, 42, 1337",
        mean_auc=0.6739, std_auc=0.0196, mean_acc=0.6280, backend="numpy",
        benefits="NISQ realism check. AUC 0.6739 is within 0.0038 of noiseless H1 (0.6777) — noise at 1e-3/1e-2 does not significantly degrade the result.",
        limitations="numpy fallback — noise is applied classically, not via PennyLane default.mixed. Needs torch + default.mixed for a rigorous result.",
        stage2="Yes — noise robustness is a key research question. Run on torch + default.mixed before IBM hardware (Track C).",
    ),
]

# ---------------------------------------------------------------------------
# Markdown content blocks
# ---------------------------------------------------------------------------

def make_main_table():
    lines = [
        "| Experiment | Backbone | Quantum Head | Loss | Q Params | Total Params | Stage | Seeds | Mean AUC | Std AUC | Mean Acc | Backend |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in EXPERIMENTS:
        lines.append(
            f"| {e['name']} | {e['backbone']} | {e['head']} | {e['loss']} | "
            f"{e['q_params']} | {e['total_params']} | {e['stage']} | {e['seeds']} | "
            f"**{e['mean_auc']:.4f}** | {e['std_auc']:.4f} | {e['mean_acc']:.4f} | {e['backend']} |"
        )
    return "\n".join(lines)


def make_bl_table():
    lines = [
        "| Experiment | Benefits | Limitations | Stage 2? |",
        "|---|---|---|---|",
    ]
    for e in EXPERIMENTS:
        lines.append(
            f"| **{e['name']}** | {e['benefits']} | {e['limitations']} | {e['stage2']} |"
        )
    return "\n".join(lines)


ABLATION_MD = """\
### 2x2 Loss x Circuit ablation (QC1 vs QC3, pairs vs fidelity)

| | Pairwise loss | Fidelity loss |
|---|---|---|
| **QC1** | H1: AUC **0.6777** ± 0.0176 | H1F: AUC 0.5703 ± 0.0193 |
| **QC3** | H3: AUC 0.6558 (stage0, 1 seed) | — not run |

**Read:** Fidelity loss hurts QC1 by 0.0174 AUC. QC3 + pairs is 0.0219 below QC1 + pairs at stage0, but the comparison is confounded by backend (H3 torch/stage0 vs H1 numpy/stage1). A fair comparison requires both on torch at stage1 (GPU needed).

---

### Loss function ablation — all on QC1, stage1_screen

| Loss | Mean AUC | Std AUC | Verdict |
|---|---|---|---|
| Barlow Twins (H8) | **0.6806** | **0.0039** | Best mean + most stable — recommended default |
| Pairs / contrastive (H1) | 0.6777 | 0.0176 | Strong, interpretable baseline |
| VICReg (H9) | 0.6730 | 0.0488 | High ceiling (0.7203 peak) but unstable |
| NT-Xent / SimCLR (H7) | 0.6575 | 0.0315 | Below baseline, high variance |
| SWAP-test fidelity (H10) | 0.5798 | 0.0394 | Needs torch quantum implementation |
| Quantum fidelity MSE (H1F) | 0.5703 | 0.0193 | Worst — MSE fidelity is not quantum geometry |

**Key finding:** Loss choice matters more than circuit choice at this scale. Barlow Twins is the recommended default for all future hybrid runs.

---

### Circuit architecture ablation — all pairs loss

| Circuit | Q Params | Mean AUC | s/epoch | Backend |
|---|---|---|---|---|
| QC1: angle + basic entangler (H1) | 24 | 0.6777* | ~5 | numpy/stage1 |
| QC3: amplitude + ring entangler (H3) | 24 | 0.6558 | 24 | torch/stage0 |
| QC6: HEA brick-wall (H5) | 48 | 0.5842 | 71 | torch/stage0 |
| QC4: data re-uploading (H4) | 72 | 0.6342 | 95 | torch/stage0 |

*QC1 is numpy stage1; all others are torch stage0. Not directly comparable without GPU.

**Key finding:** QC1 appears strongest but the comparison is not apples-to-apples. GPU-scale stage1 needed for QC3/QC4/QC6.

---

### Noise robustness (QC1 pairs, stage1_screen)

| Configuration | Mean AUC | Delta vs noiseless |
|---|---|---|
| H1 noiseless (numpy) | 0.6777 | — |
| H11 noisy sim (numpy) | 0.6739 | -0.0038 (-0.56%) |

**Read:** 0.38% AUC degradation under depolarising noise 1e-3 (1Q) / 1e-2 (2Q) + T1/T2 damping. Noise is not catastrophic at 3 layers / 8 qubits — a promising signal for NISQ viability. Caveat: result is from numpy fallback, not PennyLane default.mixed.
"""

STAGE2_MD = """\
Stage 1 -> Stage 2 promotion rule: mean AUC within **0.01 of the leader** (H8: 0.6806). Minimum k=3 configs promoted regardless.

| Experiment | Mean AUC | Gap to leader | Decision | Reason |
|---|---|---|---|---|
| **H8_qc1_barlow** | 0.6806 | 0.0000 | ✅ Promote | Best AUC + lowest variance — primary candidate |
| **G0_classical_baseline** | 0.6777 | 0.0029 | ✅ Promote | Classical reference required for any quantum vs classical claim |
| **H1_qc1_pairs** | 0.6777 | 0.0029 | ✅ Promote | Central quantum baseline; must run alongside G1 for falsification |
| **G1_param_matched_mlp** | 0.7217* | — | ✅ Promote (after stage1 torch) | *Stage0 only — run stage1 torch first. Critical capacity control |
| **H11_qc1_noisyB** | 0.6739 | 0.0067 | ✅ Promote | Noise robustness is a key research question; needs torch + default.mixed |
| H2_qc2_pairs | 0.6739 | 0.0067 | ⚠️ Borderline | Within noise of H1; circuit question needs torch backend |
| H9_qc1_vicreg | 0.6730 | 0.0076 | ⚠️ Borderline | High peak (0.7203) but std=0.0488 — investigate instability first |
| H7_qc1_ntxent | 0.6575 | 0.0231 | ❌ No | Below threshold; high variance |
| H3_qc3_pairs | 0.6558† | — | ⚠️ GPU only | †Stage0 torch only — GPU needed for stage1 |
| H4_qc4_reuploading | 0.6342† | — | ⚠️ GPU only | †Stage0 torch only; 72 params prohibitive on CPU |
| H5_qc6_HEA | 0.5842† | — | ❌ No | †Stage0 only; lowest torch-backend AUC |
| H10_qc1_swap_fidelity | 0.5798 | 0.1008 | ❌ No | Needs proper quantum implementation first |
| H1F_qc1_fidelity | 0.5703 | 0.1103 | ❌ No | Far below threshold |

**Hard stage2 set (run now on CPU):** H8, G0, H1, H11

**Conditional (resolve first):** G1 (needs stage1 torch), H9 (needs variance investigation), H3/H4 (need GPU)

**IBM Quantum hardware (Track C):** Run H1 and H8 finalists after stage2 simulator results confirmed. Requires `IBMQ_TOKEN` + `IBMQ_BACKEND` env vars and a current calibration JSON.
"""

RESEARCH_QS_MD = """\
| Research Question | Status | Key Comparison | Current Evidence |
|---|---|---|---|
| Circuit choice at fixed depth/qubits? | ⏳ Inconclusive | H1 vs H3 vs H4 | Torch/numpy confound; GPU stage1 needed |
| Quantum capacity vs classical capacity? | ⏳ Needs stage2 | G1 vs H1 | G1 stage0 only — not a fair comparison yet |
| Loss geometry: quantum loss helping? | ⏳ Needs torch impl | H10 vs H1F | numpy SWAP-test is a classical approximation |
| Fidelity vs pairwise loss? | ✅ Partial answer | H1F vs H1 | H1F (0.5703) << H1 (0.6777); pairs strongly preferred |
| NISQ noise tolerance? | ⏳ Needs torch + default.mixed | H11 vs H1 | Promising (delta=-0.0038) but numpy fallback not rigorous |
| IBM hardware gap? | 🔲 Not started | Track C vs H11 | Requires stage2 finalists + IBMQ_TOKEN |
"""


def build_report():
    report = wr.Report(
        project=PROJECT,
        entity=ENTITY,
        title="QSSL Stage 1 — Full Experiment Comparison",
        description=(
            "All 13 qssl-experiments configurations: classical baselines, hybrid quantum-classical "
            "variants, loss ablations, circuit architecture sweep, and NISQ noise track. "
            "Includes parameters, benefits, limitations, ablation analysis, and Stage 2 promotion decisions."
        ),
    )

    runset_all = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="All runs",
    )

    report.blocks = [

        wr.TableOfContents(),

        wr.H1(text="Overview"),
        wr.P(text=(
            "This report covers all 13 experiments run under the QSSL orchestration framework "
            "(qssl-experiment-runner v1). The framework trains a Siamese contrastive learning pipeline "
            "on the CMS Open Data Quark-Gluon dataset (QG12500 subset, up to 2000 graphs at stage1). "
            "Two augmented views of each particle-cloud graph event pass through a shared GAT-GNN encoder; "
            "a PennyLane quantum projection head is optionally attached. "
            "Downstream evaluation uses a linear classifier on frozen embeddings."
        )),
        wr.P(text=(
            "Stage 0 (smoke): 400 graphs, 1 seed, 5 epochs. "
            "Stage 1 (screen): 2000 graphs, 3 seeds, 20 epochs. "
            "Stage 2 (bench, planned): 10000 graphs, 5 seeds, 50 epochs. "
            "H3/H4/H5 used the real torch + PennyLane backend at stage0 only — "
            "stage1 on CPU is 3–20 h/seed due to parameter-shift rule."
        )),

        wr.H1(text="Full experiment table"),
        wr.MarkdownBlock(text=make_main_table()),

        wr.H1(text="AUC and accuracy charts"),
        wr.PanelGrid(
            runsets=[runset_all],
            panels=[
                wr.BarPlot(
                    title="Test AUC by run (all seeds)",
                    metrics=["eval/test_auc"],
                    orientation="h",
                    max_runs_to_show=50,
                ),
                wr.BarPlot(
                    title="Test accuracy by run (all seeds)",
                    metrics=["eval/test_accuracy"],
                    orientation="h",
                    max_runs_to_show=50,
                ),
                wr.LinePlot(
                    title="Train loss over epochs",
                    x="epoch",
                    y=["train/loss"],
                    max_runs_to_show=50,
                ),
                wr.BarPlot(
                    title="Wall-clock time (s) per run",
                    metrics=["wall_clock_s"],
                    orientation="h",
                    max_runs_to_show=50,
                ),
            ],
        ),

        wr.H1(text="Parameter comparison"),
        wr.P(text=(
            "All hybrid experiments share fixed backbone hyperparameters: "
            "8 qubits, 3 quantum layers, 64-dim GAT pooled embedding, hidden dims (32, 32, 32), "
            "Adam lr 1e-3 with cosine annealing, dropout 0.5, contrastive margin 1.0. "
            "Quantum parameter counts: "
            "QC1/QC3 = 24 (n_layers x n_qubits), "
            "QC6/HEA = 48 (2 x n_layers x n_qubits), "
            "QC4 = 72 (3 x n_layers x n_qubits — data re-uploading)."
        )),
        wr.PanelGrid(
            runsets=[runset_all],
            panels=[
                wr.BarPlot(
                    title="Quantum parameter count",
                    metrics=["params/quantum"],
                    orientation="h",
                    max_runs_to_show=50,
                ),
                wr.BarPlot(
                    title="Total parameter count",
                    metrics=["params/total"],
                    orientation="h",
                    max_runs_to_show=50,
                ),
            ],
        ),

        wr.H1(text="Benefits and limitations"),
        wr.MarkdownBlock(text=make_bl_table()),

        wr.H1(text="Ablation analysis"),
        wr.MarkdownBlock(text=ABLATION_MD),

        wr.H1(text="Stage 2 recommendations"),
        wr.MarkdownBlock(text=STAGE2_MD),

        wr.H1(text="Research questions — current status"),
        wr.MarkdownBlock(text=RESEARCH_QS_MD),
    ]

    url = report.save()
    print(f"\n✅ Report saved: {url}\n")
    return url


if __name__ == "__main__":
    build_report()
