#!/usr/bin/env python3
"""
B10 Metric Calibration Script  (v2 — AV-JEPA goal-aligned)
=============================================================
Pulls all B10 wandb runs, ranks (prior, pred) combos by individual metrics and
composite scores, then checks whether the metric-based ranking reproduces the
visual ground-truth from the 3D scatterplots.

Visual ground-truth (from human inspection of B10 plots):
  BEST:  prior:L1 + pred:L1   — flat wall, internally separated, semantics linearised
  GOOD:  prior:L2 + pred:L2   — ball shape, but linear semantic gradient (disentangled)
         prior:none + pred:L1  — aligned but arbitrary shape; pred is the key driver
  BAD:   prior:any + pred:none — no alignment; shape morphed toward prior but not flat
         prior:none + pred:none (baseline) — nothing
  KEY:   predictor is ALWAYS necessary; L1-pred > L2-pred geometrically

AV-JEPA goal alignment:
  Cross-modal predictability → val_align_a2b / val_align_b2a (lower = better).
    KEY finding: prior:none+pred:L1 has val_align ~4-6x higher than prior:L1+pred:L1
    despite looking visually aligned — the prior provides the canonical frame that
    makes g_{a2b} close to identity → directly AV-relevant.
  Semantic linearisation → r2_dim0_u1, r2_dim1_u2, r2_joint (linear probes + temporal pred)
  Manifold flatness → temporal dynamics ≈ linear shifts in latent space
  Shared/private disentanglement → r2_dim2_noise (cleaner cross-modal signal)
  Complementarity → r2_joint - max(r2_a, r2_b) (fusion > unimodal)
  NOTE: diagonality_ratio is MISLEADING — highest for pred:none combos due to
    trivial collapse; removed from composite scores.

Run:  python calibrate_metrics.py
Output: prints tables to stdout + saves B10_calibration.csv
"""

import re
import yaml
import wandb
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT_DIR     = SCRIPT_DIR.parent.parent.parent.parent.parent
cfg_dir      = SCRIPT_DIR.parent.parent.parent / "cfgs" / "B10_3DtoHD_fromB08"
registry_file = SCRIPT_DIR.parent.parent / "metrics_registry.yaml"
OUT_CSV      = SCRIPT_DIR / "B10_calibration.csv"

with open(registry_file) as f:
    registry = yaml.safe_load(f).get("metrics", {})

# ── WandB query ───────────────────────────────────────────────────────────────
api      = wandb.Api()
entity   = "robertkabai-um"
project  = "eb_jepa"
batch_id = "B10_3DtoHD_fromB08"

print(f"Querying wandb [{entity}/{project}] for tag [{batch_id}]...")
runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": [batch_id]}})

cfg_to_runs: dict[str, list] = {}
for r in runs:
    cfg_tag = next((t for t in r.tags if t.startswith("B10_") and ("_N" in tag or "_M" in tag or "_R" in tag)), None)
    if cfg_tag:
        cfg_to_runs.setdefault(cfg_tag, []).append(r)

print(f"  Found {len(cfg_to_runs)} configs.\n")

# ── Noise / prior / pred label maps ───────────────────────────────────────────
noise_map = {
    "1": "N1-Ext10",
    "2": "N2-Ext30",
}
prior_map = {"0": "none", "1": "L1", "2": "L2"}
pred_map  = {"0": "none", "1": "L1", "2": "L2"}

# ── Metrics of interest ───────────────────────────────────────────────────────
# Grouped by what aspect of the "good model" they should capture.
# Higher = better unless marked (lower_better=True below).
METRICS_HIGHER_BETTER = [
    "clean_flatness_ratio_a",   # flat manifold — pred:L1+prior:L1 → ~1.0
    "clean_flatness_ratio_b",
    "r2_joint",                  # semantic factor recovery
    "r2_a",
    "r2_b",
    "r2_dim0_u1",               # per-factor recovery
    "r2_dim1_u2",
    "diagonality_ratio",         # axis alignment
    "cca_diag_score",
    "retrieval_cos@1",           # cross-modal alignment
    "retrieval_l2@1",
    "retrieval_cos@5",
    "retrieval_l2@5",
    "cca_dim0",
    "cca_dim1",
]
METRICS_LOWER_BETTER = [
    "r2_dim2_noise",            # noise leakage — lower = better isolation
    "clean_orth_residual_a",    # curvature residual
    "clean_orth_residual_b",
    "val_align_a2b",            # MSE: alignment quality
    "val_align_b2a",
]
ALL_METRICS = METRICS_HIGHER_BETTER + METRICS_LOWER_BETTER

# ── Build rows ─────────────────────────────────────────────────────────────────
rows = []
for cfg_file in sorted(cfg_dir.glob("B10_[RM]*.yaml")):
    cfg_name = cfg_file.stem
    m = re.search(r"B10_([RM])(\d+)_N(\d)P(\d)(\d)", cfg_name)
    if not m:
        continue
    embed_type, dim, n_idx, p1_idx, p2_idx = m.groups()

    # Merge train + eval summaries
    merged: dict = {}
    cfg_runs = cfg_to_runs.get(cfg_name, [])
    train_run = eval_run = None
    for r in cfg_runs:
        if "eval" in r.tags:
            if eval_run is None or r.created_at > eval_run.created_at:
                eval_run = r
        else:
            if train_run is None or r.created_at > train_run.created_at:
                train_run = r
    if train_run:
        merged.update(dict(train_run.summary))
    if eval_run:
        merged.update(dict(eval_run.summary))

    row = {
        "config": cfg_name,
        "noise":  noise_map.get(n_idx, n_idx),
        "prior":  prior_map.get(p1_idx, p1_idx),
        "pred":   pred_map.get(p2_idx, p2_idx),
        "combo":  f"prior:{prior_map.get(p1_idx,'?')}+pred:{pred_map.get(p2_idx,'?')}",
        "state":  (train_run.state if train_run else "missing"),
    }
    for alias in ALL_METRICS:
        wandb_key = registry.get(alias)
        if wandb_key:
            val = merged.get(wandb_key)
            row[alias] = float(val) if isinstance(val, (int, float)) else np.nan
        else:
            row[alias] = np.nan

    rows.append(row)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} config rows. States: {df['state'].value_counts().to_dict()}\n")

# ── Composite scores (v2 — AV-JEPA goal-aligned) ─────────────────────────────
# All sub-scores min-max normalised (0=worst, 1=best within B10 range).
# KEY CHANGE: val_align (cross-modal MSE, lower=better) is now primary.
# diagonality_ratio removed — highest for pred:none (trivial collapse).

def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-12)

def neg_minmax(s: pd.Series) -> pd.Series:
    """Normalise lower-is-better metric so that 1=best."""
    lo, hi = s.min(), s.max()
    return 1.0 - (s - lo) / (hi - lo + 1e-12)

# Cross-modal predictor MSE (mean of both directions) — DIRECTLY AV-relevant
cross_mse = (df["val_align_a2b"] + df["val_align_b2a"]) / 2.0

# Semantic factor recovery (linear probe proxy)
factor_r2 = (
    df["r2_joint"] * 0.40
    + df["r2_dim0_u1"] * 0.30
    + df["r2_dim1_u2"] * 0.30
)

# Noise isolation (shared/private disentanglement)
isolation = 1.0 - df["r2_dim2_noise"].clip(0, 1)

# Manifold structure: flatness × (1 - curvature_residual_normalised)
# Multiplicative: rewards configs that do BOTH well, not just one
flatness_n = minmax(df["clean_flatness_ratio_a"])
curvature_n = neg_minmax(df["clean_orth_residual_a"])
manifold_quality = flatness_n * curvature_n  # in [0,1]×[0,1]

# Complementarity: how much does joint representation add over best single modality?
df["joint_gain"] = df["r2_joint"] - df[["r2_a", "r2_b"]].max(axis=1)

# ── S_AV: Primary AV-JEPA score ───────────────────────────────────────────────
# Design: cross-modal MSE is weighted heaviest (direct AV-predictor quality),
# then factor recovery (linear probes / temporal prediction), then structure.
df["S_AV"] = (
    neg_minmax(cross_mse) * 0.40          # AV predictor quality (primary)
    + minmax(factor_r2) * 0.35            # semantic linearisation
    + minmax(manifold_quality) * 0.15     # flatness × curvature (joint reward)
    + minmax(isolation) * 0.10            # private/shared disentanglement
)

# ── S_structure: Manifold shape quality (for sanity check) ────────────────────
# What the visual assessment was most sensitive to: shape + semantic directions.
df["S_structure"] = (
    minmax(df["clean_flatness_ratio_a"]) * 0.30
    + minmax(factor_r2) * 0.40
    + neg_minmax(df["clean_orth_residual_a"]) * 0.20
    + minmax(isolation) * 0.10
)

# ── S_old (v1 for comparison): previous composite without val_align ────────────
df["S_old"] = (
    minmax(df["clean_flatness_ratio_a"]) * 0.20
    + minmax(df["r2_joint"]) * 0.20
    + minmax(df["retrieval_l2@1"]) * 0.20
    + minmax(df["diagonality_ratio"]) * 0.15
    + neg_minmax(df["r2_dim2_noise"]) * 0.10
    + neg_minmax(df["clean_orth_residual_a"]) * 0.10
    + minmax(df["r2_dim0_u1"]) * 0.025
    + minmax(df["r2_dim1_u2"]) * 0.025
)

SCORES = ["S_AV", "S_structure", "S_old"]

# ── Helper: ranked combo table ────────────────────────────────────────────────
def combo_rank_table(data: pd.DataFrame, score_col: str, title: str) -> pd.DataFrame:
    """Group by (prior, pred) combo, mean over noise regimes, rank descending."""
    grp = data.groupby("combo")[score_col].mean().sort_values(ascending=False)
    tbl = grp.reset_index()
    tbl.columns = ["combo", "mean_score"]
    tbl["rank"] = range(1, len(tbl) + 1)
    return tbl

# ── Individual metric table (mean over all noise, per combo) ──────────────────
print("=" * 80)
print("SECTION 1 — Per-metric means by (prior, pred) combo  [averaged over all noise]")
print("=" * 80)

key_metrics = [
    "clean_flatness_ratio_a", "r2_joint", "r2_dim0_u1", "r2_dim1_u2",
    "retrieval_l2@1", "diagonality_ratio", "r2_dim2_noise",
    "clean_orth_residual_a",
]

metric_means = df.groupby("combo")[key_metrics].mean()
# Re-order columns sensibly
print("\nMean metric values per combo (sorted by clean_flatness_ratio_a):")
sorted_tbl = metric_means.sort_values("clean_flatness_ratio_a", ascending=False)
with pd.option_context("display.float_format", "{:.4f}".format,
                        "display.max_columns", 20,
                        "display.width", 200):
    print(sorted_tbl.to_string())

# ── Composite score rankings ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2 — Composite score rankings (averaged over all noise regimes)")
print("=" * 80)
print("""
Scores (v2):
  S_AV       = 0.40*val_align + 0.35*factor_R² + 0.15*flatness×curvature + 0.10*isolation
               PRIMARY: AV-JEPA cross-modal predictability goal
  S_structure= 0.30*flatness + 0.40*factor_R² + 0.20*curvature + 0.10*isolation
               Captures what visual inspection was sensitive to
  S_old      = v1 composite (diagonality+retrieval+flatness) — baseline comparison
""")

rank_tables = {}
for s in SCORES:
    tbl = combo_rank_table(df, s, s)
    rank_tables[s] = tbl
    print(f"\n  [{s}] ranking:")
    print(tbl.to_string(index=False))

# ── Consensus ranking (average rank across all 4 scores) ──────────────────────
print("\n" + "=" * 80)
print("SECTION 3 — Consensus ranking (mean rank across S1–S4)")
print("=" * 80)
all_combos = df["combo"].unique()
consensus = pd.DataFrame({"combo": all_combos})
for s in SCORES:
    tbl = rank_tables[s][["combo", "rank"]].rename(columns={"rank": f"rank_{s}"})
    consensus = consensus.merge(tbl, on="combo", how="left")
rank_cols = [f"rank_{s}" for s in SCORES]
consensus["mean_rank"] = consensus[rank_cols].mean(axis=1)
consensus = consensus.sort_values("mean_rank")
print(consensus.to_string(index=False))

# ── Per-noise regime breakdown ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4 — S_AV ranking per noise regime (does ranking vary with noise?)")
print("=" * 80)
for noise_val in sorted(df["noise"].unique()):
    sub = df[df["noise"] == noise_val]
    tbl = combo_rank_table(sub, "S_AV", noise_val)
    top4 = tbl.head(4)["combo"].tolist()
    print(f"  {noise_val}: TOP4 = {top4}")

# ── Does pred matter? ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5 — Predictor effect: mean S_AV by pred/prior (averaged over noise)")
print("=" * 80)
pred_effect = df.groupby("pred")["S_AV"].mean().sort_values(ascending=False)
print("Mean S_AV by pred type:")
print(pred_effect.to_string())

print("\nMean S_AV by prior type:")
prior_effect = df.groupby("prior")["S_AV"].mean().sort_values(ascending=False)
print(prior_effect.to_string())

# ── AV-critical: cross-modal MSE per combo ────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 7 — Cross-modal MSE per combo [val_align_a2b + val_align_b2a mean]")
print("  KEY AV metric: lower = predictor can actually do cross-modal prediction")
print("=" * 80)
cross_mse_combo = df.groupby("combo")[["val_align_a2b", "val_align_b2a"]].mean()
cross_mse_combo["mean_cross_mse"] = (cross_mse_combo["val_align_a2b"] + cross_mse_combo["val_align_b2a"]) / 2
cross_mse_combo = cross_mse_combo.sort_values("mean_cross_mse")
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(cross_mse_combo.to_string())

# ── AV-critical: joint gain per combo ─────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 8 — Joint gain per combo: r2_joint - max(r2_a, r2_b)")
print("  Measures: how much does fusing 2 modalities add over best single modality?")
print("=" * 80)
joint_gain_combo = df.groupby("combo")[["r2_joint", "r2_a", "r2_b", "joint_gain"]].mean()
joint_gain_combo = joint_gain_combo.sort_values("joint_gain", ascending=False)
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(joint_gain_combo.to_string())

# ── Cross-check: visual ground truth rank ─────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6 — Agreement check with visual ground truth (per score)")
print("=" * 80)
VISUAL_GT = [
    "prior:L1+pred:L1",     # 1st: flat wall, disentangled, semantics linearised
    "prior:L2+pred:L2",     # 2nd: ball, linear semantic gradient
    "prior:none+pred:L1",   # 3rd: aligned, arbitrary shape
    "prior:L1+pred:L2",     # 4th
    "prior:L2+pred:L1",     # 5th
    "prior:none+pred:L2",   # 6th: aligned, weaker than L1-pred
    "prior:L1+pred:none",   # 7th: shape only
    "prior:L2+pred:none",   # 7th: shape only
    "prior:none+pred:none", # 9th: nothing
]
gt_ranks_map = {c: i + 1 for i, c in enumerate(VISUAL_GT)}

from scipy.stats import spearmanr
for score_col in SCORES:
    tbl = combo_rank_table(df, score_col, score_col)
    order = tbl["combo"].tolist()
    gt_present = [c for c in VISUAL_GT if c in order]
    gtr = [gt_ranks_map[c] for c in gt_present]
    mtr = [order.index(c) + 1 for c in gt_present]
    rho, pval = spearmanr(gtr, mtr)
    marker = "✅" if rho > 0.7 else ("⚠️ " if rho > 0.4 else "❌")
    print(f"  {marker} {score_col}: Spearman ρ = {rho:.3f} (p={pval:.3f})")
    print(f"     Metric order: {order}")

print("\n  Visual GT:")
print(f"     {VISUAL_GT}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\nFull data saved → {OUT_CSV}")
print("Done.")
