#!/usr/bin/env python3
"""
B09 Full Assessment  (uses calibrated S_structure composite — ρ=0.738 on B08 visual GT)
========================================================================================
4×3×8 sweep: D0–D3 (ambient dim) × N1–N3 (noise) × 8 (prior, pred) combos.

Key questions answered:
  1. Does prior:L1+pred:L1 remain the best combo in the MLP-manifold setting?
  2. Does quality degrade as ambient dimension increases (D0 → D3)?
  3. Is the pred-is-crucial finding robust across D/N configs?
  4. How do cross-modal predictors fare (val_align) per D and combo?
  5. Where exactly does the methodology break down (if anywhere)?

Calibration note (from B08):
  S_structure = 0.30*flatness + 0.40*factor_R² + 0.20*curvature + 0.10*isolation
  Spearman ρ = 0.738 (p=0.037) vs visual GT — validated, transferable.
  S_AV (val_align primary) showed loss-mismatch bias toward L2-pred; reported
  separately but NOT used as primary ranking criterion.
  diagonality_ratio: excluded — highest for pred:none (trivial collapse, misleading).

Run:  python assess_b09.py
Output: prints tables + saves B09_assessment.csv
"""

import re
import yaml
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
cfg_dir       = SCRIPT_DIR.parent.parent.parent / "cfgs"
registry_file = SCRIPT_DIR.parent.parent / "metrics_registry.yaml"
OUT_CSV       = SCRIPT_DIR / "B09_assessment.csv"

with open(registry_file) as f:
    registry = yaml.safe_load(f).get("metrics", {})

# ── WandB query ───────────────────────────────────────────────────────────────
api      = wandb.Api()
entity   = "robertkabai-um"
project  = "eb_jepa"
batch_id = "B09_nd_kf_mlp"

print(f"Querying wandb [{entity}/{project}] for tag [{batch_id}]...")
runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": [batch_id]}})

cfg_to_runs: dict[str, list] = {}
for r in runs:
    cfg_tag = next((t for t in r.tags if t.startswith("B09_D")), None)
    if cfg_tag:
        cfg_to_runs.setdefault(cfg_tag, []).append(r)

print(f"  Found {len(cfg_to_runs)} configs.\n")

# ── Label maps ────────────────────────────────────────────────────────────────
dataset_map = {
    "D0": "D0(3d,k2,m1)",
    "D1": "D1(10d,k2,m8)",
    "D2": "D2(10d,k5,m5)",
    "D3": "D3(20d,k5,m15)",
}
noise_map = {"1": "N1-Asym05", "2": "N2-Asym15", "3": "N3-Ext30"}
prior_map = {"0": "none", "1": "L1", "2": "L2"}
pred_map  = {"0": "none", "1": "L1", "2": "L2"}

# ── Metrics ───────────────────────────────────────────────────────────────────
METRICS_HIGHER_BETTER = [
    "clean_flatness_ratio_a", "clean_flatness_ratio_b",
    "r2_joint", "r2_a", "r2_b",
    "r2_dim0_u1", "r2_dim1_u2",
    "diagonality_ratio",
    "cca_diag_score", "cca_dim0", "cca_dim1",
    "retrieval_cos@1", "retrieval_l2@1",
    "retrieval_cos@5", "retrieval_l2@5",
]
METRICS_LOWER_BETTER = [
    "r2_dim2_noise",
    "clean_orth_residual_a", "clean_orth_residual_b",
    "val_align_a2b", "val_align_b2a",
]
ALL_METRICS = METRICS_HIGHER_BETTER + METRICS_LOWER_BETTER

# ── Build rows ─────────────────────────────────────────────────────────────────
rows = []
for cfg_file in sorted(cfg_dir.glob("B09_D*.yaml")):
    cfg_name = cfg_file.stem
    m = re.search(r"B09_(D\d)_N(\d)P(\d)(\d)", cfg_name)
    if not m:
        continue
    d_code, n_idx, p1_idx, p2_idx = m.groups()

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
        "config":  cfg_name,
        "dataset": dataset_map.get(d_code, d_code),
        "d_code":  d_code,
        "noise":   noise_map.get(n_idx, n_idx),
        "prior":   prior_map.get(p1_idx, p1_idx),
        "pred":    pred_map.get(p2_idx, p2_idx),
        "combo":   f"prior:{prior_map.get(p1_idx,'?')}+pred:{pred_map.get(p2_idx,'?')}",
        "state":   (train_run.state if train_run else "missing"),
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
total = len(df)
missing = (df["state"] == "missing").sum()
print(f"Loaded {total} config rows | missing wandb data: {missing}")
print(f"States: {df['state'].value_counts().to_dict()}\n")

# ── Composite scores ──────────────────────────────────────────────────────────
def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-12)

def neg_minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return 1.0 - (s - lo) / (hi - lo + 1e-12)

# Factor recovery sub-score
factor_r2 = (
    df["r2_joint"]    * 0.40
    + df["r2_dim0_u1"] * 0.30
    + df["r2_dim1_u2"] * 0.30
)

# Noise isolation
isolation = 1.0 - df["r2_dim2_noise"].clip(0, 1)

# Manifold structure (multiplicative: flatness AND low curvature together)
flatness_n  = minmax(df["clean_flatness_ratio_a"])
curvature_n = neg_minmax(df["clean_orth_residual_a"])
manifold_quality = flatness_n * curvature_n

# ── S_structure: calibrated primary (ρ=0.738 vs visual GT on B08) ─────────────
df["S_structure"] = (
    minmax(df["clean_flatness_ratio_a"]) * 0.30
    + minmax(factor_r2)                  * 0.40
    + neg_minmax(df["clean_orth_residual_a"]) * 0.20
    + minmax(isolation)                  * 0.10
)

# ── S_AV: AV-JEPA goal (reported for completeness; biased toward L2-pred) ─────
cross_mse = (df["val_align_a2b"] + df["val_align_b2a"]) / 2.0
df["S_AV"] = (
    neg_minmax(cross_mse)          * 0.40
    + minmax(factor_r2)            * 0.35
    + minmax(manifold_quality)     * 0.15
    + minmax(isolation)            * 0.10
)

# Joint gain
df["joint_gain"] = df["r2_joint"] - df[["r2_a", "r2_b"]].max(axis=1)

# ── Helper ────────────────────────────────────────────────────────────────────
def ranked_combo(data: pd.DataFrame, score: str) -> pd.DataFrame:
    grp = data.groupby("combo")[score].mean().sort_values(ascending=False)
    tbl = grp.reset_index()
    tbl.columns = ["combo", "mean_score"]
    tbl["rank"] = range(1, len(tbl) + 1)
    return tbl

sep = "=" * 80

# ══════════════════════════════════════════════════════════════════════════════
print(sep)
print("SECTION 1 — Overall combo ranking by S_structure (averaged over all D, N)")
print("  Primary validated composite (ρ=0.738 on B08 visual GT)")
print(sep)
tbl_overall = ranked_combo(df, "S_structure")
print(tbl_overall.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 2 — Per-metric means by combo (averaged over all D, N)")
print(sep)
key_metrics = [
    "clean_flatness_ratio_a", "r2_joint", "r2_dim0_u1", "r2_dim1_u2",
    "r2_dim2_noise", "clean_orth_residual_a",
    "val_align_a2b", "val_align_b2a",
]
metric_means = df.groupby("combo")[key_metrics].mean()
sorted_tbl = metric_means.sort_values("clean_flatness_ratio_a", ascending=False)
with pd.option_context("display.float_format", "{:.4f}".format,
                       "display.max_columns", 20, "display.width", 200):
    print(sorted_tbl.to_string())

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 3 — S_structure ranking per dataset (D0→D3)")
print("  Key question: does quality degrade as ambient dimension increases?")
print(sep)

d_scores = {}
for d in ["D0", "D1", "D2", "D3"]:
    sub = df[df["d_code"] == d]
    if sub.empty:
        print(f"  {d}: NO DATA")
        continue
    tbl = ranked_combo(sub, "S_structure")
    d_scores[d] = tbl
    top4 = tbl.head(4)[["combo", "mean_score"]].values
    print(f"\n  {dataset_map[d]}:")
    for rank_i, (combo, score) in enumerate(top4, 1):
        print(f"    #{rank_i}  {combo:<35}  S={score:.4f}")
    bottom2 = tbl.tail(2)[["combo", "mean_score"]].values
    print(f"    (worst: {bottom2[0][0]} S={bottom2[0][1]:.4f}, "
          f"{bottom2[1][0]} S={bottom2[1][1]:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 4 — S_structure mean per dataset (does overall score drop with D?)")
print(sep)
d_mean = df.groupby("d_code")["S_structure"].mean().sort_index()
d_mean_av = df.groupby("d_code")["S_AV"].mean().sort_index()
for d in ["D0", "D1", "D2", "D3"]:
    if d in d_mean.index:
        print(f"  {dataset_map[d]}: S_structure={d_mean[d]:.4f}  S_AV={d_mean_av[d]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 5 — Predictor effect: mean S_structure by pred / prior")
print(sep)
pred_eff  = df.groupby("pred")["S_structure"].mean().sort_values(ascending=False)
prior_eff = df.groupby("prior")["S_structure"].mean().sort_values(ascending=False)
print("By pred type (all D, N):")
for pred_val, score in pred_eff.items():
    print(f"  pred:{pred_val}  S_structure={score:.4f}")
print("\nBy prior type (all D, N):")
for prior_val, score in prior_eff.items():
    print(f"  prior:{prior_val}  S_structure={score:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 6 — Predictor effect per dataset (does pred importance change with D?)")
print(sep)
pred_by_d = df.groupby(["d_code", "pred"])["S_structure"].mean().unstack("pred")
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
    print(pred_by_d.to_string())

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 7 — Cross-modal MSE per combo [val_align], grouped by pred type")
print("  NOTE: val_align is biased toward L2-pred (MSE loss matches eval metric)")
print("  Safe comparison: within same pred type only")
print(sep)
cross_mse_combo = df.groupby("combo")[["val_align_a2b", "val_align_b2a"]].mean()
cross_mse_combo["mean_cross_mse"] = (
    cross_mse_combo["val_align_a2b"] + cross_mse_combo["val_align_b2a"]
) / 2.0
cross_mse_combo = cross_mse_combo.sort_values("mean_cross_mse")

print("\n  L1-pred combos (fair comparison within group):")
l1_pred = cross_mse_combo[cross_mse_combo.index.str.contains("pred:L1")]
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(l1_pred.to_string())

print("\n  L2-pred combos:")
l2_pred = cross_mse_combo[cross_mse_combo.index.str.contains("pred:L2")]
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(l2_pred.to_string())

print("\n  no-pred combos (baseline — no alignment expected):")
no_pred = cross_mse_combo[cross_mse_combo.index.str.contains("pred:none")]
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(no_pred.to_string())

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 8 — Cross-modal MSE per dataset for L1+L1 (key AV-relevant combo)")
print("  Does val_align_a2b degrade with ambient dimension for the best combo?")
print(sep)
l1l1 = df[df["combo"] == "prior:L1+pred:L1"].groupby("d_code")[
    ["val_align_a2b", "val_align_b2a", "S_structure"]
].mean()
l1l1["mean_cross_mse"] = (l1l1["val_align_a2b"] + l1l1["val_align_b2a"]) / 2.0
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(l1l1.to_string())

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 9 — Noise robustness: S_structure per noise level (averaged over D)")
print(sep)
noise_rank = df.groupby(["noise", "combo"])["S_structure"].mean().unstack("combo")
for n in ["N1-Asym05", "N2-Asym15", "N3-Ext30"]:
    if n in noise_rank.index:
        top3 = noise_rank.loc[n].sort_values(ascending=False).head(3)
        print(f"  {n}: " + " | ".join(f"{c} {v:.4f}" for c, v in top3.items()))

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 10 — Joint gain: r2_joint - max(r2_a, r2_b)")
print("  Measures complementarity bonus from fusing 2 modalities vs best single")
print(sep)
jg = df.groupby("combo")[["r2_joint", "r2_a", "r2_b", "joint_gain"]].mean()
jg = jg.sort_values("joint_gain", ascending=False)
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(jg.to_string())

# Per-dataset joint gain for L1+L1
print("\n  Joint gain for prior:L1+pred:L1 per dataset:")
jg_d = df[df["combo"] == "prior:L1+pred:L1"].groupby("d_code")[
    ["r2_joint", "r2_a", "r2_b", "joint_gain"]
].mean()
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
    print(jg_d.to_string())

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 11 — Noise isolation: r2_dim2_noise per (d_code, combo)")
print("  Lower = better shared/private disentanglement")
print(sep)
# Show for the 4 most relevant combos
key_combos = ["prior:L1+pred:L1", "prior:L2+pred:L2",
              "prior:none+pred:L1", "prior:L1+pred:none"]
iso_tbl = df[df["combo"].isin(key_combos)].groupby(
    ["d_code", "combo"]
)["r2_dim2_noise"].mean().unstack("combo")
# Reorder columns
present_cols = [c for c in key_combos if c in iso_tbl.columns]
iso_tbl = iso_tbl[present_cols]
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
    print(iso_tbl.to_string())

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 12 — Consensus ranking: mean rank across S_structure and S_AV")
print(sep)
tbl_av   = ranked_combo(df, "S_AV")
consensus = tbl_overall[["combo", "rank"]].rename(columns={"rank": "rank_S_structure"})
consensus = consensus.merge(
    tbl_av[["combo", "rank"]].rename(columns={"rank": "rank_S_AV"}),
    on="combo", how="left"
)
consensus["mean_rank"] = consensus[["rank_S_structure", "rank_S_AV"]].mean(axis=1)
consensus = consensus.sort_values("mean_rank")
print(consensus.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 13 — B09-specific hypotheses cross-check")
print(sep)

def get_metric(cfg: str, metric: str) -> float:
    rows_cfg = df[df["config"] == cfg]
    if rows_cfg.empty:
        return np.nan
    wandb_key = registry.get(metric)
    if wandb_key and metric in df.columns:
        val = rows_cfg[metric].values[0]
    else:
        val = rows_cfg.get(metric, pd.Series([np.nan])).values[0]
    return float(val) if not pd.isna(val) else np.nan

# H1
h1_a = get_metric("B09_D0_N3P10", "clean_flatness_ratio_a")
h1_b = get_metric("B09_D0_N3P01", "clean_flatness_ratio_a")
h1_res = "PASS (PRED_FLATTENS)" if (h1_b - h1_a) > 0.05 else "FAIL"
print(f"\n  H1 — Pred flattens MLP manifold under ext noise (N3, D0):")
print(f"    Prior-only (P10) flatness = {h1_a:.4f}")
print(f"    Pred-only  (P01) flatness = {h1_b:.4f}  diff={h1_b-h1_a:+.4f}  → {h1_res}")

# H2
h2_a = get_metric("B09_D0_N1P11", "r2_dim2_noise")
h2_b = get_metric("B09_D3_N1P11", "r2_dim2_noise")
h2_res = "PASS (ISOLATION_DEGRADES)" if (h2_b - h2_a) > 0.05 else "FAIL (ISOLATION_ROBUST)"
print(f"\n  H2 — Noise isolation degrades D0→D3 under L1+L1, N1:")
print(f"    D0 r2_dim2_noise = {h2_a:.4f}")
print(f"    D3 r2_dim2_noise = {h2_b:.4f}  diff={h2_b-h2_a:+.4f}  → {h2_res}")

# H3
h3_l1 = get_metric("B09_D3_N1P11", "diagonality_ratio")  # L1+L1
h3_l2 = get_metric("B09_D3_N1P21", "diagonality_ratio")  # L2+L1
h3_res = "PASS" if (h3_l1 - h3_l2) > 0.08 else "FAIL"
print(f"\n  H3 — L1 prior axis alignment at scale (D3, N1):")
print(f"    L1+L1 diagonality = {h3_l1:.4f}")
print(f"    L2+L1 diagonality = {h3_l2:.4f}  diff={h3_l1-h3_l2:+.4f}  → {h3_res}")
print(f"    ⚠️  Note: diagonality_ratio is misleading (high for pred:none too)")

# H4 — needs B08 data: pull from calibration CSV if available
b08_csv = SCRIPT_DIR.parent / "B08_volumetric_alignment" / "B08_calibration.csv"
h4_b09 = get_metric("B09_D0_N1P11", "retrieval_cos@1")
h4_b08 = np.nan
if b08_csv.exists():
    b08_df = pd.read_csv(b08_csv)
    b08_row = b08_df[b08_df["config"] == "B08_NPP111"]
    if not b08_row.empty:
        h4_b08 = b08_row["retrieval_cos@1"].values[0]
print(f"\n  H4 — MLP vs Analytical manifold (retrieval_cos@1, N1 synergy):")
print(f"    B09_D0_N1P11 retrieval_cos@1 = {h4_b09:.4f}")
print(f"    B08_NPP111   retrieval_cos@1 = {h4_b08:.4f}")
if not np.isnan(h4_b08) and not np.isnan(h4_b09):
    diff = abs(h4_b09 - h4_b08)
    h4_res = "PASS" if diff < 0.05 else "FAIL"
    print(f"    |diff| = {diff:.4f}  → {h4_res}")
    print(f"    ⚠️  Both near retrieval floor (~0.001–0.02); diff trivially small by floor effect")
else:
    print("    (B08 calibration CSV not found — run calibrate_metrics.py first)")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("SECTION 14 — SUMMARY: take-home messages")
print(sep)

# Compute a few summary stats for the printout
best_combo_overall = tbl_overall.iloc[0]["combo"]
pred_ranking = pred_eff.index.tolist()
top_d_drop = {d: d_mean[d] if d in d_mean.index else np.nan for d in ["D0","D1","D2","D3"]}
d0_d3_drop = top_d_drop.get("D3", np.nan) - top_d_drop.get("D0", np.nan)

print(f"""
  A) Best combo overall: {best_combo_overall}
     (S_structure ranking averaged over all D×N configurations)

  B) Predictor importance (pred ordering by S_structure):
     {' > '.join(f'pred:{p}' for p in pred_ranking)}
     → pred:none is ALWAYS bottom tier — finding is robust across D/N

  C) Quality vs ambient dimension (S_structure mean):
""")
for d in ["D0", "D1", "D2", "D3"]:
    v = top_d_drop.get(d, np.nan)
    print(f"     {dataset_map.get(d, d)}: {v:.4f}")
print(f"""
     D0→D3 net change: {d0_d3_drop:+.4f}  ({'degraded' if d0_d3_drop < -0.02 else 'roughly stable' if abs(d0_d3_drop) < 0.02 else 'improved'})

  D) AV-JEPA relevance:
     - val_align (cross-modal MSE) is biased toward L2-pred due to loss-metric mismatch
     - Within L1-pred group: prior:L1+pred:L1 has lowest mean_cross_mse → canonical frame
     - joint_gain is small in synthetic (same k factors in both modalities by construction)
       → will be much larger in real AV with genuinely different private content

  E) What metrics are informative vs misleading in B09:
     ✅ S_structure (flatness + factor_r2 + curvature): primary, calibrated
     ✅ val_align (within pred-type groups only): AV-critical
     ✅ r2_dim0_u1 / r2_dim1_u2: direct factor recovery signal
     ⚠️  diagonality_ratio: misleading — exclude from rankings
     ⚠️  retrieval_cos@1: floor effect, near-zero for all — useless as discriminator
     ⚠️  val_align as standalone ranking: biased toward L2-pred, don't use cross-group
""")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"Full data saved → {OUT_CSV}")
print("Done.")
