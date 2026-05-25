# B09 nd-kf-mlp Generalization Sweep Plan

## Goal
A **4 × 3 × 8** sweep using the `nd-kf-mlp` dataset mode to test whether the dual-alignment methodology generalizes beyond fixed analytical manifolds (spiral/wave) into high-dimensional random MLP embeddings that better approximate real AV complexity.

## Dataset Configurations (axis D, 4 levels)

Naming: **Nd-Kf-Mc** → `d_out=N`, `k_shared=C`, `m_unique=K-C`

| Code | d_out | k_shared | m_unique | Model num_dims | hidden_units | stage_count | Rationale |
|------|-------|----------|----------|----------------|--------------|-------------|-----------|
| `D0` | 3     | 2        | 1        | 3              | 128          | 6           | **Baseline**: MLP equivalent of B08 (apples-to-apples) |
| `D1` | 10    | 2        | 8        | 10             | 256          | 6           | Same 2D shared hidden in 10D ambient → ambient curse test |
| `D2` | 10    | 5        | 5        | 10             | 256          | 6           | Larger 5D shared subspace in 10D → isolation harder |
| `D3` | 20    | 5        | 15       | 20             | 512          | 6           | 5D shared in 20D → toward realistic AV scale |

All use `data_type: nd-kf-mlp`, `num_samples=4096`, `batch_size=128`, `epochs=150`, `lr=1e-3`, `seed=12345`.

## Noise Levels (axis N, 3 levels)

| N | Label         | asym_rate_a/b | external_noise_ratio |
|---|---------------|---------------|----------------------|
| 1 | Asym 5%       | 0.05 / 0.05   | 0.00                 |
| 2 | Asym 15%      | 0.15 / 0.15   | 0.00                 |
| 3 | External 30%  | 0.00 / 0.00   | 0.30                 |

`asymmetric_noise_magnitude=0.1`, `noise_bbox_expansion=0.25` throughout.

## (Prior, Predictor) Combinations (axis PP, 8 combos)

Skip P1=0, P2=0 (no prior, no predictor). Same 8 combos as B08:

| P1 | Prior | P2 | Predictor    |
|----|-------|----|--------------|
| 0  | None  | 1  | L1 pred      |
| 0  | None  | 2  | L2 pred      |
| 1  | L1    | 0  | No pred      |
| 1  | L1    | 1  | L1 pred (synergy) |
| 1  | L1    | 2  | L2 pred      |
| 2  | L2    | 0  | No pred      |
| 2  | L2    | 1  | L1 pred      |
| 2  | L2    | 2  | L2 pred      |

`lambda_prior=0.5`, `lambda_sparse=0.1` when prior active. `lambda_pred=1.0` when pred active.

## Total Run Count

4 datasets × 3 noise × 8 combos = **96 runs**

## Naming Convention

```
B09_[DATASET_CODE]_N[n]P[p1][p2].yaml
```

Examples:
- `B09_D0_N1P11.yaml` = 3d-3f-2c baseline, Asym 5%, L1 Prior + L1 Pred (synergy baseline)
- `B09_D3_N3P01.yaml` = 20d-20f-5c, External 30%, No Prior + L1 Pred
- `B09_D2_N2P20.yaml` = 10d-10f-5c, Asym 15%, L2 Prior + No Pred

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Baseline data config | `D0` (d_out=3, k=2, m=1) | Apples-to-apples vs B08 `3d-3f-2c` |
| Ambient dim scaling | 3 → 10 → 20 | Gradual, representative of AV scale |
| Noise axis | 3 levels (asym 5%, 15%, ext 30%) | Reduced from B08's 6 for run count control |
| (Prior, Pred) axis | Same 8 combos as B08 | Full cross-batch comparability |
| Model latent dim | `num_dims = d_out` (full-rank) | Clean setup; model matches ambient dim |
| Model capacity | `hidden_units` scales with `d_out` | Prevent capacity bottleneck in high-D |
| Seeds | Single seed (12345) | Keeps total at 96 runs |
| Eval metrics | Same 3 as B08 | Cross-batch comparability |

## Bug Fix: main.py Forward Pass

`main.py` was not forwarding `k_shared`, `m_unique`, `d_out` from cfg to `DualDisentangleDataset`,
causing all nd-kf-mlp runs to silently use defaults (k=2, m=2, d_out=16). Fixed by adding:

```python
k_shared=cfg.data.get('k_shared', 2),
m_unique=cfg.data.get('m_unique', 2),
d_out=cfg.data.get('d_out', 16),
```
