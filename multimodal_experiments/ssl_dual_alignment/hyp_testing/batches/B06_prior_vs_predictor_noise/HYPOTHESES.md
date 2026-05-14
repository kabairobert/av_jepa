# B06 — Prior vs Predictor Ablation + Noise Level/Type Experiments

## Overview

Two interleaved questions tested in this batch:

### Question 1 — What drives manifold flattening?

The 3D visualisations show that training straightens the swiss roll and wave
sheet manifolds into flat planes. The question is: **which loss term causes
this** — the prior (L1/L2 shape on latent), the JEPA cross-modal predictor,
or both?

Tested via 4-way ablation on a fixed setup (affine predictor, L1 prior,
15%+15% asymmetric noise):

| Config | Prior | Pred loss | Role |
|--------|-------|-----------|------|
| C33 | ✗ | ✗ | jac-only floor |
| C31 | ✅ | ✗ | prior + jac (A) |
| C32 | ✗ | ✅ | pred + jac (B) |
| C25 | ✅ | ✅ | full baseline (C) |

The jac term is **always on** — it is the anti-collapse mechanism and
removing it makes any configuration ill-posed (trivial collapse to zero).

**Hypothesised mechanism:**
- Prior drives *dimensionality suppression* (kills off-signal dims via sparsity)
- Predictor drives *geometric linearisation* (forces manifold flat so affine
  cross-modal prediction is feasible)
- They are complementary — full outperforms either alone

### Question 2 — Noise level and type × prior type

Factorial: {L1, L2 prior} × {15%+15% asym, 10%+10%+10%ext, 25%+25% asym}

## New Metrics (added for this batch)

### `flatness_ratio` (per modality)
Fraction of latent variance explained by the top-2 PCA components.
→ 1.0 = all variance in a flat 2D plane (perfectly unrolled manifold)  
→ low = variance spread into 3rd dim (curved / noisy manifold)

### `orth_residual_mean` (per modality)
Mean distance of each point from its projection onto the top-2 PCA plane.  
→ 0.0 = all points lie in a flat plane  
→ high = manifold is curved / 3D

Both computed in `compute_geometry_metrics` (eval.py) via `manifold_flatness(z, n_signal_dims=2)`.
Registered in `metrics_registry.yaml` as `flatness_ratio_a/b` and `orth_residual_a/b`.

**Note:** Until `point_type_a/b` is implemented in `dataset.__getitem__`,
both metrics are computed over all points (manifold + noise). Noise points
artificially inflate off-plane variance → flatness is *underestimated*.
Priority: implement point_type first, then compute flatness on manifold-only subset.

## Hypotheses Summary

| ID | Claim | Primary metric | Configs |
|----|-------|----------------|---------|
| H1 | Pred drives flattening more than prior | `flatness_ratio_a` | C31 vs C32 |
| H2 | Prior drives dim suppression (dim2→0) | `r2_dim2_noise` | C32 vs C31 |
| H3 | Full outperforms both ablations (complementary) | `flatness_ratio_a`, `r2_joint` | C31/C32/C25 |
| H4 | Jac-only floor: near-zero retrieval, lowest flatness | `retrieval_cos@1` | C33 vs C25 |
| H5 | L1 prior cleaner noise dim than L2 at 15%+15% | `r2_dim2_noise` | C25 vs C28 |
| H6 | Mixed ext noise degrades more than pure asym; L1 more robust | `flatness_ratio_a` | C25/C26/C28/C29 |
| H7 | L1 more graceful than L2 at high noise (25%+25%) | `flatness_ratio_a`, `r2_joint` | C27 vs C30 |

## Configs Required

Existing (from session 2025-05-13): C25–C30 in `cfgs/`

New configs needed (add to cfgs/):
- **C31** — affine, L1 prior, **λ_pred=0**, 15%+15% asym, 0% ext
- **C32** — affine, **λ_prior=0, λ_sparse=0**, L1 pred, 15%+15% asym, 0% ext
- **C33** — affine, **λ_prior=0, λ_sparse=0, λ_pred=0**, 15%+15% asym, 0% ext (jac only)

## Implementation TODOs

1. Add `manifold_flatness()` to `eval.py` → `compute_geometry_metrics()`
2. Register `flatness_ratio_a/b`, `orth_residual_a/b` in `metrics_registry.yaml`
3. Add C31, C32, C33 YAML configs to `cfgs/`
4. Add `lambda_pred` override support to `EBMJEPALoss` (set to 0.0 to ablate)
5. Add `lambda_prior`/`lambda_sparse` = 0 path (already supported via config, verify)
6. Implement `point_type_a/b` in `dataset.py` + `eval.py` (prerequisite for per-type metrics)
