# B13: Affine Disentanglement Sweep

## Goal

Test whether `ClampedAffineCoupling` ($y = x \cdot e^s + t$) can untangle a 20D
Random MLP embedding, where B12's Additive coupling failed.

Fixed parameters: `coupling_type=affine`, `hidden_units=128`, `lambda_jac=1.0`,
`lambda_prior=0.5`, `prior_type=l2`, `mlp_depth=2`.

## Sweep Grid

| Axis | Values |
|------|--------|
| `dataset_multiplier` | 1x (4k), 16x (64k), 256x (1M) |
| `stage_count` | 6, 12 |

**6 configs total.**

## Hypotheses

### H13.1: Affine Untangling
Affine coupling (scale + shear) provides sufficient geometric flexibility to
axis-align the 2 shared dims from a 20D Random MLP embedding, whereas Additive
coupling failed.

### H13.2: Depth Dependency
Deeper flows (`stage_count=12`) achieve significantly higher disentanglement
scores than shallow (`stage_count=6`) when inverting a 2-layer neural network.

### H13.3: Data Efficiency
Untangling a non-linear MLP manifold with Affine coupling requires dense
manifold sampling, peaking at `256x`.

## Expected Variance Targets (theory)

| Dim type | Variance target |
|----------|----------------|
| Shared (2 dims) | σ² = λ_jac / (2·λ_prior) = **1.0** |
| Unique (18 dims) | σ² = λ_jac / (2·(λ_prior + λ_pred)) = **0.33** |
