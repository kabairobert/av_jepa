# B10 3D to High-Dimensional (HD) Alignment Study Plan

## Goal
A comprehensive 72-config sweep using the volumetric `3d-3f-2c` shape engine embedded into high-dimensional space (10D and 20D) to investigate whether the Dual-JEPA model can isolate the shared 2D manifold from high-dimensional distractors. This batch tests if the flow model can successfully unroll the representation when the dimensionality increases, and compares linear embedding (orthogonal rotation) versus non-linear embedding (random frozen MLP).

## Base Hyperparameters
- `num_samples`: 4,096
- `batch_size`: 128
- `epochs`: 150
- `lr`: 1.0e-3
- `type`: `3d-3f-2c-rot` or `3d-3f-2c-mlp`
- `u3a_scale`: 0.12 (thickness of Modality A spiral)
- `u3b_scale`: 0.12 (thickness of Modality B wave)
- `turns`: 1.0
- `wave_amplitude`: 1.0
- `predictor_type`: `affine` (Weight acts as per-dimension scale to select active dimensions)

## Grid Structure (72 Configurations)

### E - Embed Type (2 levels)
- `R`: **Orthogonal Rotation** (`data_type: 3d-3f-2c-rot`)
- `M`: **Random MLP** (`data_type: 3d-3f-2c-mlp`)

### D - Embedding Dimension (2 levels)
- `10`: **10 Dimensions** (`embed_dim: 10`)
- `20`: **20 Dimensions** (`embed_dim: 20`)

### N - Noise Levels (2 levels)
- `1`: **External 10%** (`external_noise_ratio: 0.1`)
- `2`: **External 30%** (`external_noise_ratio: 0.3`)

### P1 - Prior Types (3 levels)
- `0`: **None** (`lambda_prior: 0.0`, `lambda_sparse: 0.0`)
- `1`: **L1** (`prior_type: l1`, `lambda_prior: 0.5`, `lambda_sparse: 0.1`)
- `2`: **L2** (`prior_type: l2`, `lambda_prior: 0.5`, `lambda_sparse: 0.1`)

### P2 - Predictor Types (3 levels)
- `0`: **None** (`lambda_pred: 0.0`)
- `1`: **L1** (`pred_loss: l1`, `lambda_pred: 1.0`)
- `2`: **L2** (`pred_loss: l2`, `lambda_pred: 1.0`)

#### Total Combinations:
2 (Embed Type) × 2 (Embed Dim) × 2 (Noise) × 3 (Prior) × 3 (Predictor) = **72 configurations**.

## Naming Convention
YAML configs follow the pattern `B10_[E][D]_N[N]P[P1][P2].yaml`.

Examples:
- `B10_R10_N1P11`: 10D Rotation, 10% Ext Noise, L1 Prior, L1 Predictor
- `B10_M20_N2P02`: 20D MLP, 30% Ext Noise, No Prior, L2 Predictor
- `B10_R20_N1P00`: 20D Rotation, 10% Ext Noise, No Prior, No Predictor (Baseline)
