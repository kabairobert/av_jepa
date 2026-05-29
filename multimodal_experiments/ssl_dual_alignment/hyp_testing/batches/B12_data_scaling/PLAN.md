# B12 Data Scaling Experiment Plan (Final)

## Goal
A sweep to investigate the hypothesis that increasing dataset size prevents the model from overfitting to finite dataset noise (coincidental alignment between random noise dimensions) and forces it to learn the true shared 2D manifold.

We keep the **number of epochs constant at 150** (matching B10 baseline runs) and scale up the data sizes up to **256x**.

## Scaling Schedule
All configurations use **Batch Size capped at 4,096** for scales 32x and higher, keeping the epochs constant at 150.

| Scale  | Num Samples | Batch Size | Updates / Epoch | Epochs | Total Updates | Est. Time (Upper Bound) |
|--------|-------------|------------|-----------------|--------|---------------|-------------------------|
| **1x** | 4,096       | 128        | 32              | 150    | 4,800         | ~25 mins                |
| **2x** | 8,192       | 256        | 32              | 150    | 4,800         | ~25 mins                |
| **4x** | 16,384      | 512        | 32              | 150    | 4,800         | ~25 mins                |
| **8x** | 32,768      | 1024       | 32              | 150    | 4,800         | ~25 mins                |
| **16x**| 65,536      | 2048       | 32              | 150    | 4,800         | ~25 mins                |
| **32x**| 131,072     | 4096       | 32              | 150    | 4,800         | ~30 mins                |
| **64x**| 262,144     | 4096       | 64              | 150    | 9,600         | ~1 - 1.5 hours          |
| **128x**| 524,288    | 4096       | 128             | 150    | 19,200        | ~2 - 3 hours            |
| **256x**| 1,048,576  | 4096       | 256             | 150    | 38,400        | ~4 - 5 hours            |

## Fixed Hyperparameters
- `lr`: 1.0e-3
- `u3a_scale`: 0.12, `u3b_scale`: 0.12, `turns`: 1.0, `wave_amplitude`: 1.0
- `external_noise_ratio`: 0.1 (N1)
- `prior_type`: l2, `lambda_prior`: 0.5, `lambda_sparse`: 0.1 (P2)
- `pred_loss`: l1, `lambda_pred`: 1.0 (P1)
- `predictor_type`: affine

### Grid Structure (1x2 Configuration per scale)
- **E - Embed Type:** `M` (Random MLP)
- **D - Embedding Dimension:** `20`

## Naming Convention
YAML configs follow the pattern `B12_[Scale]_M20_N1P21.yaml`.
Examples:
- `B12_1x_M20_N1P21.yaml`
- `B12_256x_M20_N1P21.yaml`
