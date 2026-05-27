# B11 Data Scaling Alignment Study

## Goal
A 24-config sweep to investigate the hypothesis that increasing dataset size prevents the model from overfitting to finite dataset noise (coincidental alignment between random noise dimensions) and forces it to learn the true shared 2D manifold. 

To isolate the effect of "data diversity" from "training compute", we strictly fix the total number of gradient updates (10,240) and batch size (512). As dataset size doubles, the number of epochs halves.

## Scaling Schedule
All configurations use **Batch Size = 512** and target **10,240 Total Updates**.

| Scale | Num Samples | Updates / Epoch | Epochs |
|-------|-------------|-----------------|--------|
| **1x**| 4,096       | 8               | 1,280  |
| **2x**| 8,192       | 16              | 640    |
| **4x**| 16,384      | 32              | 320    |
| **8x**| 32,768      | 64              | 160    |
| **16x**| 65,536      | 128             | 80     |
| **32x**| 131,072     | 256             | 40     |

## Fixed Hyperparameters
- `batch_size`: 512
- `lr`: 1.0e-3 (We keep LR constant since BS is constant)
- `u3a_scale`: 0.12
- `u3b_scale`: 0.12
- `turns`: 1.0
- `wave_amplitude`: 1.0
- `external_noise_ratio`: 0.1 (N1)
- `prior_type`: l2, `lambda_prior`: 0.5, `lambda_sparse`: 0.1 (P2)
- `pred_loss`: l1, `lambda_pred`: 1.0 (P1)
- `predictor_type`: affine

## Grid Structure (24 Configurations)

### S - Scale (6 levels)
- `1x` to `32x` as defined above.

### E - Embed Type (2 levels)
- `R`: **Orthogonal Rotation** (`data_type: 3d-3f-2c-rot`)
- `M`: **Random MLP** (`data_type: 3d-3f-2c-mlp`)

### D - Embedding Dimension (2 levels)
- `10`: **10 Dimensions** (`embed_dim: 10`)
- `20`: **20 Dimensions** (`embed_dim: 20`)

## Naming Convention
YAML configs follow the pattern `B11_[Scale]_[E][D]_N1P21.yaml`.

Examples:
- `B11_1x_R10_N1P21`: 1x scale (4k pts), 10D Rotation, 10% Ext Noise, L2 Prior, L1 Predictor
- `B11_32x_M20_N1P21`: 32x scale (131k pts), 20D MLP, 10% Ext Noise, L2 Prior, L1 Predictor
