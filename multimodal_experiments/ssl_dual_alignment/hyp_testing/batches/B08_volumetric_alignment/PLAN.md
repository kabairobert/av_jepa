# B08 Volumetric Alignment Study Plan

## Goal
A comprehensive $6 \times 8$ sweep using the volumetric `3d-3f-2c` shape engine to investigate how Prior and Predictor choices interact with various noise regimes to drive representation flattening, subspace isolation, and strict axis alignment.

## Base Hyperparameters
- `num_samples`: 4,096
- `batch_size`: 128
- `epochs`: 150
- `lr`: 1.0e-3
- `type`: `3d-3f-2c` (Volumetric Spiral and Volumetric Wave)
- `u3a_scale`: 0.12 (thickness of Modality A spiral)
- `u3b_scale`: 0.12 (thickness of Modality B wave)
- `turns`: 1.0
- `wave_amplitude`: 1.0

## Grid Structure

### N - Noise Levels (6 levels)
- `1`: **Asymmetric 5%** (`asym05_ext0`, magnitude 0.1, bbox+25%)
- `2`: **Asymmetric 15%** (`asym15_ext0`, magnitude 0.1, bbox+25%)
- `3`: **Asymmetric 25%** (`asym25_ext0`, magnitude 0.1, bbox+25%)
- `4`: **External 10%** (`asym0_ext10`, magnitude 0.1, bbox+25%)
- `5`: **External 30%** (`asym0_ext30`, magnitude 0.1, bbox+25%)
- `6`: **External 50%** (`asym0_ext50`, magnitude 0.1, bbox+25%)

### P1 - Prior Types (3 levels)
- `0`: **None** (`lambda_prior: 0.0`, `lambda_sparse: 0.0`)
- `1`: **L1** (`prior_type: l1`, `lambda_prior: 0.5`, `lambda_sparse: 0.1`)
- `2`: **L2** (`prior_type: l2`, `lambda_prior: 0.5`, `lambda_sparse: 0.1`)

### P2 - Predictor Types (3 levels)
- `0`: **None** (`lambda_pred: 0.0`, `predictor_type: affine`)
- `1`: **L1** (`pred_loss: l1`, `lambda_pred: 1.0`, `predictor_type: affine`)
- `2`: **L2** (`pred_loss: l2`, `lambda_pred: 1.0`, `predictor_type: affine`)

#### (Prior, Predictor) combinations:
- All combinations except Prior: **None**, Predictor: **None**. Total 9-1 = 8 combinations.

## Naming Convention
YAML configs follow the pattern `B08_NPP[N][P1][P2].yaml`.

Examples:
- `B08_NPP111`: 5% Asym, L1 Prior, L1 Predictor (Synergy baseline)
- `B08_NPP302`: 25% Asym, No Prior, L2 Predictor
- `B08_NPP620`: 50% Ext, L2 Prior, No Predictor
