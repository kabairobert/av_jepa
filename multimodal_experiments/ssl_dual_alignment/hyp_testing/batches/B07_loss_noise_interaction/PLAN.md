# B07 Loss-Noise Interaction Study Plan

## Goal
A comprehensive $3 \times 3 \times 5$ factorial sweep to investigate how Prior and Predictor choices interact with various noise regimes to drive manifold flattening and dimensionality suppression.

## Base Hyperparameters
- `num_samples`: 4,096 (standard manifold density)
- `batch_size`: 128 (matches production V-JEPA limitations)
- `epochs`: 200 (reduced from 300 for faster iteration)
- `lr`: 1.0e-3 (standard baseline)

## Grid Structure

### N - Noise Levels (5 levels)
- `1`: **Asymmetric 15%** (`asym15_ext0`, magnitude 0.1, bbox+25%)
- `2`: **Asymmetric 25%** (`asym25_ext0`, magnitude 0.1, bbox+25%)
- `3`: **Asymmetric 37.5%** (`asym37.5_ext0`, magnitude 0.1, bbox+25%)
- `4`: **External 50%** (`asym0_ext50`, magnitude 0.1, bbox+25%)
- `5`: **High Noise Mixture** (`asym9_ext75`, magnitude 0.1, bbox+25%)

### P1 - Prior Types (3 levels)
- `0`: **None** (`lambda_prior: 0.0`, `lambda_sparse: 0.0`)
- `1`: **L1** (`prior_type: l1`, `lambda_prior: 0.5`, `lambda_sparse: 0.1`)
- `2`: **L2** (`prior_type: l2`, `lambda_prior: 0.5`, `lambda_sparse: 0.1`)

### P2 - Predictor Types (3 levels)
- `0`: **None** (`lambda_pred: 0.0`, `predictor_type: affine`)
- `1`: **L1** (`pred_loss: l1`, `lambda_pred: 1.0`, `predictor_type: affine`)
- `2`: **L2** (`pred_loss: l2`, `lambda_pred: 1.0`, `predictor_type: affine`)

## Naming Convention
Files follow the pattern `NPP[N][P1][P2].yaml`.

Examples:
- `NPP111`: 15% Asym, L1 Prior, L1 Predictor (Baseline)
- `NPP300`: 37.5% Asym, No Prior, No Predictor (Jacobian Floor)
- `NPP521`: 75% Ext, L2 Prior, L1 Predictor

## Execution Workflow
1. **Research & Bug Fixes**: Address reported bugs in `main.py` and `losses.py`.
2. **Config Generation**: Scripted generation of all 45 `.yaml` files.
3. **Hypotheses Definition**: Define `hypotheses.yaml` focusing on interaction effects.
4. **Execution**: Launch sweep using `sweep.py`.
5. **WandB Migration**: (Optional) Retrospectively tag existing `B06` runs if they remain valid after bug fixes.

## Key Questions
- Does the Predictor's flattening pressure (H1) hold up under extreme external noise?
- Is L1 Prior significantly more robust than L2 as the noise-to-signal ratio increases?
- Is there a "regime change" where L2 Predictor outperforms L1?
