# B15 Failure Analysis: SNR Collapse

## Overview
Batch B15 (`B15_capacity_scaling`) failed to learn dual alignment and separate noise/signal dimensions. The primary cause of failure was a catastrophic Signal-to-Noise Ratio (SNR) collapse in the dataset generation for the `nd-kf-mlp` mode.

## Core Issues

1. **Variance Scaling Problem in `dataset.py`**
   - The dataset uses frozen random MLPs to project latent variables into high-dimensional space.
   - However, the outputs of these MLPs were not normalized. As a result, the variance of the signal varied wildly across dimensions and runs.

2. **Manifold Noise Dominance**
   - In B15, `manifold_noise` was set to `1.0`.
   - Because the underlying MLP signal was not normalized to unit variance, injecting Gaussian noise with a standard deviation of 1.0 completely overwhelmed the structural signal in dimensions where the MLP output naturally had a small variance.
   - The model was effectively tasked with aligning pure noise.

3. **Bounding Box Estimation Mismatch**
   - The bounding boxes (`ma, Ma, mb, Mb`) used for uniform corruption were estimated on the raw, unnormalized signal. When noise was added, these bounds became mathematically invalid, breaking the external corrupt noise distribution.

## Resolution (Implemented in B16)

To resolve these issues for Batch B16 (`B16_capacity_scaling_fixed`), the following fixes were implemented:
1. **Signal Normalization:** The random MLP outputs (`xa, xb`) are now explicitly normalized to zero mean and unit variance per dimension *before* noise is added.
2. **Correct Bounding Boxes:** The bounding boxes are now calculated on the post-normalization ranges.
3. **SNR Calibration:** The `manifold_noise` standard deviation was reduced from `1.0` to `0.02` to ensure the structural signal remains dominant.
4. **Final Normalization Check:** A final normalization step guarantees the complete `data_a, data_b` matrices have a standard deviation of exactly 1.0 before being fed to the model.
