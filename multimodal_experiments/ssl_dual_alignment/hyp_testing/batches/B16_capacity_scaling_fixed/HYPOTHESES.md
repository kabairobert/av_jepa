# B16: Capacity Scaling with Fixed SNR for High-Dimensional Alignment

## Goal

Validate that the capacity scaling grid (dims 32 to 512, stages 8 to 16) successfully learns dual alignment and separates noise/signal dimensions once the Signal-to-Noise Ratio (SNR) collapse is resolved. The SNR is resolved by:
1. Normalizing the random MLP outputs to zero mean and unit variance per dimension in `dataset.py` before adding noise.
2. Fixing the bounding box estimation to use the post-normalization ranges.
3. Reducing the manifold noise std from 1.0 to 0.02 (`manifold_noise_a = 0.02`, `manifold_noise_b = 0.02`).

Fixed parameters:
- `dataset`: `nd-kf-mlp` (Frozen random MLPs projecting latents)
- `shared_factor_dist`: `normal` (Standard normal distribution for latent space)
- `k_shared`: `3` (3 common factors)
- `m_unique`: `0` (0 unique factors)
- `mlp_depth`: `2` (same as B15)
- `num_samples`: `1048576` (1.05M dense dataset)
- `batch_size`: `4096`
- `epochs`: `150` (extended from 100 to allow full convergence at 512D)

## Sweep Grid

Logarithmic dimension-to-stage scaling with custom width:

| Dimension ($D$) | Flow Stages ($S$) | Hidden Units |
|:---:|:---:|:---:|
| 16 | 6 | 128 |
| 32 | 8 | 128 |
| 64 | 10 | 128 |
| 128 | 12 | 128 |
| 256 | 14 | 128 |
| 512 | 16 | 256 |

**6 configurations total.**

## Hypotheses

### H16.1: Alignment and Dimension Separation under Correct SNR
With the SNR fixed (normalization + noise=0.02), all configurations from 16D to 512D will successfully align the 3 shared latent coordinates and achieve a clean separation between signal and noise dimensions (manifested by CCA Rank of exactly 3, high R2 joint, and high diagonality score).

### H16.2: Scalability of Flow Adapters to High Dims
The logarithmic stage scaling ($S = 2 \log_2(D) - 2$) remains sufficient to untangle the normalized MLP projection across all dimensions up to 512D.
