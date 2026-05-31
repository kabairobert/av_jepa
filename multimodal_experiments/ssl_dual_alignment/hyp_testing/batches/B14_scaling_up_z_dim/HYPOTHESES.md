# B14: High-Dimensional Z Scaling Sweep

## Goal

Evaluate the scalability of non-volume-preserving `ClampedAffineCoupling` flows when the ambient observation dimension scales up drastically up to 512D. Specifically, verify if a logarithmic scaling of the flow depth (number of coupling stages) is sufficient to isolate and align low-dimensional intrinsic manifolds embedded in high-dimensional noise.

Fixed parameters:
- `dataset`: `nd-kf-mlp` (Frozen random MLPs projecting latents)
- `shared_factor_dist`: `normal` (Standard normal distribution for latent space)
- `k_shared`: `3` (3 common factors)
- `m_unique`: `0` (0 unique factors)
- `mlp_depth`: `3`
- `num_samples`: `1048576` (1.05M dense dataset)
- `batch_size`: `4096`
- `epochs`: `100`

## Sweep Grid

Logarithmic dimension-to-stage scaling:

| Dimension ($D$) | Flow Stages ($S$) |
|:---:|:---:|
| 32 | 4 |
| 64 | 6 |
| 128 | 8 |
| 256 | 10 |
| 512 | 12 |

**5 configurations total.**

## Hypotheses

### H14.1: Logarithmic Depth Scaling Sufficiency
As the ambient observation dimension $D$ scales up exponentially to 512D, a logarithmic scaling of the flow stages ($S \approx 2 \cdot \log_2(D)$) is sufficient to untangle and align the low-dimensional intrinsic manifold without suffering from representational or optimization collapse.

### H14.2: High-Dimensional Noise Suppression
The non-volume-preserving scale parameter in `ClampedAffineCoupling` can successfully suppress the extra uninformative dimensions (e.g. shrinking 509 non-signal dimensions in the 512D run) even under complex nonlinear random MLP projections.

### H14.3: 3D Common Factor Retrieval & Visualization
The Found Common Rank Suite will successfully capture exactly 3 active dimensions mapping cleanly to the 3 standard normal shared factors, and the Plotly/Matplotlib visualizers will successfully render the intrinsic 3D geometry using Rainbow Hue ($u_1$), Saturation/Lightness ($u_2$), and Size ($u_3$) concurrently.
