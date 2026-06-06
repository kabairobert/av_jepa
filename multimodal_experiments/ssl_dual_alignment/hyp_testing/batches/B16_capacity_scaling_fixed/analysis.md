# B16 Batch Analysis: Why `nd-kf-mlp` Failed and How to Fix It

## Observation
The B16 capacity scaling experiment (using `nd-kf-mlp`) failed to align properly, with output plots looking worse than the input plots. This was true even for 16D models, whereas similar dimensionality (20D) in the B13 experiment (using `3d-3f-2c-mlp`) aligned successfully.

## The Mathematical Cause: Manifold Collapse in Ambient Space
The core issue is a structural mismatch between the intrinsic dimensionality of the generated data and the volume-preserving requirements of the Normalizing Flow model trained with EBM loss.

In B16, the data configuration uses `k_shared=3` and `m_unique=0`. 
- **Dimensionality:** Because the input to the MLP is exactly 3-dimensional, the output is a **degenerate 3D manifold** embedded in a 16D ambient space. 
- **Noise:** At the very end of the pipeline, a tiny isotropic noise (`manifold_noise = 0.02`) is added. Thus, 3 dimensions have an intrinsic variance of `~1.0`, while the remaining 13 orthogonal dimensions have a variance of exactly `0.02^2 = 0.0004`. The data possesses near-zero volume in 16D space.
- **The Flow Model Objective:** The EBM loss (`lambda_jac=1.0`, `lambda_prior=0.5`) mathematically enforces Maximum Likelihood matching to an Isotropic Gaussian `N(0, I)` with variance `1.0`. To satisfy this objective, the Normalizing Flow is forced to stretch the 13 degenerate dimensions from a variance of `0.0004` to `1.0` (a massive `50x` scale expansion). This tears the manifold apart.

### Why B13 (`3d-3f-2c-mlp`) Worked
In B13, the base 3D signal is padded with 17 dimensions of pure `N(0,1)` noise **before** passing through the MLP. The input to the B13 MLP is a full-rank 20D distribution (variance `~1.0` everywhere). The resulting output is fully volume-filling in the ambient space. The Flow model can seamlessly map this to a 20D Gaussian prior.

## Step-by-Step Exact Code Execution Order

You correctly identified that the steps are **not** executed in the same order between the two methods. Here is the rigorous execution trace derived directly from `dataset.py`:

| Execution Order | `3d-3f-2c-mlp` (B13) | `nd-kf-mlp` (with `m_unique>0`) |
| :--- | :--- | :--- |
| **1. Shared Latents** | `u1, u2` (Uniform) mapped to 3D shape | `u_s` (Gaussian) |
| **2. Small Perturbation** | Add `manifold_noise=0.02` to the 3D shape | *(Skipped)* |
| **3. Non-Common Dims** | Pad 3D shape with `17` dims of `N(0,1)` | Sample `u_ua, u_ub` (Gaussian), concat with `u_s` |
| **4. Mixing (MLP)** | MLP (`20 -> 64 -> 20`) tangles signal+noise | MLP (`16 -> 64 -> 16`) tangles shared+unique |
| **5. Normalization** | Output normalized (Mean 0, Std 1) | Output normalized (Mean 0, Std 1) |
| **6. Post-Jitter** | *(None)* | Add `manifold_noise=0.02` |

### Is Everything The Same If We Add `m_unique`?
Almost everything, and critically, the **ambient volume problem is perfectly fixed**. 

If you use `nd-kf-mlp` with `m_unique: 13`, the MLP receives a full-rank 16D input (3 shared + 13 unique, all variance 1.0). The MLP outputs a full-rank 16D continuous embedding space. The Flow model will easily align this without blowing up.

The **only remaining difference** (besides the base shape) is the placement of the 0.02 `manifold_noise`:
- In `3d-3f-2c-mlp`, the 0.02 noise is a physical perturbation applied *before* the MLP, so it gets non-linearly warped by the network.
- In `nd-kf-mlp`, the 0.02 noise is a post-embedding jitter applied *after* the MLP.

This difference is mathematically minor for alignment difficulty, because the 13 dimensions of `m_unique` provide the massive ambient volume needed by the Flow model.

## Conclusion and Realistic Modeling
Setting `m_unique: 13` makes `nd-kf-mlp` the **most realistic** synthetic proxy for Audio-Video SSL embeddings. Real SSL models (Wav2Vec, DINO) don't output degenerate empty space. They mix shared physical events (speech) and modality-unique events (background traffic, lighting) *inside the network* to produce a dense, full-rank embedding space. The post-embedding 0.02 jitter in `nd-kf-mlp` simply acts as tiny numerical precision noise, which is completely fine once the manifold is naturally full-rank.
