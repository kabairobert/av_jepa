# B16b Analysis: Predictor Collapse & Gaussian Unidentifiability

## Observation
Output point cloud fully separates valid pairs and external noise, but $u_1, u_2, u_3$ axes appear visually jumbled. Initially assumed to be pure Gaussian rotational symmetry artifact.

## 📊 Empirical Reality: Predictor Collapse
W&B metrics reveal a true failure, not just a visual artifact. Comparing B16b (failed) vs B13/B16c (working):
- **Working (B13 256D, B16c 32D):** Predictor successfully isolates 2 aligned dimensions (`found_rank_pred_r2_1 = 2`). Predictor $R^2$ reaches ~0.5-0.6 on specific dimensions.
- **Failing (B16b 16D, 32D):** Predictor $R^2$ is exactly 0.0 across all dimensions (`found_rank_pred_r2 = 0`). Predictor weights collapsed to zero.

## 🔩Mech: Sparsity Penalty vs Entangled Gaussian Signal
1. **Gaussian Data (`nd-kf-mlp`)**: Uses $u_1, u_2, u_3 \sim \mathcal{N}(0,1)$. Unlike the deterministic 3D spiral in B13/B16c, the Gaussian signal has no distinct topological boundaries or non-Gaussian marginals.
2. **Signal Dilution**: Passed through the MLP, the $3/D$ variance is uniformly spread across all $D$ dimensions.
3. **Predictor Collapse**: The EBM objective (`lambda_sparse=0.1`) penalizes predictor weights. Because the Gaussian signal is heavily entangled and lacks distinct non-Gaussian features to anchor alignment, the network cannot find a cheap mapping to align the views. The sparsity penalty overpowers the weak alignment signal → predictor weights drop to 0.
4. **Result**: The EBM loss (`lambda_prior=0.5`) forces both views to standard normal $\mathcal{N}(0, I_D)$ independently. No cross-view alignment occurs.

## ✔️Verify
Check `geom/pred_r2_dimX` and `found_rank_pred_r2_1` in W&B. Value of 0 means the affine predictor failed. Working models (even up to 256D) must show `found_rank_pred_r2_1 > 0`.

## 🔧Fix
To fix `nd-kf-mlp` capacity scaling:
1. Use `shared_factor_dist='uniform'`. Uniform distribution breaks rotational symmetry, providing non-Gaussian marginals that allow the affine predictor to identify and align the true independent components (similar to ICA).
2. If predictor still collapses, reduce `lambda_sparse` to prevent early weight death before alignment is found.
