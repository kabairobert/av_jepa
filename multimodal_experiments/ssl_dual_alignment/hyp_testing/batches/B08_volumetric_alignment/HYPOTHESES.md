# B08 — Volumetric Alignment Hypotheses

This batch studies the geometric mechanics of Priors and Predictors on the volumetric `3d-3f-2c` shape manifold under varying asymmetric and external noise regimes.

---

## Hypotheses

### H1 — Predictor's flattening pressure under external noise
* **Question:** Does the Predictor's flattening pressure hold up under significant external noise?
* **Mechanism:** The cross-modal predictor encourages representation flattening to ease function approximation. Under 30% external noise (N5), where nearly a third of the points are uniform garbage, the predictor's geometry-shaping force remains the primary flattener.
* **Claim:** Even under 30% external noise, a Predictor-only run (`B08_NPP501`) produces higher clean manifold flatness (`geom/za/clean_flatness_ratio`) than a Prior-only run (`B08_NPP510`).

---

### H2 — Sparsity Prior + Predictor weight decay drives Subspace Isolation (Type 1)
* **Question:** How does the model isolate modality-unique dimensions?
* **Mechanism:** Independence of the modality-unique factors ($u_{3a} \perp u_{3b}$) results in zero cross-modal covariance: $\operatorname{Cov}(z_{A, 2}, z_{B, 2}) = 0$. This naturally drives the optimal predictor coefficient $w_2^* = 0$. 
  The L1 weight penalty $\lambda_{\text{sparse}} |w_2|$ acts as a strict thresholding force, overriding finite-sample sample covariance noise and locking $w_2$ exactly at 0. Combined with the volume-preserving Jacobian penalty, the flow networks must isolate the unique factor to coordinate 2 to avoid predictive error, while keeping shared factors in coordinates 0 and 1.
* **Claim:** Under low baseline noise (N1), the optimal synergy run (`B08_NPP111`) successfully separates the shared 2D manifold from the unique dimension. The leakage of the unique dimension with shared factors (`geom/za/r2_dim2_noise`) will be $< 0.15$, and diagonal predictor weights will successfully partition ($w_{\text{shared}} > 0.5$ and $w_{\text{unique}} \le 0.5$).

---

### H3 — L1 vs L2 Prior on Type 2 Axis Alignment
* **Question:** Does L1 Prior drive superior axis alignment within the shared subspace?
* **Mechanism:** L1 Prior (Laplace) acts as a coordinate-wise sparsity penalty, forcing active latents onto coordinate axes. L2 Prior (Gaussian) is rotationally invariant, leaving the 2D shared subspace arbitrarily rotated.
* **Claim:** Across all noise regimes, representations trained with L1 Prior (`B08_NPP*1*`) will achieve a significantly higher diagonality ratio (`geom/za/diagonality_ratio`) within the shared 2D subspace than those trained with L2 Prior (`B08_NPP*2*`).

---

### H4 — L1 vs L2 Predictor Robustness to Asymmetric Corruptions
* **Question:** Is L1 Predictor significantly more robust than L2 Predictor under severe asymmetric corruptions?
* **Mechanism:** L1 Predictor (MAE loss) minimizes absolute errors, behaving as a robust median estimator that ignores extreme asymmetric outliers. L2 Predictor (MSE loss) collapses because squared errors heavily penalize corruptions, biasing representation alignment.
* **Claim:** Under high asymmetric noise (N3: 25% Asym), L1 Predictor (`B08_NPP3*1`) achieves significantly higher cross-modal retrieval accuracy (`geom/retrieval_cos@1`) than L2 Predictor (`B08_NPP3*2`).
