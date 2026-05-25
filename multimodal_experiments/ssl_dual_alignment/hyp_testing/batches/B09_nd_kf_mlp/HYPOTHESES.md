# B09 — nd-kf-mlp Generalization Hypotheses

This batch tests whether the dual-alignment methodology generalizes from fixed analytical manifolds (spiral/wave in B07/B08) to high-dimensional random MLP embeddings that better approximate real AV complexity.

---

## Hypotheses

### H1 — Predictor Flattening Robustness on MLP Manifold

* **Question:** Does the predictor's flattening pressure survive on random MLP manifolds under significant external noise?
* **Mechanism:** In B08 (analytical manifold), the predictor remained the dominant flattening force under 30% external noise. The MLP baseline (`D0`) uses the same factor structure (k=2, m=1, d_out=3) but applies a random frozen MLP projection. If the predictor's geometry-shaping effect is genuine (not artefactual to spiral geometry), it should transfer.
* **Claim:** Under 30% external noise (N3), Predictor-only run (`B09_D0_N3P01`) achieves higher `clean_flatness_ratio` than Prior-only run (`B09_D0_N3P10`). The margin should exceed 0.05 (matching B08's H1 threshold).

---

### H2 — Subspace Isolation Degrades with Ambient Dimension

* **Question:** Does unique-factor isolation (Type 1 separation) break as ambient dimension grows from 3D → 10D → 20D?
* **Mechanism:** The shared-unique decomposition relies on $\operatorname{Cov}(z_{A, \text{unique}}, z_{B, \text{unique}}) \approx 0$. As ambient dimension grows, the number of spurious covariance terms grows quadratically ($O(d^2)$), while signal-to-noise for individual factor estimates degrades. The L1 sparsity prior thresholds are calibrated for low-D and may saturate in high-D.
* **Claim:** Under L1+L1 synergy (P11) and Asym 5% noise (N1), the unique-factor leakage metric `r2_dim_noise` increases significantly across dataset configs: `B09_D3_N1P11` > `B09_D1_N1P11` > `B09_D0_N1P11`. Specifically, D3 shows leakage ≥ 0.05 above D0.

---

### H3 — L1 vs L2 Prior Axis Alignment Persists at Scale

* **Question:** Does L1 Prior's coordinate-alignment advantage over L2 Prior hold regardless of ambient dimension?
* **Mechanism:** L1 Prior is a coordinate-wise sparsity penalty, forcing active latents onto axes. L2 Prior is rotationally invariant, leaving the shared subspace arbitrarily rotated. This rotational-invariance property is geometric and should be dimension-agnostic.
* **Claim:** Across all dataset configs (D0–D3) and noise levels, runs with L1 Prior (`P1=1`) achieve higher `diagonality_ratio` within the shared subspace than matched runs with L2 Prior (`P1=2`). The advantage is preserved even at D3 (20D ambient), confirming the geometric mechanism is not scale-dependent.

---

### H4 — MLP Manifold vs Structured Manifold Generalization

* **Question:** Does random MLP projection break cross-modal alignment relative to the structured analytical manifold?
* **Mechanism:** The analytical manifold (B08's `3d-3f-2c`) provides a smooth, well-conditioned latent space. The MLP baseline (`D0`) applies a frozen random MLP which may introduce local non-linearity and conditioning issues. If the methodology is robust, retrieval accuracy should be close across the two.
* **Claim:** The L1+L1 synergy run on the MLP baseline (`B09_D0_N1P11`) achieves `retrieval_cos@1` within **5% absolute** of the best matched B08 run (`B08_NPP111`). The methodology is not brittle to manifold shape.
