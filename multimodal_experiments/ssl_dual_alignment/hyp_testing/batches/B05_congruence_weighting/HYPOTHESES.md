# B05: Sigmoid Congruence Gate on Pred/Sparse Losses

**Focus question:** Does applying a per-sample sigmoid congruence gate to `pred_loss` (and optionally `sparse_loss`) improve geometry and retrieval under high noise, without gating `jac_loss` or `prior_loss`?

**Configs used:** C19–C24 (all 3D-2f, high noise: ext_noise=0.15, asym_rate=0.10)  
**Implementation:** `congruence_mode` ∈ {`none`, `pred_only`, `pred_and_sparse`}; gate weight `w_i = σ(−L_pred_i / τ)`, normalised to unit-mean per batch.

**Background:** The old `noise_reweighting='full'` API applied the sigmoid gate to all four losses including `jac_loss` and `prior_loss`. These two are geometry regularisers: suppressing them on samples where the model is uncertain corrupts the latent manifold exactly where regularisation is most needed. The correct decomposition gates only the prediction-quality proxies (`pred_loss`, optionally `sparse_loss`) and keeps `jac_loss` and `prior_loss` uniformly weighted. B05 tests this design principle.

---

## H1: Congruence gate (pred_only, τ=0.5) improves geometry vs no gate

**Claim:** `C20` (gate on pred only) outperforms `C19` (no gate) on `r2_joint`, `retrieval_cos@1`, and `pca_axis_align_a/b` under high noise.

**Mechanism:** Noisy/incongruent samples produce high per-sample pred loss → low gate weight → their cross-modal gradients are down-weighted → the geometric structure emerges from cleaner, congruent samples only.

**Assumption:** `pred_loss_per_sample` correlates with sample noise level. If this correlation breaks (e.g., model capacity dominates), the gate becomes uninformative.

**Configs compared:** C19 (none) vs C20 (pred_only, τ=0.5)  
**Primary metrics:** `r2_joint`, `retrieval_cos@1`, `pca_axis_align_a`, `pca_axis_align_b`  
**Decision rules:**
- `r2_joint(C20) − r2_joint(C19) > 0.05` → GATE_IMPROVES_GEOMETRY
- `retrieval_cos@1(C20) − retrieval_cos@1(C19) > 0.05` → GATE_IMPROVES_RETRIEVAL
- `pca_axis_align_a(C20) − pca_axis_align_a(C19) > 0.05` → GATE_IMPROVES_AXIS_ALIGN

**Check plots:**
- [ ] val_loss curves C19 vs C20
- [ ] r2_joint, r2_a, r2_b bar: C19 vs C20
- [ ] 3D latent scatter colour-coded by u1/u2: C19 vs C20
- [ ] per-sample w_i histogram at epoch 50/150/300

---

## H2: pred_and_sparse adds no benefit over pred_only

**Claim:** `C21` (gate on pred and sparse) shows no consistent improvement over `C20` (gate on pred only); the extra sparsity gate introduces noise without signal.

**Mechanism:** `sparse_loss` is a global (per-predictor-weight) structural penalty — it does not vary per sample in a way that correlates with sample noise. Gating it effectively oscillates the regularisation strength based on noisy per-sample estimates, adding gradient variance without gain.

**Configs compared:** C20 (pred_only) vs C21 (pred_and_sparse)  
**Primary metrics:** `r2_joint`, `retrieval_cos@1`, `val_loss`  
**Decision rules:**
- `|r2_joint(C21) − r2_joint(C20)| < 0.05` → SPARSE_GATE_NEUTRAL
- `val_loss(C21) − val_loss(C20) > 0.01` → SPARSE_GATE_HURTS_LOSS

**Check plots:**
- [ ] val_loss curves C20 vs C21
- [ ] r2_joint bar: C19 vs C20 vs C21 (full comparison)

---

## H3: Gate effectiveness is robust to τ in [0.25, 1.0]

**Claim:** `r2_joint` should not vary by more than 0.05 across τ ∈ {0.25, 0.5, 1.0}, suggesting a plateau of gate effectiveness. Sharp gates (τ→0) behave like hard sample selection (brittle); soft gates (τ→∞) degenerate to uniform weighting.

### H3a — τ=0.25 not significantly worse than τ=0.5

**Configs compared:** C20 (τ=0.5) vs C22 (τ=0.25)  
**Decision rule:** `|r2_joint(C22) − r2_joint(C20)| < 0.05` → TAU_ROBUST_025  

**Check plots:**
- [ ] r2_joint vs τ bar: C22 (0.25) vs C20 (0.5) vs C23 (1.0)
- [ ] w_i distribution histograms for τ=0.25, 0.5, 1.0

### H3b — τ=1.0 not significantly worse than τ=0.5

**Configs compared:** C20 (τ=0.5) vs C23 (τ=1.0)  
**Decision rule:** `|r2_joint(C23) − r2_joint(C20)| < 0.05` → TAU_ROBUST_10  

**Check plots:**
- [ ] r2_joint vs τ bar: C22 vs C20 vs C23

---

## H4: Congruence gate benefits transfer to MLP predictor

**Claim:** `C24` (MLP + gate, τ=0.5) outperforms `C15` from B04 (MLP + no gate) by `r2_joint > +0.03`. The gate is predictor-agnostic because it operates on `pred_loss_per_sample`, not on predictor internals.

**Secondary concern:** MLP has higher capacity → `pred_loss_per` may be less correlated with sample noise → gate potentially less effective than for diagonal predictor. H4 tests whether the benefit still transfers.

**Configs compared:** C15 (B04, MLP, no gate) vs C24 (MLP, pred_only, τ=0.5)  
**Primary metrics:** `r2_joint`, `retrieval_cos@1`  
**Decision rule:** `r2_joint(C24) − r2_joint(C15) > 0.03` → GATE_TRANSFERS_TO_MLP  

**Check plots:**
- [ ] r2_joint bar: C15 (no gate) vs C24 (gate, MLP)
- [ ] Compare gate effect size: diag (H1: C19→C20) vs MLP (H4: C15→C24)

---

## H5 (exploratory, cross-batch): Uniform jac/prior better than gated jac/prior

**Claim:** Old `noise_reweighting='full'` (B02: C09, where `jac_loss` was also gated) produces worse geometry than new `congruence_mode='pred_only'` (C20), because gating `jac_loss` suppresses geometric regularisation exactly where the encoder is most uncertain.

**Mechanism:** The Jacobian penalty enforces volume-preserving, well-conditioned flow maps. On noisy samples, the encoder output is uncertain but the Jacobian still needs to be regularised to prevent degenerate geometry — suppressing it selectively creates holes in the manifold's regularisation coverage.

**Configs compared:** C09 from B02 (affine+old full reweighting) vs C20 (diag+new pred_only gate)  
**Primary metrics:** `r2_joint`, `pca_axis_align_a`, `cca_diag_score`  
**Decision rules:**
- `r2_joint(C20) − r2_joint(C09) > 0.05` → JAC_UNIFORM_BETTER
- `pca_axis_align_a(C20) − pca_axis_align_a(C09) > 0.05` → JAC_UNIFORM_AXIS_ALIGN_BETTER

**⚠️ Confounds:** C09 uses affine predictor + asymmetric noise setup; C20 uses diagonal predictor. Treat as exploratory/directional only. Main signal: are `pca_axis_align` and `cca_diag_score` systematically higher for C20?

**Check plots:**
- [ ] r2_joint, pca_axis_align bar: C09 vs C20
- [ ] CCA heatmap: C09 vs C20
- [ ] val_loss comparison curves
