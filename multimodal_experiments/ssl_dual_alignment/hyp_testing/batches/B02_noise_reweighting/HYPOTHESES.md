# B02: Noise Reweighting → Robustness Under High Noise

**Focus question:** Does per-sample reweighting based on cross-modal predictability improve geometry quality and disentanglement under high noise conditions?

**Configs used:** C03 (low noise baseline), C07 (high noise, no reweighting), C08 (high noise, pred_only), C09 (high noise, full)

---

## H6: High noise degrades geometry quality vs low noise baseline

**Claim:** AffinePredictor + L1 prior under high noise (C07) produces significantly lower `r2_mean` and `retrieval_l2_at_1` than the same config under low noise (C03), confirming the noise sensitivity of the current training objective.

**Configs compared:** C03 (affine+l1, low noise) vs C07 (affine+l1, high noise, no reweighting)

**Primary metrics:** `r2_mean`, `retrieval_l2_at_1`

**Expected outcome:** C03 > C07 (high noise hurts)

**Decision rule:** `r2_mean(C03) > r2_mean(C07) + threshold` → SUPPORTED (noise hurts)

**Check plots:**
- [ ] Training loss curves C03 vs C07 (noise → instability?)
- [ ] 3D latent scatter C07 (how corrupted is geometry?)

---

## H7: pred_only reweighting partially recovers geometry under high noise

**Claim:** `noise_reweighting=pred_only` (C08) recovers some geometry quality lost in C07 by down-weighting unpredictable (noisy) pairs at the predictor loss only, while leaving the flow geometry unaffected.

**Configs compared:** C07 (high noise, none) vs C08 (high noise, pred_only)

**Primary metrics:** `r2_mean`, `retrieval_l2_at_1`, `align_mse_normalized`

**Expected outcome:** C08 > C07 on `r2_mean` and `retrieval_l2_at_1`

**Decision rule:** `r2_mean(C08) > r2_mean(C07) + threshold` → SUPPORTED

**Check plots:**
- [ ] Per-sample weight distribution during training for C08 (are noisy samples actually down-weighted?)
- [ ] `z_a_norm_mean` for C07 vs C08 (does pred_only protect unimodal geometry?)

---

## H8: full reweighting recovers more geometry than pred_only but risks unimodal integrity

**Claim:** `noise_reweighting=full` (C09) achieves higher `r2_mean` than `pred_only` (C08) by also down-weighting noisy pairs in the flow geometry loss, but may show lower `z_a_norm_mean` / `z_b_norm_mean` indicating the prior shrinks unimodal-specific dimensions.

**Configs compared:** C08 (pred_only) vs C09 (full)

**Primary metrics:** `r2_mean`, `retrieval_l2_at_1`, `z_a_norm_mean`, `z_b_norm_mean`

**Expected outcome:** C09 ≥ C08 on `r2_mean`, but C09 < C08 on `z_a_norm_mean` (unimodal compression)

**Decision rules:**
- `r2_mean(C09) > r2_mean(C08) + threshold` → full OUTPERFORMS pred_only
- `z_a_norm_mean(C09) < z_a_norm_mean(C08) - threshold` → unimodal geometry compressed (⚠️ check)

**Check plots:**
- [ ] 3D latent scatter C08 vs C09 (residual unimodal dims preserved?)
- [ ] CCA per-dim for C08 vs C09
