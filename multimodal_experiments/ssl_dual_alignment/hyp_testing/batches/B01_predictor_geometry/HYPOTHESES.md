# B01: Predictor Type & Prior → Geometry Quality

**Focus question:** Does predictor expressiveness and prior type determine how well the model recovers shared latent geometry?

**Configs used:** C01, C02, C03, C04, C05, C10 (all 3D2f)

---

## H1: Laplace prior (L1) produces sparser, more axis-aligned geometry than Gaussian (L2)

**Claim:** DiagonalPredictor + L1 prior produces higher `r2_mean` and `cca_effective_rank` than DiagonalPredictor + L2 prior, because L1 encourages sparse latent codes with cleaner factor separation.

**Configs compared:** C01 (diag+l1) vs C02 (diag+l2)

**Primary metrics:** `r2_mean`, `cca_effective_rank`

**Expected outcome:** C01 > C02 on both metrics

**Decision rule:** `r2_mean(C01) > r2_mean(C02) + threshold` → SUPPORTED

**Check plots:**
- [ ] 3D latent scatter z_A and z_B for C01 vs C02
- [ ] Per-dim CCA correlation for C01 vs C02

---

## H2: AffinePredictor fixes cloth distortion vs DiagonalPredictor

**Claim:** AffinePredictor + L1 prior produces higher `r2_mean` and `retrieval_l2_at_1` than DiagonalPredictor + L1 prior, because per-dimension bias absorbs scale/offset differences between modalities that diagonal cannot.

**Configs compared:** C01 (diag+l1) vs C03 (affine+l1)

**Primary metrics:** `r2_mean`, `retrieval_l2_at_1`, `align_mse_normalized`

**Expected outcome:** C03 > C01 on `r2_mean` and `retrieval_l2_at_1`

**Decision rule:** `r2_mean(C03) > r2_mean(C01) + threshold` → SUPPORTED

**Check plots:**
- [ ] 3D latent scatter z_A and z_B for C01 vs C03 (cloth distortion visible?)
- [ ] `align_mse_normalized` training curves for C01 vs C03

---

## H3: Affine + L2 prior degrades geometry vs Affine + L1 prior

**Claim:** AffinePredictor + L2 prior produces lower `r2_mean` than AffinePredictor + L1 prior, because Gaussian prior does not encourage axis-aligned sparse factors.

**Configs compared:** C03 (affine+l1) vs C04 (affine+l2)

**Primary metrics:** `r2_mean`, `cca_effective_rank`

**Expected outcome:** C03 > C04

**Decision rule:** `r2_mean(C03) > r2_mean(C04) + threshold` → SUPPORTED

**Check plots:**
- [ ] z_A norm mean for C03 vs C04 (does L2 prior cause over-shrinkage?)

---

## H4: MLP predictor with L2 prior undoes axis alignment vs structured predictors

**Claim:** MLPPredictor + L2 prior produces lower `cca_effective_rank` and worse `retrieval_l2_at_1` than AffinePredictor + L1 prior, because MLP absorbs cross-modal structure into predictor weights rather than forcing the flows to produce aligned geometry.

**Configs compared:** C03 (affine+l1) vs C05 (mlp+l2)

**Primary metrics:** `cca_effective_rank`, `retrieval_l2_at_1`, `r2_mean`

**Expected outcome:** C03 > C05 on `cca_effective_rank` and `retrieval_l2_at_1`

**Decision rule:** `cca_effective_rank(C03) > cca_effective_rank(C05) + threshold` → SUPPORTED

**Check plots:**
- [ ] CCA per-dim correlation for C03 vs C05
- [ ] 3D latent scatter for C05 (does MLP produce unaligned geometry?)

---

## H5: L1 prediction loss vs L2 prediction loss has minor effect on geometry

**Claim:** AffinePredictor + L1 prior with L2 pred loss produces similar `r2_mean` to L1 pred loss, because the prior (not pred loss function) is the dominant geometry-shaping term.

**Configs compared:** C03 (affine+l1prior+l1pred) vs C10 (affine+l1prior+l2pred)

**Primary metrics:** `r2_mean`, `val_pred_loss`

**Expected outcome:** C03 ≈ C10 on `r2_mean` (difference < threshold)

**Decision rule:** `|r2_mean(C03) - r2_mean(C10)| < threshold` → SUPPORTED (pred loss is minor)

**Check plots:**
- [ ] `val_pred_loss` curves for C03 vs C10 (convergence speed difference?)
