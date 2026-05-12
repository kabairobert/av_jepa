# B04: One-Stage vs Two-Stage Training Under High Noise

**Focus question:** Does two-stage training universally improve cross-modal geometry and retrieval under high noise, and how is this modulated by predictor type and prediction-loss type?

**Configs used:** C11–C18 (all 3D-2f, high noise: ext_noise=0.15, asym_rate=0.10)  
**Predecessor batch:** B03 tested two-stage for MLP at low noise (C05 vs C06). B04 extends to high noise across four predictor/pred-loss combinations and adds geometry diagnostics.

**Background:** Two-stage training decouples flow geometry (stage 1: prior only) from cross-modal alignment (stage 2: predictor on frozen flows). Under high noise, stage 1 must build a clean unimodal manifold before alignment pressure is introduced. If noise corrupts stage 1 geometry, stage 2 alignment inherits that corruption; if two-stage successfully shields geometry from noisy cross-modal gradients, the manifold should be more axis-aligned and disentangled than one-stage.

---

## H1: Two-stage training outperforms one-stage across all four predictor/loss pairs

Four sub-claims, one per config pair. Each tests `r2_joint` and `retrieval_cos@1` with a +0.05 threshold.

### H1a — Diagonal + L1 prior + L1 pred

**Claim:** C12 (two-stage) > C11 (one-stage) on `r2_joint` and `retrieval_cos@1`.

**Configs compared:** C11 (1-stage) vs C12 (2-stage)  
**Primary metrics:** `r2_joint`, `retrieval_cos@1`  
**Decision rule:** `r2_joint(C12) − r2_joint(C11) > 0.05` → SUPPORTED  

**Check plots:**
- [ ] val_loss curves C11 vs C12
- [ ] 3D latent scatter: axis-alignment comparison C11 vs C12

### H1b — Diagonal + L1 prior + L2 pred

**Claim:** C14 (two-stage) > C13 (one-stage).  
**Configs compared:** C13 vs C14  
**Decision rule:** same thresholds as H1a  

**Check plots:**
- [ ] val_loss curves C13 vs C14
- [ ] 3D latent scatter: axis-alignment comparison C13 vs C14

### H1c — MLP + L2 prior + L1 pred

**Claim:** C16 (two-stage) > C15 (one-stage).  
**Configs compared:** C15 vs C16  
**Decision rule:** same thresholds  

**Check plots:**
- [ ] val_loss curves C15 vs C16
- [ ] 3D latent scatter: axis-alignment comparison C15 vs C16

### H1d — MLP + L2 prior + L2 pred

**Claim:** C18 (two-stage) > C17 (one-stage).  
**Configs compared:** C17 vs C18  
**Decision rule:** same thresholds  

**Check plots:**
- [ ] val_loss curves C17 vs C18
- [ ] 3D latent scatter: axis-alignment comparison C17 vs C18

---

## H2: MLP predictor gains more from two-stage than diagonal (under L1 pred-loss)

**Claim:** The delta `r2_joint` from two-stage is larger for MLP (C15→C16) than for diagonal (C11→C12), because MLP can exploit the clean geometry built in stage 1 more than a diagonal predictor.

**Configs compared (1-stage references):** C11 (diag) vs C15 (MLP)  
**Primary metrics:** `r2_joint`, `r2_a`, `r2_b`  
**Decision rule:** `r2_joint(C15) − r2_joint(C11) < 0.10` → PRED_TYPE_MINOR_AT_1STAGE (gate for whether predictor type already separates the configs pre-two-stage)

**Check plots:**
- [ ] Delta r2_joint: (C16−C15) vs (C12−C11) — MLP gain vs diagonal gain from two-stage
- [ ] r2_a and r2_b separately for all four pairs

---

## H3: L2 pred-loss hurts more in one-stage than L1 pred-loss (diagonal predictor)

**Claim:** `r2_joint(C11) > r2_joint(C13) + 0.05` — L1 pred-loss is more noise-robust than L2 in the one-stage regime, because L2 amplifies large-error samples that correlate with noisy data. Two-stage should partially close this gap.

**Configs compared:** C11 (diag+L1pred, 1-stage) vs C13 (diag+L2pred, 1-stage)  
**Primary metrics:** `r2_joint`, `val_loss`  
**Decision rule:** `r2_joint(C11) − r2_joint(C13) > 0.05` → L1PRED_BETTER_IN_1STAGE  

**Check plots:**
- [ ] Delta r2_joint one-stage: L1pred (C11) vs L2pred (C13)
- [ ] Delta r2_joint two-stage: L1pred (C12) vs L2pred (C14) — does gap shrink?

---

## H4: High noise degrades one-stage geometry but two-stage partially recovers it

**Claim:** C11 (high noise, 1-stage) < C03 from B01 (low noise, 1-stage); C12 (high noise, 2-stage) recovers toward C03. Recovery threshold: `r2_joint(C12) − r2_joint(C11) > 0.08`.

**Configs compared:** C11 vs C12 (primary); C03 (B01 low-noise reference) as context  
**Primary metrics:** `r2_joint`, `r2_a`, `r2_b`  
**Decision rule:** `r2_joint(C12) − r2_joint(C11) > 0.08` → TWO_STAGE_RECOVERS_UNDER_NOISE  

**Check plots:**
- [ ] C03 (B01, low-noise) vs C11 (high-noise 1-stage) vs C12 (high-noise 2-stage) — r2_joint bar
- [ ] val_loss convergence curves all three

---

## H5 (exploratory): L1 prior+diagonal vs L2 prior+MLP under high noise one-stage

**Claim:** Descriptive comparison of C11 vs C15 to characterise how much of the variance in one-stage high-noise results is explained by prior type vs predictor type. No strong directional prediction — results inform future batch design.

**Configs compared:** C11 vs C15  
**Primary metrics:** `r2_joint`, `r2_a`, `r2_b`, `val_loss`  
**Decision rule:** `|r2_joint(C11) − r2_joint(C15)| < 0.20` → COMPARABLE_SETTINGS  

**Check plots:**
- [ ] Descriptive: r2_a, r2_b, r2_joint bar chart for C11 vs C15

---

## H_geo: Two-stage produces more axis-aligned unimodal geometry

### H_geo_P1 — Diagonal pair (P1: C11 vs C12)

**Claim:** Two-stage yields higher `pca_axis_align_a/b` (>+0.10) for the diagonal predictor.

**Check plots:**
- [ ] PCA eigenvector plot: C11 vs C12 — are top-2 PCs parallel to coord axes?
- [ ] 3D scatter colour-coded by u1 and u2 separately: C11 vs C12

### H_geo_P3 — MLP pair (P3: C15 vs C16)

**Claim:** Same for MLP predictor.

**Check plots:**
- [ ] PCA eigenvector plot: C15 vs C16
- [ ] 3D scatter colour-coded by u1 and u2 separately: C15 vs C16

---

## H_disent: Two-stage yields better per-dim disentanglement of u1/u2

Expected pattern: dim0→u1, dim1→u2, dim2→noise (low R²). Two-stage should sharpen this assignment.

### H_disent_P1 — C11 vs C12 (diagonal)

**Primary metrics:** `r2_dim0_u1`, `r2_dim1_u2`, `r2_dim2_noise`  
**Decision rules:**
- `r2_dim0_u1(C12) − r2_dim0_u1(C11) > 0.05` → DIM0_U1_IMPROVED
- `r2_dim1_u2(C12) − r2_dim1_u2(C11) > 0.05` → DIM1_U2_IMPROVED
- `r2_dim2_noise(C12) < 0.10` → NOISE_DIM_CLEAN

**Check plots:**
- [ ] Per-dim R² bar: dim0→u1, dim1→u2, dim2→noise for C11 vs C12

### H_disent_P3 — C15 vs C16 (MLP)

Same structure as H_disent_P1, applied to MLP pair.

**Check plots:**
- [ ] Per-dim R² bar: dim0→u1, dim1→u2, dim2→noise for C15 vs C16

---

## H_align: Two-stage improves cross-modal CCA diagonality and retrieval

### H_align_P1 — C11 vs C12 (diagonal)

**Claim:** Two-stage → higher `cca_diag_score`, `retrieval_cos@1`, `retrieval_cos@5` (+0.05 threshold).

**Check plots:**
- [ ] CCA correlation matrix heatmap: C11 vs C12
- [ ] Retrieval@1 and @5 bar: C11 vs C12

### H_align_P3 — C15 vs C16 (MLP)

Same structure applied to MLP pair.

**Check plots:**
- [ ] CCA correlation matrix heatmap: C15 vs C16
- [ ] Retrieval@1 and @5 bar: C15 vs C16
