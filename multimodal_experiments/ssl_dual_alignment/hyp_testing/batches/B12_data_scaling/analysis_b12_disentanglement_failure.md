# B12 Disentanglement Failure Analysis

## Summary of Observations

| Condition | 10D | 20D |
|---|---|---|
| Orthogonal Rotation | ✅ Works at all scales | ✅ Works at all scales |
| Random MLP, 10D | ✅ Works at 1x (4k) | N/A |
| Random MLP, 20D | N/A | ❌ Fails even at 256x (1M pts) |

> [!IMPORTANT]
> The core pattern: **volume-preserving** (rotation) transforms succeed; **volume-collapsing** (Random MLP) transforms fail at 20D. This is the central clue.

---

## Root Cause Analysis

### 🔩 Mechanism 1: The Random MLP Destroys the Manifold's Geometric Separability

The data pipeline does:
1. Generate 3D base manifold (spiral + wave) with 2 shared factors + 1 unique per modality
2. Pad to 20D with i.i.d. Gaussian noise (17 noise dims)
3. Apply frozen `_FrozenMLP(20) → 20` independently per modality

**The critical issue:** The Random MLP (3 layers, hidden=64, GELU) takes 20D input and outputs 20D. It's a **non-linear mixing** of *all* 20 input dimensions — signal AND noise — into *every* output dimension. After the MLP:

- **Every output dim is a nonlinear function of 3 signal dims + 17 noise dims**
- The SNR per output dim is roughly **3/20 = 15%** of the input variance being signal
- Two *different* random MLPs applied to modalities A and B create **two completely different nonlinear entanglements** of the same underlying signal with different noise

💭**Assume:** With orthogonal rotation, the signal is still linearly separable (just rotated). The flow's additive coupling layers can undo linear rotations relatively easily. But with the Random MLP, the signal is **nonlinearly buried** — you need to invert a random neural network, which requires much more expressive capacity.

### 🔩 Mechanism 2: Flow Architecture is Too Shallow for This Task

The current flow: `6 stages × (Reflection + AdditiveCoupling + Permutation + AdditiveCoupling + Permutation + ActNorm)` = 6 stages with **additive** coupling.

> [!WARNING]
> **Additive coupling layers cannot change volume** (Jacobian determinant = 0 for all couplings). The entire flow is volume-preserving by construction. It can only rearrange and shift coordinates — it cannot selectively compress or discard dimensions.

This is **exactly the wrong inductive bias** for this task:

| Property | Rotation embedding | MLP embedding |
|---|---|---|
| Information distribution | Signal on 3 axes, noise on 17 | Signal smeared across all 20 |
| Required transform | Re-rotation (linear, volume-preserving) | Nonlinear compression (volume-collapsing) |
| Flow capability | ✅ Additive coupling can approximate | ❌ Cannot collapse dims by design |

The flow *preserves all information* — it cannot learn to "ignore" the 17 noise dims because it has no mechanism to compress them. It maps 20D input → 20D output preserving volume. The JEPA predictor then has to work in a 20D space where signal and noise are equally spread.

### 🔩 Mechanism 3: Affine Predictor is Too Weak for the Residual Problem

The `AffinePredictor` does `z_B ≈ w ⊙ z_A + b`. This is a per-dimension independent scaling. 

**Your expectation was correct in principle:** sparsity on weights should push non-shared dims to 0. But the problem is *upstream*:

- After the flow, the shared signal is **still entangled across all 20 dims** (because the flow can't disentangle the MLP's nonlinear mixing)
- The affine predictor can only work *per-dim*. Even if 2 latent dims contain most of the shared variance, they also contain noise variance from the MLP mixing
- So the predictor can't cleanly zero-out dims — every dim has *some* signal, making the optimal w vector dense rather than sparse

**Expected predictor weights pattern:**
- ✅ Rotation: Clear binary pattern (2 dims with |w|≈1, 18 dims with |w|≈0)
- ❌ MLP: All weights similar magnitude (everything is mixed, can't separate)

### 🔩 Mechanism 4: 10% Common Dimensions is NOT the Bottleneck

At 20D with `external_noise_ratio: 0.1`, you have:
- 3 signal dims embedded in 20D (3/20 = 15% of dims carry signal before MLP)
- After MLP mixing, this becomes ~15% of variance per output dim

This is NOT fundamentally too low. The rotation case succeeds at the same ratio. **The issue is the nonlinear mixing**, not the fraction.

---

## Lessons from Rombach et al. (2020) — cINN for Network-to-Network Translation

> [!TIP]
> Rombach's paper is directly relevant but reveals a **crucial difference** in problem setup.

### What Rombach does differently

1. **Conditional INN (cINN):** Their flow is *conditioned* on the source representation z_Φ. The flow maps `z_Θ → v | z_Φ` where v ~ N(0,1). The conditioning allows the flow to use z_Φ information when transforming z_Θ.

2. **20 coupling blocks** with affine coupling (not additive): Their architecture is significantly deeper and uses **affine coupling** which *can* change volume (via learned scale factors).

3. **Fixed pretrained representations:** Both source and target representations are from powerful pretrained models (BERT, BigGAN, ResNet). These already have meaningful structure. Your frozen random MLP creates *adversarial* structure — maximally difficult.

4. **KL divergence loss:** They minimize `E[KL(p(v|z_Φ) || N(0,1))]` which explicitly trains the flow to make the residual independent of the conditioning. This is fundamentally different from your JEPA mutual-prediction objective.

### 🔭 Hypothesis from Rombach: The key paper finding

> *"Deterministic MLPs fail to capture ambiguities in deeper layers and collapse to predicting the mean image."*

Rombach explicitly shows MLPs fail for this translation task — you need invertible networks. But their success relies on:
- **Affine (not additive) coupling** — volume change is possible
- **Conditioning** — the flow sees both modalities
- **Deep architecture** — 20 blocks vs your 6 stages
- **KL to unit Gaussian** — explicit independence objective

---

## Ranked Hypotheses for Why B12 Fails

### H1: Flow is volume-preserving → cannot discard noise dims (🥇 Most Likely)

**Evidence:** Rotation works (volume-preserving sufficient), MLP fails (need volume-collapsing).

**Test:** Replace additive coupling with **affine coupling** (learnable scale+shift). This adds per-dim scaling to the coupling, allowing the flow to compress/expand different dims. The Jacobian determinant becomes non-zero, which the existing `lambda_jac` loss term already handles.

### H2: Flow is too shallow for nonlinear MLP inversion (🥈)

**Evidence:** 6 stages with hidden=128 may not have enough capacity to approximate the inverse of a 3-layer MLP(20→64→64→20).

**Test:** Increase `stage_count` from 6 to 12 or 20. Also increase `hidden_units` from 128 to 256.

### H3: Affine predictor can't handle entangled representations (🥉)

**Evidence:** Per-dim predictor assumes disentangled representation; if flow can't disentangle, predictor is also stuck.

**Test:** Use `MLPPredictor` or `BlockDiagonalPredictor` (block_size=4 or 10) to allow cross-dim prediction. This partially compensates for imperfect disentanglement by the flow.

### H4: Loss weighting doesn't provide enough signal for selective dimension suppression

The current config has `lambda_sparse: 0.1` and `congruence_mode: none`.

**Test:** 
- Increase `lambda_sparse` to 0.5 or 1.0
- Enable `congruence_mode: pred_only` to downweight high-error samples
- Consider adding a **mutual information minimization** term between predicted and unpredicted subspaces

### H5: The frozen MLP distortion is fundamentally too adversarial for this architecture

💭**Assume:** A random MLP with Xavier init and GELU creates a distortion that is *maximally hard* for a shallow additive-coupling flow to invert. Real modalities (audio, video) have much more structured distortions.

**Implication:** This may be testing the wrong thing. The MLP distortion is harder than any real-world scenario. Consider:
- Using a **1-layer linear MLP** (random matrix, no nonlinearity) as an intermediate difficulty
- Using a **random MLP with bottleneck** (20→8→20) to make the signal easier to recover

---

## Proposed Experiments (Priority Order)

### Experiment B13a: Affine Coupling Flow
Replace `AdditiveCoupling` with a coupling that has learned scale AND shift. This is the single most impactful change — it breaks the volume-preservation constraint.

### Experiment B13b: Deeper + Wider Flow
Keep additive coupling but test `stage_count: 12, 16, 20` and `hidden_units: 256`. Tests if raw capacity is the bottleneck.

### Experiment B13c: MLP Predictor
Switch `predictor_type: mlp` with hidden_dim=64. If the flow produces a partially-disentangled representation, the MLP predictor can learn the residual cross-dim mapping.

### Experiment B13d: Bottleneck MLP Distortion
Replace the frozen `_FrozenMLP(20, hidden=64)` with a bottleneck `_FrozenMLP(20→8→20)` to see if the issue is recoverability of the signal.

### Experiment B13e: Higher Sparsity + Congruence
Test `lambda_sparse: 0.5, 1.0` and `congruence_mode: pred_only`. Keeps architecture the same, only tunes loss.

---

## ❓ Open Questions for You

1. **Flow architecture change:** Should I implement affine coupling (with learnable scale) in the flow? This is the highest-impact change but modifies core architecture in [ssl_disentangling.py](file:///home/rkabai/github/eb_jepa_private/multimodal_experiments/initial_trials/ssl_disentangling.py).

2. **Is the Random MLP distortion meant to simulate real pretrained networks?** If yes, Rombach's approach suggests you need a *conditional* flow (sees both modalities) with affine coupling, which is architecturally different from the current dual-encoder JEPA setup.

3. **Priority:** Do you want to run all B13 experiments, or start with the highest-impact one (affine coupling)?

4. **Checkpoint inspection:** I attempted to load the B12 256x checkpoint to inspect predictor weights — if it completes, I'll add the actual weight values below. The pattern of weights will confirm whether H1 or H3 is dominant.

---

## 📊 Key Diagnostic to Check (from existing runs)

Look at these WandB metrics for B12_256x:
- `geom/pred_w_dim{0..19}` — predictor weight magnitudes. If all similar → H1 confirmed (flow can't disentangle)
- `geom/found_rank_pred_w_3` — how many dims have |w| > 0.3. Should be 2 for success. If >> 2, predictor is spreading weight.
- `geom/za/flatness_ratio` — if high, flow flattened the manifold correctly. If low, flow failed upstream.
- `geom/za_zb_pearson_dim{i}` — per-dim correlation between z_A and z_B. With rotation, expect 2 high dims. With MLP, expect all low (entangled).
