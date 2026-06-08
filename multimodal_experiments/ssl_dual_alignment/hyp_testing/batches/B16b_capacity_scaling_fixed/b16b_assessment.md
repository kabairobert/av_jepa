# B16b Forensic Assessment: What Actually Failed?

## Executive Summary

The original [ANALYSIS.md](file:///gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B16b_capacity_scaling_fixed/ANALYSIS.md) claimed B16b's "jumbled colors" were a **pure visual artifact** of Gaussian rotational symmetry — the model succeeded but visualization couldn't show it. **Empirical W&B metrics disprove this.** The model genuinely failed to align the two views.

---

## 1. Evidence Inventory

### 📊 W&B Metrics Comparison (from API queries)

| Metric | B16b 16D | B16b 32D | B16c 16D | B16c 32D | B13 256x S12 |
|:-------|:---------|:---------|:---------|:---------|:-------------|
| `zjoint/r2_mean` | **0.676** | **0.672** | 0.868 | 0.808 | 0.731 |
| `za/r2_mean` | **0.438** | **0.503** | 0.712 | 0.647 | 0.617 |
| `zb/r2_mean` | **0.546** | **0.503** | 0.821 | 0.701 | 0.628 |
| `found_rank_pred_r2_1` | **0** | **0** | 2 | 2 | 2 |
| `found_rank_pred_w_3` | **0** | **0** | 2 | 2 | 2 |
| `found_rank_pred_w_5` | **0** | **0** | 2 | 2 | 2 |
| Non-zero `pred_r2_dim` count | **0/16** | **0/32** | 2/16 | 2/32 | 2/20 |
| `found_rank_pearson_1` | **3** | - | 2 | 2 | 2 |
| `found_rank_cca_3` | **3** | - | 2 | 2 | 2 |

### 🗝️ Key Observations

1. **Predictor weights collapsed to near-zero.** `found_rank_pred_w_3 = 0` means all weight magnitudes dropped below 0.3 (initialized at 1.0). This is not rotation — it's weight death.

2. **Zero predictor R².** `pred_r2_dim{i} = 0` for ALL dimensions. The affine predictor maps $z_A \mapsto w \odot z_A + b$; if $w \approx 0$, output is constant bias vector → $R^2 = 0$ regardless of rotation.

3. **Ridge R² is mediocre, not high.** `zjoint/r2_mean ≈ 0.67` vs 0.81-0.87 for working runs. The original ANALYSIS said to check this metric and expected it high. It's not.

4. **Pearson/CCA show 3 dims BUT at low threshold only.** `found_rank_pearson_1 = 3` (threshold 0.1) but `found_rank_pearson_3 = 0` (threshold 0.3). Marginal correlations exist but weak.

---

## 2. Hypothesis Assessment

### ❌ Original Hypothesis: "Gaussian Rotational Symmetry (Visual Artifact Only)"

**Claim**: Model successfully extracted 3D shared subspace; jumbled colors = arbitrary rotation $R$ of identical Gaussians; check `r2_mean` to confirm.

**Verdict: WRONG.**

Evidence against:
- `r2_mean` is 0.67 not >0.9 → subspace not well-extracted
- Predictor weights collapsed → the mechanism the model uses to align views is dead
- Working runs (B16c, B13) achieve `r2_mean` 0.73-0.87 with `found_rank_pred_r2 = 2`

> [!IMPORTANT]
> The Gaussian unidentifiability argument is **theoretically correct** — you genuinely cannot identify Gaussian ICA components. But it's describing a **secondary problem that never got the chance to manifest** because the model failed at a more fundamental level first (predictor collapse).

### ✅ Revised Hypothesis: "Predictor Weight Collapse under Gaussian Data"

**Claim**: L1 sparsity penalty (`lambda_sparse=0.1`) killed predictor weights because Gaussian data provides insufficient gradient signal to maintain them.

**Evidence for:**
- `found_rank_pred_w_3 = 0` (weights dead) in B16b
- `found_rank_pred_w_3 = 2` (weights alive) in B16c/B13 with **identical `lambda_sparse=0.1`**
- Only difference: B16b uses Gaussian latents; B16c/B13 use Uniform/spiral latents

**Mechanism** (from [losses.py L207-213](file:///gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/losses.py#L207-L213)):
```python
sparse_loss = lambda_sparse * sum(p.abs().sum() for name, p in predictor.named_parameters() if 'weight' in name)
```
L1 penalty applies constant downward pressure on every weight magnitude. The alignment gradient (from `pred_loss`) must overcome this. With Gaussian data, the alignment gradient is **diffuse** (spread equally across all rotations of the shared subspace) → per-weight gradient is weak → L1 wins → weights → 0.

---

## 3. Contra-Hypotheses & Challenges

### 🔭 Challenge 1: "Maybe Gaussian data is just harder, not impossible"

Could the model succeed with more epochs, lower `lambda_sparse`, or different learning rate?

**Assessment**: Plausible. The L1-vs-gradient balance is a quantitative race, not a qualitative impossibility. Reducing `lambda_sparse` from 0.1 to e.g. 0.01 might let weights survive long enough to find alignment. **But** even if weights survive, the Gaussian unidentifiability problem (original ANALYSIS) would then kick in as a secondary issue — the model would find *some* alignment but it wouldn't be the unique one matching the true latent axes.

### 🔭 Challenge 2: "Ridge R² of 0.67 is not terrible — maybe partial success?"

`zjoint/r2_mean = 0.67` means ridge regression from concatenated latents $[z_A, z_B]$ to true $u_s$ achieves 67% variance explained. Isn't that partial success?

**Assessment**: Misleading. Ridge operates on the **joint** representation. Even if the two flow encoders independently map to good density-matching representations (which the EBM prior forces), ridge can extract some shared signal from the concatenated space. This doesn't require the **predictor** to work — it just requires the flows to individually preserve some shared structure. The prior+Jacobian loss alone (without any predictor) would still produce representations where ridge can find *some* correlation.

**Key distinction**: Working runs achieve 0.73-0.87 with active predictors. The 0.67 could be the "baseline" from prior+Jacobian alone without predictor alignment.

### 🔭 Challenge 3: "Why does B16c work if lambda_sparse is the same?"

Both B16b and B16c use `lambda_sparse=0.1`. B16c works, B16b doesn't. If sparsity is the killer, why only B16b?

**Assessment**: The **data distribution** determines the alignment gradient magnitude:
- **B16c** (`3d-3f-2c-mlp`): Uses Uniform $u_1, u_2$ mapped through spiral/wave → non-Gaussian marginals after MLP → sharp features → strong alignment gradient on specific dimensions → gradient > L1 pressure on 2 dims → weights survive on exactly 2 dims
- **B16b** (`nd-kf-mlp`): Uses Gaussian $u_1, u_2, u_3$ → rotationally symmetric → alignment gradient diffused equally over all rotations → per-dimension gradient ≈ (total gradient) / D → each dim's gradient < L1 threshold → all weights die

This is exactly the ICA non-identifiability problem, but manifesting as an **optimization failure** rather than a "the solution exists but isn't unique" issue.

### 🔭 Challenge 4: "CCA found_rank_cca_3 = 3 at threshold 0.3"

B16b shows `found_rank_cca_3 = 3` (3 CCA components with correlation > 0.3). Doesn't this mean the flows DID align 3 dimensions?

**Assessment**: Weak evidence. CCA finds the **optimal linear projection** to maximize correlation. Even weakly correlated high-dimensional data can show multiple CCA components above 0.3 if the total signal is spread thinly. Compare: B16c has `found_rank_cca_3 = 2` — fewer components but stronger (concentrated on the true shared dims). B16b's 3 weak components suggest diffuse, unfocused correlation — consistent with the flows individually encoding shared info but not explicitly aligning it via the predictor.

### 🔭 Challenge 5: "Could there be a different bug (not sparsity)?"

**Assessment**: The sparsity explanation is parsimonious — it explains WHY B16b fails while B16c works with identical architecture, loss weights, and training setup. The only diff is data distribution. But we should verify:

> [!WARNING]
> **🔓 Open: Actual predictor weight values not yet retrieved.** W&B API timed out on detailed `geom/pred_w_dim{i}` queries. `found_rank_pred_w_3 = 0` proves max|w| < 0.3, but we don't know if weights are literally 0.0 (L1-killed) or ~0.2 (weak but alive). This matters for the mechanism:
> - Literal 0.0 → L1 hard-thresholding (classic LASSO behavior)
> - ~0.1-0.2 → weights survived but too weak to align → optimization got stuck
>
> **To resolve**: Load B16b checkpoint and inspect `predictor_a2b.weight` directly.

---

## 4. Causal Chain

```
Gaussian shared latents (u_s ~ N(0,I))
    ↓
Random MLP mixes shared + unique → all D output dims carry ≈equal shared signal
    ↓
EBM prior forces z → N(0,I_D) → all dims variance 1.0
    ↓
Predictor alignment gradient diffused across all D dims equally (rotational symmetry)
    ↓
Per-dim gradient magnitude ≈ O(1/D) × (total shared signal)
    ↓
L1 penalty (lambda_sparse=0.1) provides constant O(0.1) downward pressure per weight
    ↓
For D >> 1: gradient per dim << L1 pressure → all weights shrink to 0
    ↓
Predictor dead → no cross-view alignment → EBM loss = prior + jacobian only
    ↓
Both views independently match N(0,I) but NOT aligned to each other
```

vs B16c:
```
Uniform latents (u_1, u_2 ~ U[0,1]) + spiral/wave topology
    ↓
Non-Gaussian marginals after MLP → rotational symmetry broken
    ↓
Alignment gradient concentrated on ≤k specific dimensions
    ↓
Per-dim gradient on active dims >> L1 pressure → 2 weights survive
    ↓
Predictor alive → cross-view alignment achieved on 2 dims
```

---

## 5. Summary Table

| Aspect | Original ANALYSIS | Revised Assessment |
|:-------|:-----------------|:-------------------|
| **Core claim** | Visual artifact only | Real model failure |
| **Predictor status** | Working (implied) | Collapsed (weights → 0) |
| **R² interpretation** | "Check r2_mean, expect high" | 0.67 = low, confirms failure |
| **Gaussian rotation** | Primary cause | Secondary issue (never reached) |
| **Fix: uniform dist** | Breaks visual ambiguity | Breaks optimization deadlock |
| **Fix: reduce sparse** | Not mentioned | Alternative/complementary fix |

---

## 6. 🔓 Open Questions

1. **Exact weight values**: Load checkpoint to see if weights are literal 0.0 (L1 hard threshold) or small nonzero (~0.1-0.2)
2. **Loss component history**: How did `pred_loss`, `sparse_loss`, `prior_loss`, `jac_loss` evolve over 100 epochs? Did predictor loss plateau early (suggesting it tried but couldn't beat L1)?
3. **B16b 64D**: Does the same pattern hold? Expected: worse (larger D = more diffused gradient)
4. **Ablation**: Would `lambda_sparse=0.0` with Gaussian data produce a model with nonzero predictor weights but scrambled axes (confirming rotation is secondary)?
5. **Baseline R²**: What `zjoint/r2_mean` do you get from a model trained with `lambda_pred=0.0` (no predictor at all)? If ~0.67, confirms B16b predictor added nothing.
