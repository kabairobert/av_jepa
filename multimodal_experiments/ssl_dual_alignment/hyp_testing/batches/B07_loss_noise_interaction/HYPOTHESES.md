# B07 — Loss-Noise Interaction Study

## Overview

This batch investigates how the choices of Prior (None, L1, L2) and Predictor (None, L1, L2) interact with varying regimes of noise. While previous batches established that the cross-modal Predictor drives manifold flattening and the Prior drives dimensionality suppression under low noise, B07 pushes these mechanisms to their breaking points.

We execute a $6 \times 3 \times 3$ factorial sweep across 6 noise regimes:
1. **N1:** 15% Asymmetric (Baseline)
2. **N2:** 25% Asymmetric
3. **N3:** 37.5% Asymmetric
4. **N4:** 50% External
5. **N5:** 75% External + 9% Asymmetric (High Mixed Noise)
6. **N6:** 25% Asymmetric + Increased Dense Manifold Noise (4%)

## Hypotheses

### H1 — Predictor's flattening pressure under extreme external noise
**Question:** Does the Predictor's flattening pressure hold up under extreme external noise?
**Mechanism:** The predictor was previously shown to be the main driver of geometric flattening at low noise. We test if this holds under 50% external noise (N4), where half the points have no manifold structure at all.
**Claim:** Even when half the points lack manifold structure, the Predictor (`B07_NPP401`) alone produces higher manifold flatness (`flatness_ratio_a`) than the Prior alone (`B07_NPP410`). The Predictor remains the primary flattening driver.

### H2 — L1 vs L2 Prior robustness to asymmetric noise scaling
**Question:** Is L1 Prior significantly more robust than L2 as the noise-to-signal ratio increases?
**Mechanism:** L1 induces sparsity, which acts as implicit outlier rejection. As asymmetric noise scales up, L2's smooth quadratic penalty might be overwhelmed by large corruptions, whereas L1 maintains strict dimensionality suppression.
**Claim:** As asymmetric noise increases to 37.5% (N3), L1 Prior (`B07_NPP311`) maintains dimensionality suppression (`r2_dim2_noise`) significantly better than L2 Prior (`B07_NPP321`).

### H3 — Regime change for Predictors (Dense manifold noise)
**Question:** Is there a "regime change" where L2 Predictor outperforms L1?
**Mechanism:** The L1 Predictor is highly robust to sparse or asymmetric outliers. However, when noise is dense and Gaussian (spread across the manifold itself), an L2 Predictor—which naturally models Gaussian errors—might become superior.
**Claim:** Under increased dense Gaussian manifold noise (N6), the L2 Predictor (`B07_NPP612`) achieves better cross-modal alignment (`retrieval_cos@1`, `r2_joint`) than the L1 Predictor (`B07_NPP611`).

### H4 — Drivers of common manifold extraction (Low Noise)
**Question:** What is driving the common manifold (mutual information)'s extraction and alignment? Is it mainly the predictor, the prior, or the synergy of both?
**Mechanism:** Cross-modal alignment intrinsically relies on mapping corresponding views together. The Predictor applies direct structural alignment, while the Prior shapes the independent latent spaces. 
**Claim:** At the 15% baseline noise (N1), the Predictor is the primary driver of common manifold extraction. Predictor-only (`B07_NPP101`) significantly outperforms Prior-only (`B07_NPP110`) on cross-modal retrieval, while their synergy (`B07_NPP111`) yields the optimal state.

### H5 — Synergy dependency shifts under Extreme Mixed Noise
**Question:** Is the driving force behind common manifold extraction the same across all noise levels and scenarios?
**Mechanism:** At massive noise levels (N5: 75% Ext), the predictor might collapse if it attempts to align entirely uncorrelated points without any filtering. In this regime, the Prior's structural filtering becomes a strict prerequisite for the Predictor to function.
**Claim:** The regime shifts under extreme mixed noise (N5). Neither the Predictor alone (`B07_NPP501`) nor the Prior alone (`B07_NPP510`) can effectively extract the common manifold. Only their synergy (`B07_NPP511`) yields meaningful retrieval, shifting the alignment dependency from "Predictor-driven" to "Synergy-dependent".
