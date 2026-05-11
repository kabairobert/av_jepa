# B03: Two-Stage Training → Predictor Quality

**Focus question:** Does decoupling flow training from predictor training (two-stage) improve alignment quality for expressive predictors like MLP, compared to joint one-stage training?

**Configs used:** C05 (mlp+l2, one-stage), C06 (mlp+l2, two-stage)

**Background:** In one-stage training, the MLP predictor can absorb cross-modal structure during flow training, preventing the flows from developing aligned geometry. Two-stage training first trains flows to convergence under the prior alone, then trains the predictor on frozen flows — testing whether a well-structured geometry makes cross-modal alignment easier for the predictor.

---

## H9: Two-stage MLP training produces better geometry than one-stage MLP

**Claim:** MLPPredictor + L2 prior trained two-stage (C06) produces higher `r2_mean`, `cca_effective_rank`, and `retrieval_l2_at_1` than the same config trained one-stage (C05), because the flows develop better unimodal geometry before alignment pressure is introduced.

**Configs compared:** C05 (mlp+l2, one-stage) vs C06 (mlp+l2, two-stage)

**Primary metrics:** `r2_mean`, `cca_effective_rank`, `retrieval_l2_at_1`

**Expected outcome:** C06 > C05 on all three metrics

**Decision rule:** `r2_mean(C06) > r2_mean(C05) + threshold` → SUPPORTED

**Check plots:**
- [ ] Training loss breakdown by stage for C06 (does stage 1 converge cleanly?)
- [ ] 3D latent scatter at end of stage 1 vs end of stage 2 for C06
- [ ] CCA per-dim for C05 vs C06 (does two-stage improve cross-modal alignment?)

---

## H10: Two-stage diagonal (oracle comparison) — unimodal geometry without alignment pressure

**Supplementary:** If H9 is SUPPORTED, a follow-up would test DiagonalPredictor two-stage vs one-stage (C01 one-stage vs a future C11 two-stage diagonal). This would answer: "does a good unimodal geometry make alignment easier for the predictor, even without alignment pressure on the flows?" — reserved for a future batch.

**Status:** Deferred to future batch B04.
