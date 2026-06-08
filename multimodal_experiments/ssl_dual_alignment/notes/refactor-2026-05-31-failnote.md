# Dataset Refactor Eval Failure Note

**Issue:** Evaluating pre-refactor checkpoints (e.g., W&B run `autqos6l` trained on May 30) yields extremely high loss (~9M) and exploding latent values (~80,000), resulting in completely random/scattered plots.

**Root Cause:** The `dataset.py` refactor on May 31/Jun 3 fundamentally altered the synthetic data generation logic for `3d-3f-2c-mlp`.
1. **Architectural Mismatch:** `_FrozenMLP` hardcoded a 3-layer MLP with `hidden_dim=64`. The refactored `FrozenRandomMLP` respects the config (`mlp_depth: 2`) and dynamically sets `hidden_dim` to `embed_dim` (20).
2. **RNG Sequence Desync:** Even if the MLP architecture is restored, the refactor changed the sequence of random number generation (e.g., removing `_get_bbox()`). This shifted all subsequent `self.rng` draws, creating a slightly different base data manifold before the MLP is even applied.

Normalizing Flows act as highly sensitive affine expanders/contractors. Because the generated evaluation data points no longer perfectly align with the exact manifold the model memorized during training, passing this out-of-distribution data through the flow causes catastrophic exponential amplification of the latents.

**Take-Home Message:**
When training Normalizing Flows on deterministically generated synthetic datasets, the evaluation data distribution must match the training distribution perfectly. Any change to the RNG call sequence (even unused draws) or data generation pipeline completely invalidates the learned bijection for old checkpoints.

**Course of Action:**
Do not attempt to patch the current codebase to support old checkpoints; the RNG desync is too complex to reverse-engineer cleanly.
- **Option A:** To evaluate `autqos6l`, use `git checkout` to revert to the exact commit the model was trained on (prior to the May 31 dataset refactor) and run the evaluation there.
- **Option B:** Train a new model from scratch using the modernized dataset generation logic.
