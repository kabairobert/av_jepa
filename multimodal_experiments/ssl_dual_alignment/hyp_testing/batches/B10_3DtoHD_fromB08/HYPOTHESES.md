# B10 — 3D to High-Dimensional (HD) Alignment Hypotheses

This batch investigates the ability of the Dual-JEPA model to isolate a shared 2D manifold embedded inside high-dimensional (10D and 20D) spaces. It specifically tests how flow models unroll high-dimensional representations when exposed to linear (orthogonal rotation) versus non-linear (random frozen MLP) embedding strategies under varying external noise regimes.

---

## Hypotheses

### H1 — The Non-Linear HD Bottleneck (Rotation vs MLP)
* **Question:** How does the type of high-dimensional embedding affect the model's ability to unroll and isolate the manifold?
* **Mechanism:** The flow model is highly capable of inverting linear rotations via orthogonal matrices, recovering the true coordinates easily. However, a non-linear frozen MLP drastically distorts the underlying geometry across all dimensions, making the unrolling process a much harder optimisation problem for the normalising flows.
* **Claim:** The flow model effectively isolates the 2D shared manifold from high-dimensional linearly rotated spaces, but struggles significantly more when the space is distorted by a non-linear MLP.

---

### H2 — Dimensionality Degradation (10D vs 20D)
* **Question:** Does increasing the ambient space dimensionality naturally degrade alignment?
* **Mechanism:** As the dimensionality expands from 10D to 20D, the volume of the noise space grows exponentially (curse of dimensionality). The predictor and flow networks have a harder time filtering out the non-shared coordinates when the true signal (2D) represents a much smaller fraction of the total representation capacity.
* **Claim:** Increasing the ambient space from 10D to 20D degrades retrieval accuracy and increases noise leakage, particularly under high external noise (N2=30%).

---

### H3 — Synergy is Strictly Required in HD
* **Question:** Is the Predictor alone still sufficient to unroll the manifold in 20D space?
* **Mechanism:** In 3D (B08), the predictor alone provided enough gradient pressure to flatten the space. However, in 20D space, the problem is severely under-constrained. Without a sparsity prior penalising active dimensions, the predictor will fail to isolate the true 2D signal from the 18D noise dimensions.
* **Claim:** In 20D HD space, the predictor alone is insufficient. The L1 Prior + L1 Predictor synergy is strictly required to effectively isolate the 2D subspace and flatten the manifold.

---

### H4 — L1 vs L2 Axis Alignment survives HD Distortion
* **Question:** Does the L1 prior successfully enforce axis alignment even when the manifold is deeply embedded in a distorted 20D space?
* **Mechanism:** The Laplace (L1) prior acts as a coordinate-wise sparsity penalty. Even if the generative mapping (MLP) heavily rotates and distorts the features across all 20 dimensions, the L1 prior forces the inverse mapping (the flow model) to collapse the independent shared factors onto orthogonal coordinate axes, unlike the Gaussian (L2) prior which is rotationally invariant.
* **Claim:** Even when deeply buried in a 20D MLP-distorted space, the L1 Prior forces strict axis-alignment (diagonality) of the recovered 2D manifold, whereas the L2 Prior fails to achieve such alignment.
