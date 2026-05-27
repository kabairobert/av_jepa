# B11 — Data Scaling Alignment Hypotheses

This batch investigates the hypothesis that increasing dataset size prevents the model from overfitting to coincidental noise alignments (finite dataset noise) and forces it to learn the true shared 2D manifold, especially in distorted high-dimensional spaces.

---

## Hypotheses

### H1 — Data Scaling Overcomes Noise Overfitting
* **Question:** Does scaling the dataset size improve the model's ability to isolate the shared manifold from external noise?
* **Mechanism:** With tiny datasets (e.g., 4k points), high-dimensional noise patterns can coincidentally correlate across modalities, leading the network to overfit to the noise rather than finding the true underlying signal. Providing a massively larger dataset (e.g., 131k points) breaks these spurious finite-dataset correlations, forcing the normalising flows to rely exclusively on the genuine shared 2D manifold.
* **Claim:** At low data scales (1x), the model overfits to coincidental correlations in the external noise, failing to perfectly isolate the 2D manifold. Increasing data scale (to 32x) breaks these spurious correlations, resulting in significantly higher retrieval accuracy and lower noise leakage.

---

### H2 — The MLP Bottleneck Diminishes with Scale
* **Question:** Can a larger dataset provide the necessary gradient signal to unroll the severe non-linear distortions introduced by a random frozen MLP?
* **Mechanism:** A non-linear MLP creates a highly distorted, non-convex optimisation landscape compared to a simple linear rotation. On a small dataset, the flow model struggles to learn this complex inverse mapping. With a larger dataset, the dense sampling of the manifold provides consistent and robust gradient signals that enable the flow model to successfully unroll the MLP.
* **Claim:** At 1x scale, the MLP distortion creates a hard optimisation landscape, causing much worse performance than linear Rotation. At 32x scale, the abundance of data provides enough consistent gradient signal to unroll the MLP, significantly closing the performance gap between Rotation and MLP embeddings.

---

### H3 — Dimensionality Robustness via Data
* **Question:** Can data scaling overcome the curse of dimensionality when moving from 10D to 20D?
* **Mechanism:** As ambient dimensionality increases, the volume of the noise space grows exponentially, making the shared 2D manifold harder to locate (the curse of dimensionality). A small dataset is sparse in 20D, exacerbating this issue. A large dataset densely populates the space, allowing the synergy of the L1 Predictor and L2 Prior to reliably filter out the non-shared coordinates regardless of ambient dimension.
* **Claim:** At low data scales (1x), increasing ambient space from 10D to 20D degrades performance severely due to the curse of dimensionality. At high data scales (32x), the model becomes robust to ambient dimensionality, showing minimal degradation from 10D to 20D.
