# B12 — Data Scaling Alignment Hypotheses (Constant Epochs)

This batch investigates the hypothesis that increasing dataset size prevents the model from overfitting to coincidental noise alignments (finite dataset noise) and forces it to learn the true shared 2D manifold, when training for a **constant 150 epochs** (allowing updates to scale with dataset size).

---

## Hypotheses

### H1 — Data Scaling Overcomes Noise Overfitting (Constant Epochs)
* **Question:** Does scaling the dataset size improve the model's ability to isolate the shared manifold from external noise when epochs are kept constant?
* **Mechanism:** With small datasets (e.g., 4k points), high-dimensional noise patterns can coincidentally correlate across modalities, leading the network to overfit to the noise rather than finding the true underlying signal. When we keep epochs constant at 150, scaling up the dataset size (up to 1,048,576 points at 256x) also scales the number of gradient update steps. This joint increase in data diversity and training time breaks the spurious finite-dataset correlations, allowing the normalising flows to accurately isolate the genuine shared 2D manifold.
* **Claim:** At low data scales (1x), the model overfits to coincidental correlations in the external noise due to data scarcity and low updates. Increasing data scale to 256x breaks these spurious correlations, resulting in significantly higher retrieval accuracy and robust manifold isolation.
