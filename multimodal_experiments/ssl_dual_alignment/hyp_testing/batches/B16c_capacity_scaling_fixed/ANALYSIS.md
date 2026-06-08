# B16c Analysis: 64D Capacity Limit

## Observation
Color/lightness dims ($u_1, u_2$) form well-aligned gradient → remain orthogonal up to 32D. Starting from 64D → structure breaks down → gradient lost.

## 🔩Mech: MLP Bottleneck Bug
Breakdown at 64D caused by artificial bottleneck bug in dataset generation code (`dataset.py`).

1. **Default Hidden Dimension**: `3d-3f-2c-mlp` dataset type passes data through random neural network to embed 3D manifold into $D$ dimensions. Done using `FrozenRandomMLP`.
2. **Bug**: In `dataset.py`, `FrozenRandomMLP` called without explicit `hidden_dim` arg. Class defaults to `hidden_dim=64`.
3. **Information Loss**: For any $D > 64$ (e.g., 128D, 256D, 512D), MLP architecture effectively became:
   `Linear(D -> 64) -> GELU -> Linear(64 -> D)`
   Forced high-dimensional data through strict 64-dimensional bottleneck → destroyed rank and topological structure before reaching model.

## Why 32D Worked
For $D \le 64$ (like 16D, 32D), `hidden_dim=64` strictly larger than input dimension → no bottleneck. Network safely projected data to 64D and back to $D$ without losing info. Unlike B16b, B16c data uses Uniform distributions ($u_1, u_2$) + non-Gaussian spiral topology → breaks rotational symmetry → allows model/visualizer perfectly align axes without rotational scrambling.

## 🔧Fix Applied
Bug patched in `dataset.py` by passing `hidden_dim=D` to `FrozenRandomMLP`:
```python
mlp_a = FrozenRandomMLP(D, D, hidden_dim=D, depth=self.mlp_depth, generator=self.torch_rng).eval()
```
Ensures capacity scales properly with embedding dimension → removes bottleneck. Future runs $\ge 64D$ retain structure.
