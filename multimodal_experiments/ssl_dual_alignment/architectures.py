import torch
import torch.nn as nn
from multimodal_experiments.initial_trials.ssl_disentangling import FlowModel, build_flow_layers

class DualPairModel(torch.nn.Module):
    """Wraps Modality A and B models into a single forward pass."""
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, inputs_a, inputs_b):
        output_a, jac_a = self.model_a(inputs_a)
        output_b, jac_b = self.model_b(inputs_b)
        return torch.cat([output_a, output_b, jac_a.unsqueeze(-1), jac_b.unsqueeze(-1)], dim=1)


class EBMMJEPAModel(torch.nn.ModuleDict):
    """Flexible registry for Multi-Modal JEPA components.

    Inherits from nn.ModuleDict to ensure all registered sub-modules (encoders,
    adapters, predictors) are automatically captured in checkpoints.
    """
    def forward(self, *args, **kwargs):
        # Primary task execution (e.g. encoders) is handled by 'dual_model'
        return self['dual_model'](*args, **kwargs)


class DiagonalPredictor(torch.nn.Module):
    """Scales input dimensions independently via learnable weights."""
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * self.weight


class AffinePredictor(torch.nn.Module):
    """Per-dimension scale + bias: z_B ≈ w ⊙ z_A + b."""
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.weight * x + self.bias


class BlockDiagonalPredictor(torch.nn.Module):
    """Block-diagonal weight matrix: allows within-block mixing, not cross-block.
    
    Reserved for post-affine experiments. block_size=1 reduces to AffinePredictor.
    """
    def __init__(self, num_dims: int, block_size: int = 2):
        super().__init__()
        assert num_dims % block_size == 0, f"num_dims ({num_dims}) must be divisible by block_size ({block_size})"
        self.block_size = block_size
        n_blocks = num_dims // block_size
        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(block_size, block_size, bias=True)
            for _ in range(n_blocks)
        ])
        # Near-identity initialization
        for b in self.blocks:
            torch.nn.init.eye_(b.weight)
            if b.bias is not None:
                torch.nn.init.zeros_(b.bias)

    def forward(self, x):
        chunks = x.split(self.block_size, dim=-1)
        return torch.cat([b(c) for b, c in zip(self.blocks, chunks)], dim=-1)


class MLPPredictor(torch.nn.Module):
    """Standard MLP cross-modal predictor.

    Uses LayerNorm instead of BatchNorm1d: correct at any batch size (including
    batch=1 during eval) and normalises across features rather than batch samples,
    which is more appropriate for a per-sample cross-modal mapping.
    """
    def __init__(self, dim=1, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim)
        )
        # Small weight initialization for neutral starting alignment
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
