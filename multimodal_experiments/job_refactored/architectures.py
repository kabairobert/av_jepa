import torch
import torch.nn as nn
from multimodal_experiments.initial_trials.ssl_disentangling import FlowModel, build_flow_layers

class DualPairModel(torch.nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, inputs_a, inputs_b):
        output_a, jac_a = self.model_a(inputs_a)
        output_b, jac_b = self.model_b(inputs_b)
        return torch.cat([output_a, output_b, jac_a.unsqueeze(-1), jac_b.unsqueeze(-1)], dim=1)

class DiagonalPredictor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * self.weight

class MLPPredictor(torch.nn.Module):
    def __init__(self, dim=1, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return self.net(x)
