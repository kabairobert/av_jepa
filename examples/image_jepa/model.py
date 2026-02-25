"""
Model definitions for image_jepa (CIFAR-10).

Kept in a separate module so visualization and notebook utilities can import
just the model without pulling in training-only dependencies (fire, wandb, etc.).
"""

import torch.nn as nn
import torchvision
from torchvision.models import VisionTransformer


class ResNet18(nn.Module):
    """ResNet-18 backbone implementation."""

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.features_dim = 512

    def forward(self, x):
        return self.backbone(x)


class ImageSSL(nn.Module):
    """Image Self-Supervised Learning model implementation."""

    def __init__(
        self, backbone, features_dim, proj_hidden_dim=2048, proj_output_dim=2048
    ):
        super().__init__()
        self.backbone = backbone
        self.features_dim = features_dim

        # Projector: 3-layer MLP with BN
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def build_model(cfg):
    """Build and return an ImageSSL model (without moving to device)."""
    if cfg.model.type == "resnet":
        backbone = ResNet18()
        features_dim = backbone.features_dim
    elif cfg.model.type == "vit_s":
        features_dim = 384
        backbone = VisionTransformer(
            image_size=32, patch_size=8, hidden_dim=features_dim,
            num_layers=12, num_heads=6, mlp_dim=4 * features_dim,
        )
        backbone.heads = nn.Identity()
    elif cfg.model.type == "vit_b":
        features_dim = 768
        backbone = VisionTransformer(
            image_size=32, patch_size=8, hidden_dim=features_dim,
            num_layers=12, num_heads=12, mlp_dim=4 * features_dim,
        )
        backbone.heads = nn.Identity()
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    model = ImageSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim,
    )
    if not cfg.model.use_projector:
        model.projector = nn.Identity()
    return model, features_dim


def build_linear_probe(features_dim, num_classes=10):
    """Build and return a LinearProbe (without moving to device)."""
    from examples.image_jepa.eval import LinearProbe
    return LinearProbe(feature_dim=features_dim, num_classes=num_classes)
