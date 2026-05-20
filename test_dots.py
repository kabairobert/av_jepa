import torch
from multimodal_experiments.ssl_dual_alignment.dataset import DualDisentangleDataset
import numpy as np

dset = DualDisentangleDataset(
    data_type='3d-2f-common',
    num_samples=4096,
    manifold_noise_a=0.02,
    manifold_noise_b=0.02,
    asymmetric_noise_magnitude=0.1,
    asymmetric_noise_rate_a=0.15,
    asymmetric_noise_rate_b=0.15,
    external_noise_ratio=0.0,
    noise_bbox_expansion=0.25,
    seed=12345
)

pts_a = dset.point_type_a
print("Unique point types A:", np.unique(pts_a))
print("Counts A:", np.bincount(pts_a))

pts_b = dset.point_type_b
print("Unique point types B:", np.unique(pts_b))
print("Counts B:", np.bincount(pts_b))

param_values = dset.param_values
print("Min param value:", param_values.min(axis=0))
print("Max param value:", param_values.max(axis=0))
