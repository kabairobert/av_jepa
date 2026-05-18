import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
import wandb

# ---- Point color helpers ----
# point_type codes:
#   0 = manifold
#   1 = asym_a_good  (A = manifold side)
#   2 = asym_b_corrupt (B = noisy side)  -> gray
#   3 = asym_b_good  (B = manifold side)
#   4 = asym_a_corrupt (A = noisy side)  -> gray
#   5 = external (both sides)            -> near-black #1a1a1a (90% darkness)

_GRAY_CORRUPT = '#808080'
_BLACK_EXTERNAL = '#000000'   # Pure black to distinguish from dark blue turbo(0)


def _get_point_colors(param_values_1d, point_types, cmap='rainbow'):
    """Per-point color: manifold=rainbow, corrupted=gray, external=near-black."""
    norm_p = (param_values_1d - param_values_1d.min()) / (
        param_values_1d.max() - param_values_1d.min() + 1e-12)
    
    colormap = cm.get_cmap(cmap)
    gray_rgba = mcolors.to_rgba(_GRAY_CORRUPT)
    black_rgba = mcolors.to_rgba(_BLACK_EXTERNAL)
    colors = []
    for i, pt in enumerate(point_types):
        if pt == 5:
            colors.append(black_rgba)
        elif pt in (2, 4):   # corrupted / noise side of asymmetric pair
            colors.append(gray_rgba)
        else:                 # manifold (0), asym_a_good (1), asym_b_good (3)
            colors.append(colormap(float(norm_p[i])))
    return colors


def plot_original_spaces(data_a, data_b, param_values,
                         point_type_a=None, point_type_b=None, axis_box=None):
    """Plots raw Modality A and Modality B datasets with noise-aware coloring."""
    is_3d = data_a.shape[1] >= 3
    fig = plt.figure(figsize=(12, 6))

    # For 2D param_values (multi-factor case), extract first factor for coloring
    if isinstance(param_values, np.ndarray) and param_values.ndim == 2:
        param_values_1d = param_values[:, 0]
    else:
        param_values_1d = param_values

    # Build per-point color arrays
    pt_a = point_type_a if point_type_a is not None else np.zeros(len(data_a), dtype=np.int32)
    pt_b = point_type_b if point_type_b is not None else np.zeros(len(data_b), dtype=np.int32)
    c_a = _get_point_colors(param_values_1d, pt_a)
    c_b = _get_point_colors(param_values_1d, pt_b)

    if is_3d:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(data_a[:, 0], data_a[:, 1], data_a[:, 2],
                    c=c_a, s=5, alpha=0.5)
        ax1.set_title('Modality A')
        ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2'); ax1.set_zlabel('Dim 3')
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(data_b[:, 0], data_b[:, 1], data_b[:, 2],
                    c=c_b, s=5, alpha=0.5)
        ax2.set_title('Modality B')
        ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2'); ax2.set_zlabel('Dim 3')
    else:
        ax1 = fig.add_subplot(121)
        ax1.scatter(data_a[:, 0], data_a[:, 1], c=c_a, alpha=0.5)
        ax1.set_title('Modality A')
        ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2')
        ax1.axis('equal')
        ax2 = fig.add_subplot(122)
        ax2.scatter(data_b[:, 0], data_b[:, 1], c=c_b, alpha=0.5)
        ax2.set_title('Modality B')
        ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2')
        ax2.axis('equal')

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
    return fig


def plot_dual_geometry_reshaping_view(dual_model, data_a, data_b, param_values, device,
                                      point_type_a=None, point_type_b=None, axis_box=None):
    """Plots 4-way view: Input A -> Output A -> Output B -> Input B."""
    dual_model.eval()
    with torch.no_grad():
        output_a, _ = dual_model.model_a(torch.tensor(data_a, device=device, dtype=torch.float32))
        output_b, _ = dual_model.model_b(torch.tensor(data_b, device=device, dtype=torch.float32))

    output_a = output_a.detach().cpu().numpy()
    output_b = output_b.detach().cpu().numpy()

    if np.isnan(output_a).any() or np.isnan(output_b).any() or np.isinf(output_a).any() or np.isinf(output_b).any():
        fig = plt.figure(figsize=(18, 4))
        fig.suptitle('Self-Supervised Dual Geometry Reshaping (NaN/Inf detected in outputs)')
        return fig

    # For 2D param_values (multi-factor case), extract first factor for coloring
    if isinstance(param_values, np.ndarray) and param_values.ndim == 2:
        param_values_1d = param_values[:, 0]
    else:
        param_values_1d = param_values

    # Normalize color code (for turbo fallback on output spaces)
    color_code = (param_values_1d - np.min(param_values_1d)) / (
        np.max(param_values_1d) - np.min(param_values_1d) + 1e-12)

    # Build noise-aware colors for input spaces; use param color for latent outputs
    pt_a = point_type_a if point_type_a is not None else np.zeros(len(data_a), dtype=np.int32)
    pt_b = point_type_b if point_type_b is not None else np.zeros(len(data_b), dtype=np.int32)
    c_in_a = _get_point_colors(param_values_1d, pt_a)
    c_in_b = _get_point_colors(param_values_1d, pt_b)
    # Latent outputs: retain noise-type coloring (same point indices)
    c_out_a = _get_point_colors(param_values_1d, pt_a)
    c_out_b = _get_point_colors(param_values_1d, pt_b)

    is_3d = data_a.shape[1] >= 3
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle('Self-Supervised Dual Geometry Reshaping')

    if is_3d:
        axs = [fig.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
        axs[0].scatter(data_a[:, 0], data_a[:, 1], data_a[:, 2], c=c_in_a, s=10, alpha=0.85)
        axs[1].scatter(output_a[:, 0], output_a[:, 1], output_a[:, 2], c=c_out_a, s=10, alpha=0.85)
        axs[2].scatter(output_b[:, 0], output_b[:, 1], output_b[:, 2], c=c_out_b, s=10, alpha=0.85)
        axs[3].scatter(data_b[:, 0], data_b[:, 1], data_b[:, 2], c=c_in_b, s=10, alpha=0.85)
        for i in range(4):
            axs[i].set_xlabel('Dim 1'); axs[i].set_ylabel('Dim 2'); axs[i].set_zlabel('Dim 3')
    else:
        axs = [fig.add_subplot(1, 4, i+1) for i in range(4)]
        axs[0].scatter(data_a[:, 0], data_a[:, 1], c=c_in_a, s=10, alpha=0.85)
        axs[1].scatter(output_a[:, 0], output_a[:, 1], c=c_out_a, s=10, alpha=0.85)
        axs[2].scatter(output_b[:, 0], output_b[:, 1], c=c_out_b, s=10, alpha=0.85)
        axs[3].scatter(data_b[:, 0], data_b[:, 1], c=c_in_b, s=10, alpha=0.85)
        for i in range(4):
            axs[i].set_xlabel('Dim 1'); axs[i].set_ylabel('Dim 2'); axs[i].axis('equal')

    axs[0].set_title('Input Space A')
    axs[1].set_title('Output Space A')
    axs[2].set_title('Output Space B')
    axs[3].set_title('Input Space B')

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.85, wspace=0.3)
    return fig


def log_plots_to_wandb(dual_model, dataset, device, step, wandb_run):
    """Generates and logs visualizations to W&B."""
    data_a = dataset.data_a.numpy()
    data_b = dataset.data_b.numpy()
    param_values = dataset.param_values
    pt_a = getattr(dataset, 'point_type_a', None)
    pt_b = getattr(dataset, 'point_type_b', None)

    fig_spaces = plot_original_spaces(data_a, data_b, param_values, pt_a, pt_b)
    fig_reshaping = plot_dual_geometry_reshaping_view(
        dual_model, data_a, data_b, param_values, device, pt_a, pt_b)

    if wandb_run:
        wandb.log({
            "original_spaces": wandb.Image(fig_spaces),
            "geometry_reshaping": wandb.Image(fig_reshaping)
        }, step=step)

    plt.close(fig_spaces)
    plt.close(fig_reshaping)
