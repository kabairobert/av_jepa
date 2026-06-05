import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multimodal_experiments.ssl_dual_alignment.dataset import PointType

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


def to_numpy(val):
    """Convert PyTorch tensor or array-like to numpy array."""
    if val is None:
        return None
    if hasattr(val, 'cpu'):
        return val.cpu().numpy()
    if hasattr(val, 'numpy'):
        return val.numpy()
    return np.asarray(val)


def minmax_normalize(val, eps=1e-8):
    """Normalize array-like to [0, 1] using min-max scaling."""
    v_min = val.min()
    v_max = val.max()
    return (val - v_min) / (v_max - v_min + eps)


def project_to_3d(data):
    if data.shape[1] <= 3:
        return data
    from sklearn.decomposition import PCA
    return PCA(n_components=3).fit_transform(data)


def get_hsv_colors(u1: np.ndarray, u2: np.ndarray, format_type: str = 'rgba') -> list:
    """Computes HSV mapping (u1 -> Hue, u2 -> Saturation) returning either RGBA tuples or Plotly RGB strings."""
    hue = u1 * 360.0
    saturation = 0.1 + u2 * 0.9
    value = np.ones_like(u1)
    
    import colorsys
    colors = []
    for h, s, v in zip(hue, saturation, value):
        r, g, b = colorsys.hsv_to_rgb((h % 360.0) / 360.0, s, v)
        if format_type == 'plotly':
            colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
        else:
            colors.append((r, g, b, 1.0))
    return colors


def get_point_sizes(param_values, default_size=5.0) -> np.ndarray | float:
    """Return point sizes mapped from u3 (if present) or default_size."""
    vals = to_numpy(param_values).astype(float)
    if vals.ndim == 2 and vals.shape[1] >= 3:
        return 2 + minmax_normalize(vals[:, 2]) * 8
    return default_size


def get_point_colors(param_values, point_types=None, format_type='rgba', cmap='rainbow'):
    """Per-point color: manifold=rainbow/hsv, corrupted=gray, external=black/near-black.

    Supports:
        format_type='rgba': returns list of RGBA tuples (matplotlib style)
        format_type='plotly': returns list of 'rgb(R,G,B)' plotly strings
    """
    vals = to_numpy(param_values).astype(float)

    if vals.ndim == 1:
        normalized = minmax_normalize(vals)
        if format_type == 'plotly':
            from plotly.colors import sample_colorscale
            base_colors = sample_colorscale("Rainbow", normalized)
        else:
            colormap = plt.colormaps[cmap]
            base_colors = [colormap(float(p)) for p in normalized]
    else:
        u1_norm = minmax_normalize(vals[:, 0])
        u2_norm = minmax_normalize(vals[:, 1])
        base_colors = get_hsv_colors(u1_norm, u2_norm, format_type=format_type)

    if point_types is None:
        return base_colors

    gray_color = 'rgb(128,128,128)' if format_type == 'plotly' else mcolors.to_rgba(_GRAY_CORRUPT)
    black_color = 'rgb(0,0,0)' if format_type == 'plotly' else mcolors.to_rgba(_BLACK_EXTERNAL)

    colors = []
    for i, pt in enumerate(point_types):
        pt_val = int(pt)
        if pt_val == PointType.EXTERNAL:
            colors.append(black_color)
        elif pt_val in (PointType.ASYM_B_CORRUPT, PointType.ASYM_A_CORRUPT):   # corrupted / noise side of asymmetric pair
            colors.append(gray_color)
        else:                 # manifold (0), asym_a_good (1), asym_b_good (3)
            colors.append(base_colors[i])
    return colors


def plot_original_spaces(data_a, data_b, param_values,
                         point_type_a=None, point_type_b=None):
    """Plots raw Modality A and Modality B datasets with noise-aware coloring."""
    is_3d = data_a.shape[1] >= 3
    fig = plt.figure(figsize=(12, 6))

    data_a_proj = project_to_3d(data_a)
    data_b_proj = project_to_3d(data_b)

    # Build per-point color arrays
    pt_a = point_type_a if point_type_a is not None else np.zeros(len(data_a), dtype=np.int32)
    pt_b = point_type_b if point_type_b is not None else np.zeros(len(data_b), dtype=np.int32)
    c_a = get_point_colors(param_values, pt_a)
    c_b = get_point_colors(param_values, pt_b)
    s_points = get_point_sizes(param_values, default_size=5.0)

    if is_3d:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(data_a_proj[:, 0], data_a_proj[:, 1], data_a_proj[:, 2],
                    c=c_a, s=s_points, alpha=0.5)
        ax1.set_title('Modality A')
        ax1.set_xlabel('PC 1' if data_a.shape[1] > 3 else 'Dim 1')
        ax1.set_ylabel('PC 2' if data_a.shape[1] > 3 else 'Dim 2')
        ax1.set_zlabel('PC 3' if data_a.shape[1] > 3 else 'Dim 3')
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(data_b_proj[:, 0], data_b_proj[:, 1], data_b_proj[:, 2],
                    c=c_b, s=s_points, alpha=0.5)
        ax2.set_title('Modality B')
        ax2.set_xlabel('PC 1' if data_b.shape[1] > 3 else 'Dim 1')
        ax2.set_ylabel('PC 2' if data_b.shape[1] > 3 else 'Dim 2')
        ax2.set_zlabel('PC 3' if data_b.shape[1] > 3 else 'Dim 3')
    else:
        ax1 = fig.add_subplot(121)
        ax1.scatter(data_a[:, 0], data_a[:, 1], c=c_a, s=s_points, alpha=0.5)
        ax1.set_title('Modality A')
        ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2')
        ax1.axis('equal')
        ax2 = fig.add_subplot(122)
        ax2.scatter(data_b[:, 0], data_b[:, 1], c=c_b, s=s_points, alpha=0.5)
        ax2.set_title('Modality B')
        ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2')
        ax2.axis('equal')

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
    return fig


def plot_dual_geometry_reshaping_view(dual_model, data_a, data_b, param_values, device,
                                      point_type_a=None, point_type_b=None):
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

    data_a_proj = project_to_3d(data_a)
    output_a_proj = project_to_3d(output_a)
    output_b_proj = project_to_3d(output_b)
    data_b_proj = project_to_3d(data_b)

    # Build noise-aware colors for input spaces; use param color for latent outputs
    pt_a = point_type_a if point_type_a is not None else np.zeros(len(data_a), dtype=np.int32)
    pt_b = point_type_b if point_type_b is not None else np.zeros(len(data_b), dtype=np.int32)
    c_in_a = get_point_colors(param_values, pt_a)
    c_in_b = get_point_colors(param_values, pt_b)
    # Latent outputs: retain noise-type coloring (same point indices)
    c_out_a = get_point_colors(param_values, pt_a)
    c_out_b = get_point_colors(param_values, pt_b)
    
    s_points = get_point_sizes(param_values, default_size=5.0) * 2  # double size for the 4-way plot default to match the original s=10 vs s=5

    is_3d = data_a.shape[1] >= 3
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle('Self-Supervised Dual Geometry Reshaping')

    if is_3d:
        axs = [fig.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
        axs[0].scatter(data_a_proj[:, 0], data_a_proj[:, 1], data_a_proj[:, 2], c=c_in_a, s=s_points, alpha=0.85)
        axs[1].scatter(output_a_proj[:, 0], output_a_proj[:, 1], output_a_proj[:, 2], c=c_out_a, s=s_points, alpha=0.85)
        axs[2].scatter(output_b_proj[:, 0], output_b_proj[:, 1], output_b_proj[:, 2], c=c_out_b, s=s_points, alpha=0.85)
        axs[3].scatter(data_b_proj[:, 0], data_b_proj[:, 1], data_b_proj[:, 2], c=c_in_b, s=s_points, alpha=0.85)
        for i in range(4):
            dim_str = "PC" if data_a.shape[1] > 3 else "Dim"
            axs[i].set_xlabel(f'{dim_str} 1')
            axs[i].set_ylabel(f'{dim_str} 2')
            axs[i].set_zlabel(f'{dim_str} 3')
    else:
        axs = [fig.add_subplot(1, 4, i+1) for i in range(4)]
        axs[0].scatter(data_a[:, 0], data_a[:, 1], c=c_in_a, s=s_points, alpha=0.85)
        axs[1].scatter(output_a[:, 0], output_a[:, 1], c=c_out_a, s=s_points, alpha=0.85)
        axs[2].scatter(output_b[:, 0], output_b[:, 1], c=c_out_b, s=s_points, alpha=0.85)
        axs[3].scatter(data_b[:, 0], data_b[:, 1], c=c_in_b, s=s_points, alpha=0.85)
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
    import wandb
    # Ensure all components are on CPU before converting to numpy for plotting
    data_a = to_numpy(dataset.data_a)
    data_b = to_numpy(dataset.data_b)
    param_values = to_numpy(dataset.param_values)
    pt_a = to_numpy(getattr(dataset, 'point_type_a', None))
    pt_b = to_numpy(getattr(dataset, 'point_type_b', None))

    # Subsample to max 5000 points to prevent OOM and slow Matplotlib rendering
    N = data_a.shape[0]
    if N > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, size=5000, replace=False)
        data_a = data_a[idx]
        data_b = data_b[idx]
        if param_values is not None:
            param_values = param_values[idx]
        if pt_a is not None:
            pt_a = pt_a[idx]
        if pt_b is not None:
            pt_b = pt_b[idx]

    fig_spaces = plot_original_spaces(data_a, data_b, param_values, pt_a, pt_b)
    fig_reshaping = None
    try:
        fig_reshaping = plot_dual_geometry_reshaping_view(
            dual_model, data_a, data_b, param_values, device, pt_a, pt_b)

        if wandb_run:
            wandb.log({
                "original_spaces": wandb.Image(fig_spaces),
                "geometry_reshaping": wandb.Image(fig_reshaping)
            }, step=step)
    finally:
        if fig_spaces is not None:
            plt.close(fig_spaces)
        if fig_reshaping is not None:
            plt.close(fig_reshaping)
