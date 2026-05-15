import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_device, setup_seed, setup_wandb, load_checkpoint
from multimodal_experiments.ssl_dual_alignment.dataset import DualDisentangleDataset
from multimodal_experiments.ssl_dual_alignment.model_builder import build_model_and_predictors
from multimodal_experiments.ssl_dual_alignment.losses import EBMJEPALoss
from multimodal_experiments.initial_trials.ssl_disentangling import SupervisedFactorLoss
from multimodal_experiments.ssl_dual_alignment.vis import log_plots_to_wandb

logger = get_logger(__name__)


def _to_bool_or_none(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    sval = str(val).strip().lower()
    if sval in ("1", "true", "yes", "y", "on"):
        return True
    if sval in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Cannot parse boolean value: {val}")


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pth\.tar$", path.name)
    if match:
        return int(match.group(1))
    return -1


def _discover_checkpoints(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("epoch_*.pth.tar"), key=_checkpoint_epoch)
    latest = run_dir / "latest.pth.tar"
    if latest.exists():
        if not ckpts:
            ckpts.append(latest)
        else:
            newest_epoch = max(ckpts, key=lambda p: p.stat().st_mtime)
            if latest.stat().st_mtime > newest_epoch.stat().st_mtime:
                ckpts.append(latest)
    return ckpts


def _get_color_values(param_values: np.ndarray) -> np.ndarray:
    """Convert param values to RGB colors.

    For 1D param_values: use Turbo colorscale.
    For 2D param_values: use HSV encoding (u1 -> Hue, u2 -> Saturation [0.2, 1]).
    """
    if hasattr(param_values, 'numpy'):
        param_values = param_values.numpy()

    if param_values.ndim == 1:
        vals = param_values
        denom = (vals.max() - vals.min()) + 1e-8
        normalized = (vals - vals.min()) / denom
        from plotly.colors import sample_colorscale
        return sample_colorscale("Turbo", normalized)
    else:
        u1 = param_values[:, 0]
        u2 = param_values[:, 1]
        hue = u1 * 360.0
        saturation = 0.2 + u2 * 0.8
        value = np.ones_like(u1)
        import colorsys
        color_list = []
        for h, s, v in zip(hue, saturation, value):
            r, g, b = colorsys.hsv_to_rgb((h % 360.0) / 360.0, s, v)
            color_list.append('rgb({},{},{})'.format(int(r * 255), int(g * 255), int(b * 255)))
        return color_list


def _get_point_type_colors(param_values: np.ndarray, point_types: np.ndarray) -> list:
    """Return Plotly color strings using point_type coloring.

    Manifold points use Turbo by param value. Corrupted points are gray.
    External points are near-black.
    """
    if param_values.ndim == 2:
        param_values = param_values[:, 0]
    vals = param_values.astype(float)
    denom = (vals.max() - vals.min()) + 1e-8
    normalized = (vals - vals.min()) / denom
    from plotly.colors import sample_colorscale
    base_colors = sample_colorscale("Turbo", normalized)

    colors = []
    for i, pt in enumerate(point_types):
        if int(pt) == 5:
            colors.append("rgb(26,26,26)")
        elif int(pt) in (2, 4):
            colors.append("rgb(128,128,128)")
        else:
            colors.append(base_colors[i])
    return colors


def _build_interactive_4way_html(
    data_a: np.ndarray,
    data_b: np.ndarray,
    out_a: np.ndarray,
    out_b: np.ndarray,
    param_values: np.ndarray,
    point_type_a: Optional[np.ndarray] = None,
    point_type_b: Optional[np.ndarray] = None,
    min_height_px: int = 420,
    axis_box: Optional[np.ndarray] = None,
) -> Optional[str]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:
        logger.warning("Plotly not available; skipping interactive 3D plot: %s", exc)
        return None

    if point_type_a is not None:
        color_vals_a = _get_point_type_colors(param_values, point_type_a)
    else:
        color_vals_a = _get_color_values(param_values)
    if point_type_b is not None:
        color_vals_b = _get_point_type_colors(param_values, point_type_b)
    else:
        color_vals_b = _get_color_values(param_values)

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "scatter3d"}] * 4],
        subplot_titles=("Input Space A", "Output Space A", "Output Space B", "Input Space B"),
    )

    def _scatter(xyz, name, colors):
        return go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors, showscale=False),
            name=name,
        )

    fig.add_trace(_scatter(data_a, "Input A", color_vals_a), row=1, col=1)
    fig.add_trace(_scatter(out_a, "Output A", color_vals_a), row=1, col=2)
    fig.add_trace(_scatter(out_b, "Output B", color_vals_b), row=1, col=3)
    fig.add_trace(_scatter(data_b, "Input B", color_vals_b), row=1, col=4)

    scene_cube = dict(aspectmode="cube")
    fig.update_layout(
        autosize=True, height=min_height_px,
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=False, hovermode="closest",
        scene=scene_cube, scene2=scene_cube, scene3=scene_cube, scene4=scene_cube,
    )

    html_body = fig.to_html(full_html=False, include_plotlyjs="cdn", default_width="100%", default_height="100%")
    return f"<div style='width:100%;height:100%;min-height:{int(min_height_px)}px'>{html_body}</div>"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def linear_probe_r2(z: np.ndarray, u: np.ndarray) -> dict:
    """Fit Ridge regression z -> u, return R2 per factor and mean."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    reg = Ridge(alpha=1.0).fit(z, u)
    u_pred = reg.predict(z)
    if u.ndim == 1:
        return {'r2_u0': float(r2_score(u, u_pred)), 'r2_mean': float(r2_score(u, u_pred))}
    r2_per = [float(r2_score(u[:, i], u_pred[:, i])) for i in range(u.shape[1])]
    result = {f'r2_u{i}': v for i, v in enumerate(r2_per)}
    result['r2_mean'] = float(np.mean(r2_per))
    return result


def per_dim_disentanglement(z_a: np.ndarray, u: np.ndarray) -> dict:
    """Per-dim linear R2: regress each dim of z_A independently onto each factor.

    Returns:
      r2_dim{i}_u{j}: R2 of z_A[:, i] -> u[:, j]
      r2_dim2_noise:  max R2 of z_A[:, 2] onto any factor (should be ~0 if noise dim is clean)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    if u.ndim == 1:
        u = u[:, None]

    n_dims = z_a.shape[1]
    n_factors = u.shape[1]
    results = {}

    for i in range(n_dims):
        zi = z_a[:, i:i+1]
        for j in range(n_factors):
            uj = u[:, j]
            reg = Ridge(alpha=1.0).fit(zi, uj)
            r2 = float(r2_score(uj, reg.predict(zi)))
            results[f'r2_dim{i}_u{j}'] = r2

    # Canonical aliases used in hypotheses.yaml
    # dim0 -> u1 (factor 0), dim1 -> u2 (factor 1), dim2 -> max(any factor) = noise purity
    results['r2_dim0_u1'] = results.get('r2_dim0_u0', 0.0)   # u1 = factor index 0
    results['r2_dim1_u2'] = results.get('r2_dim1_u1', 0.0)   # u2 = factor index 1
    if n_dims > 2:
        noise_r2s = [results.get(f'r2_dim2_u{j}', 0.0) for j in range(n_factors)]
        results['r2_dim2_noise'] = float(max(noise_r2s))      # high = noise dim is NOT clean
    return results


def pca_axis_alignment(z: np.ndarray, n_active: int = 2) -> float:
    """Measure how axis-aligned the top-n_active PCA components of z are.

    For each of the top-n_active eigenvectors, compute max |cosine| with any coord axis.
    Return the mean of these max cosines across the active components.

    Interpretation:
      1.0  = each active PC perfectly parallel to a coord axis (ideal)
      ~0.57 = random orientation in 3D (1/sqrt(3))
    """
    z_centered = z - z.mean(axis=0)
    _, _, Vt = np.linalg.svd(z_centered, full_matrices=False)
    n_dims = z.shape[1]
    identity = np.eye(n_dims)
    scores = []
    for i in range(min(n_active, Vt.shape[0])):
        cosines = np.abs(Vt[i] @ identity)   # |cos| with each coord axis
        scores.append(float(cosines.max()))
    return float(np.mean(scores))


def manifold_flatness(z: np.ndarray, n_plane: int = 2) -> dict:
    """PCA-based flatness of a latent manifold.

    flatness_ratio: variance explained by top-n_plane PCs (1.0 = perfectly flat).
    orth_residual_mean: mean distance to the top-n_plane PCA subspace (0 = flat).
    """
    z_centered = z - z.mean(axis=0)
    _, svals, vt = np.linalg.svd(z_centered, full_matrices=False)
    total_var = float(np.sum(svals**2))
    top_var = float(np.sum(svals[:n_plane]**2))
    flatness_ratio = top_var / total_var if total_var > 1e-12 else 0.0

    basis = vt[:n_plane].T
    proj = z_centered @ basis
    recon = proj @ basis.T
    residual = z_centered - recon
    orth_residual_mean = float(np.linalg.norm(residual, axis=1).mean())

    return {
        'flatness_ratio': float(flatness_ratio),
        'orth_residual_mean': orth_residual_mean,
    }


def retrieval_accuracy(z_a: np.ndarray, z_b: np.ndarray, ks=(1, 5)) -> dict:
    """For each z_A[i], find k-nearest z_B by L2 and cosine. Check if correct index in top-k."""
    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
    results = {}
    for dist_fn, name in [(euclidean_distances, 'l2'), (cosine_distances, 'cos')]:
        D = dist_fn(z_a, z_b)
        for k in ks:
            top_k = np.argsort(D, axis=1)[:, :k]
            hits = float(np.mean([i in top_k[i] for i in range(len(z_a))]))
            results[f'retrieval_{name}@{k}'] = hits
    return results


def cca_score(z_a: np.ndarray, z_b: np.ndarray) -> dict:
    """CCA between z_A and z_B.

    Returns:
      cca_corr_dim{i}: canonical correlation for each component
      cca_effective_rank: number of components with corr > 0.5
      cca_diag_score: diagonality of cross-correlation matrix in CCA space
        = sum(|diag(C)|) / sum(|C|) where C[i,j] = corr(z_a_c[:,i], z_b_c[:,j])
        1.0 = dim-i of z_A maps exactly to dim-i of z_B
        0.0 = fully off-diagonal (rotated alignment)
    """
    from sklearn.cross_decomposition import CCA
    n_components = min(z_a.shape[1], z_b.shape[1])
    try:
        cca = CCA(n_components=n_components).fit(z_a, z_b)
        z_a_c, z_b_c = cca.transform(z_a, z_b)
        corrs = [float(np.corrcoef(z_a_c[:, i], z_b_c[:, i])[0, 1]) for i in range(n_components)]

        # Full cross-correlation matrix for diagonality score
        C = np.zeros((n_components, n_components))
        for i in range(n_components):
            for j in range(n_components):
                C[i, j] = abs(float(np.corrcoef(z_a_c[:, i], z_b_c[:, j])[0, 1]))
        total = C.sum()
        diag_score = float(np.diag(C).sum() / total) if total > 1e-8 else 0.0

    except Exception as exc:
        logger.warning("CCA failed: %s", exc)
        corrs = [0.0] * n_components
        diag_score = 0.0

    result = {f'cca_corr_dim{i}': c for i, c in enumerate(corrs)}
    result['cca_effective_rank'] = float(np.sum(np.array(corrs) > 0.5))
    result['cca_diag_score'] = diag_score
    return result


def compute_geometry_metrics(
    dual_model: torch.nn.Module,
    dataset: DualDisentangleDataset,
    device: torch.device,
    max_points: int = 4096,
) -> dict:
    """Collect z_A, z_B from model; compute R2, per-dim disentanglement,
    PCA axis-alignment, retrieval, CCA, and norm diagnostics.

    Returns flat dict suitable for wandb.log.
    """
    dual_model.eval()
    data_a = dataset.data_a
    data_b = dataset.data_b
    param_values = np.asarray(dataset.param_values)
    idxs = None

    if data_a.shape[0] > max_points:
        idxs = np.random.choice(data_a.shape[0], size=max_points, replace=False)
        data_a = data_a[idxs]
        data_b = data_b[idxs]
        param_values = param_values[idxs]

    with torch.no_grad():
        out_a, _ = dual_model.model_a(data_a.to(device).float())
        out_b, _ = dual_model.model_b(data_b.to(device).float())
    z_a = out_a.cpu().numpy()
    z_b = out_b.cpu().numpy()

    if np.isnan(z_a).any() or np.isnan(z_b).any() or np.isinf(z_a).any() or np.isinf(z_b).any():
        logger.warning("NaN/Inf detected in latents! Skipping geometry metric computations.")
        return {}

    metrics = {}
    u = param_values  # shape (N,) or (N, n_factors)

    # --- Joint + per-modality linear probe R2 ---
    for prefix, z in [('za', z_a), ('zb', z_b), ('zjoint', np.concatenate([z_a, z_b], axis=1))]:
        probe = linear_probe_r2(z, u)
        for k, v in probe.items():
            metrics[f'geom/{prefix}/{k}'] = v

    # --- Per-dim disentanglement (z_A only — primary geometry space) ---
    if param_values.ndim == 2 and z_a.shape[1] >= 3:
        pdis = per_dim_disentanglement(z_a, u)
        for k, v in pdis.items():
            metrics[f'geom/za/{k}'] = v

    # --- PCA axis-alignment ---
    n_active = 2 if param_values.ndim == 2 else 1
    metrics['geom/pca_axis_align_a'] = pca_axis_alignment(z_a, n_active=n_active)
    metrics['geom/pca_axis_align_b'] = pca_axis_alignment(z_b, n_active=n_active)

    # --- Manifold flatness ---
    if param_values.ndim == 2 and z_a.shape[1] >= 2:
        flat_a = manifold_flatness(z_a, n_plane=2)
        flat_b = manifold_flatness(z_b, n_plane=2)
        metrics['geom/za/flatness_ratio'] = flat_a['flatness_ratio']
        metrics['geom/za/orth_residual_mean'] = flat_a['orth_residual_mean']
        metrics['geom/zb/flatness_ratio'] = flat_b['flatness_ratio']
        metrics['geom/zb/orth_residual_mean'] = flat_b['orth_residual_mean']

    # --- Retrieval ---
    ret = retrieval_accuracy(z_a, z_b)
    for k, v in ret.items():
        metrics[f'geom/{k}'] = v

    # --- CCA (includes cca_diag_score) ---
    cca = cca_score(z_a, z_b)
    for k, v in cca.items():
        metrics[f'geom/{k}'] = v

    # --- Norm diagnostics ---
    norms_a = np.linalg.norm(z_a, axis=1)
    norms_b = np.linalg.norm(z_b, axis=1)
    metrics['geom/z_a_norm_mean'] = float(norms_a.mean())
    metrics['geom/z_b_norm_mean'] = float(norms_b.mean())

    # --- Norms by point type ---
    pt_a = getattr(dataset, "point_type_a", None)
    pt_b = getattr(dataset, "point_type_b", None)
    if pt_a is not None and pt_b is not None:
        pt_a = np.asarray(pt_a)
        pt_b = np.asarray(pt_b)
        if idxs is not None:
            pt_a = pt_a[idxs]
            pt_b = pt_b[idxs]

        def _mean_or_nan(values, mask):
            return float(values[mask].mean()) if mask.any() else float("nan")

        metrics['geom/z_a_norm_manifold'] = _mean_or_nan(norms_a, pt_a == 0)
        metrics['geom/z_a_norm_asym_corrupt'] = _mean_or_nan(norms_a, pt_a == 4)
        metrics['geom/z_a_norm_external'] = _mean_or_nan(norms_a, pt_a == 5)

        metrics['geom/z_b_norm_manifold'] = _mean_or_nan(norms_b, pt_b == 0)
        metrics['geom/z_b_norm_asym_corrupt'] = _mean_or_nan(norms_b, pt_b == 2)
        metrics['geom/z_b_norm_external'] = _mean_or_nan(norms_b, pt_b == 5)

    return metrics


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def _eval_loop(
    loader: DataLoader,
    dual_model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loss_type: str,
    predictors: Dict[str, Optional[torch.nn.Module]],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    dual_model.eval()
    metrics = {
        "loss": 0.0,
        "align_mse_a2b": 0.0,
        "align_mse_b2a": 0.0,
        "align_mse_a2b_manifold": 0.0,
        "align_mse_b2a_manifold": 0.0,
        "align_mse_a2b_asym_corrupt": 0.0,
        "align_mse_b2a_asym_corrupt": 0.0,
        "align_mse_a2b_external": 0.0,
        "count_manifold": 0,
        "count_asym": 0,
        "count_external": 0,
        "num_batches": 0,
    }

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
            data_a = batch["data_a"].to(device)
            data_b = batch["data_b"].to(device)
            corr_target = batch["corr_target"].to(device)

            outputs = dual_model(data_a, data_b)
            if loss_type == "ebm":
                loss = loss_fn(outputs)
            else:
                loss = loss_fn(corr_target, outputs)

            d = (outputs.shape[1] - 2) // 2
            z_a, z_b = outputs[:, :d], outputs[:, d:2 * d]
            if predictors["a2b"] is not None and predictors["b2a"] is not None:
                pred_a2b = predictors["a2b"](z_a)
                pred_b2a = predictors["b2a"](z_b)
                err_a2b = F.mse_loss(pred_a2b, z_b).item()
                err_b2a = F.mse_loss(pred_b2a, z_a).item()
                metrics["align_mse_a2b"] += err_a2b
                metrics["align_mse_b2a"] += err_b2a

                pt_a = batch.get("point_type_a", None)
                pt_b = batch.get("point_type_b", None)
                if pt_a is not None and pt_b is not None:
                    pt_a = pt_a.to(device)
                    pt_b = pt_b.to(device)
                    external_mask = (pt_a == 5) | (pt_b == 5)
                    manifold_mask = (pt_a == 0) & (pt_b == 0)
                    asym_mask = (~external_mask) & (~manifold_mask)

                    if manifold_mask.any():
                        metrics["align_mse_a2b_manifold"] += F.mse_loss(
                            pred_a2b[manifold_mask], z_b[manifold_mask], reduction="sum"
                        ).item()
                        metrics["align_mse_b2a_manifold"] += F.mse_loss(
                            pred_b2a[manifold_mask], z_a[manifold_mask], reduction="sum"
                        ).item()
                        metrics["count_manifold"] += int(manifold_mask.sum().item())
                    if asym_mask.any():
                        metrics["align_mse_a2b_asym_corrupt"] += F.mse_loss(
                            pred_a2b[asym_mask], z_b[asym_mask], reduction="sum"
                        ).item()
                        metrics["align_mse_b2a_asym_corrupt"] += F.mse_loss(
                            pred_b2a[asym_mask], z_a[asym_mask], reduction="sum"
                        ).item()
                        metrics["count_asym"] += int(asym_mask.sum().item())
                    if external_mask.any():
                        metrics["align_mse_a2b_external"] += F.mse_loss(
                            pred_a2b[external_mask], z_b[external_mask], reduction="sum"
                        ).item()
                        metrics["count_external"] += int(external_mask.sum().item())

            metrics["loss"] += float(loss.item())
            metrics["num_batches"] += 1

            if max_batches is not None and (bi + 1) >= int(max_batches):
                break

    denom = max(metrics["num_batches"], 1)
    out = {
        "loss": metrics["loss"] / denom,
        "align_mse_a2b": metrics["align_mse_a2b"] / denom,
        "align_mse_b2a": metrics["align_mse_b2a"] / denom,
    }
    if metrics["count_manifold"] > 0:
        out["align_mse_a2b_manifold"] = metrics["align_mse_a2b_manifold"] / metrics["count_manifold"]
        out["align_mse_b2a_manifold"] = metrics["align_mse_b2a_manifold"] / metrics["count_manifold"]
    if metrics["count_asym"] > 0:
        out["align_mse_a2b_asym_corrupt"] = metrics["align_mse_a2b_asym_corrupt"] / metrics["count_asym"]
        out["align_mse_b2a_asym_corrupt"] = metrics["align_mse_b2a_asym_corrupt"] / metrics["count_asym"]
    if metrics["count_external"] > 0:
        out["align_mse_a2b_external"] = metrics["align_mse_a2b_external"] / metrics["count_external"]
    return out


def evaluate_and_log_checkpoint(
    eval_set: DualDisentangleDataset,
    eval_loader: DataLoader,
    dual_model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loss_type: str,
    predictors: Dict[str, Optional[torch.nn.Module]],
    device: torch.device,
    step: int,
    wandb_run,
    *,
    checkpoint_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    max_batches: Optional[int] = None,
    log_interactive_3d: bool = True,
    interactive_min_height: int = 420,
    max_interactive_points: int = 2000,
    log_prefix: str = "val",
    is_3d: Optional[bool] = None,
) -> Dict[str, float]:
    metrics = _eval_loop(eval_loader, dual_model, loss_fn, loss_type, predictors, device, max_batches=max_batches)

    logs = {
        f"{log_prefix}/loss": metrics["loss"],
        f"{log_prefix}/align_mse_a2b": metrics["align_mse_a2b"],
        f"{log_prefix}/align_mse_b2a": metrics["align_mse_b2a"],
    }
    if "align_mse_a2b_manifold" in metrics:
        logs[f"{log_prefix}/align_mse_a2b_manifold"] = metrics["align_mse_a2b_manifold"]
        logs[f"{log_prefix}/align_mse_b2a_manifold"] = metrics["align_mse_b2a_manifold"]
    if "align_mse_a2b_asym_corrupt" in metrics:
        logs[f"{log_prefix}/align_mse_a2b_asym_corrupt"] = metrics["align_mse_a2b_asym_corrupt"]
        logs[f"{log_prefix}/align_mse_b2a_asym_corrupt"] = metrics["align_mse_b2a_asym_corrupt"]
    if "align_mse_a2b_external" in metrics:
        logs[f"{log_prefix}/align_mse_a2b_external"] = metrics["align_mse_a2b_external"]
    if checkpoint_name is not None:
        logs["eval/checkpoint_name"] = checkpoint_name
    if checkpoint_path is not None:
        logs["eval/checkpoint_path"] = checkpoint_path
    logs["eval/checkpoint_step"] = int(step)

    if wandb_run:
        import wandb
        wandb.log(logs, step=step)
        log_plots_to_wandb(dual_model, eval_set, device, step, wandb_run)

        # Geometry metrics: R2, per-dim disentanglement, PCA axis-align, retrieval, CCA, norms
        geom_metrics = compute_geometry_metrics(dual_model, eval_set, device)
        wandb.log(geom_metrics, step=step)

        if is_3d is None:
            is_3d = str(getattr(eval_set, "data_type", "")).startswith("3d")

        if is_3d and log_interactive_3d:
            data_a = eval_set.data_a.numpy()
            data_b = eval_set.data_b.numpy()
            param_values = eval_set.param_values
            if data_a.shape[0] > max_interactive_points:
                idxs = np.random.choice(data_a.shape[0], size=max_interactive_points, replace=False)
                data_a = data_a[idxs]
                data_b = data_b[idxs]
                param_values = param_values[idxs]

            dual_model.eval()
            with torch.no_grad():
                out_a, _ = dual_model.model_a(torch.tensor(data_a, device=device, dtype=torch.float32))
                out_b, _ = dual_model.model_b(torch.tensor(data_b, device=device, dtype=torch.float32))
            out_a = out_a.detach().cpu().numpy()
            out_b = out_b.detach().cpu().numpy()

        if np.isnan(out_a).any() or np.isnan(out_b).any() or np.isinf(out_a).any() or np.isinf(out_b).any():
            logger.warning("NaN/Inf detected in latents! Skipping interactive 3D plot.")
            return metrics

            pt_a = getattr(eval_set, "point_type_a", None)
            pt_b = getattr(eval_set, "point_type_b", None)
            if pt_a is not None and data_a.shape[0] == eval_set.data_a.shape[0]:
                pt_a = np.asarray(pt_a)
                pt_b = np.asarray(pt_b)
            if pt_a is not None and data_a.shape[0] < eval_set.data_a.shape[0]:
                pt_a = np.asarray(pt_a)[idxs]
                pt_b = np.asarray(pt_b)[idxs]

            html = _build_interactive_4way_html(
                data_a, data_b, out_a, out_b, np.asarray(param_values),
                point_type_a=pt_a,
                point_type_b=pt_b,
                min_height_px=int(interactive_min_height),
            )
            if html is not None:
                wandb.log({"interactive_3d_4way_html": wandb.Html(html)}, step=step)

    return metrics


def run(
    folder: Optional[str] = None,
    cfg: Optional[str] = None,
    checkpoint: Optional[str] = None,
    log_wandb: Optional[bool] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_batches: Optional[int] = None,
    log_interactive_3d: bool = True,
    interactive_min_height: int = 420,
    max_interactive_points: int = 2000,
    **overrides,
):
    if folder is None and checkpoint is None and cfg is None:
        raise ValueError("Provide at least one of: folder, checkpoint, or cfg")

    folder_path = Path(folder) if folder is not None else None
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None

    cfg_path = None
    if cfg is not None:
        cfg_path = Path(cfg)
    elif folder_path is not None:
        candidate = folder_path / "config.yaml"
        if candidate.exists():
            cfg_path = candidate
    elif checkpoint_path is not None:
        candidate = checkpoint_path.parent / "config.yaml"
        if candidate.exists():
            cfg_path = candidate

    if cfg_path is None:
        raise ValueError("Could not resolve config path. Pass --cfg or provide --folder with config.yaml")

    cfg_obj = load_config(str(cfg_path))
    if overrides:
        override_dict: dict = {}
        for key, value in overrides.items():
            keys = key.split(".")
            current = override_dict
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
        from omegaconf import OmegaConf
        cfg_obj = OmegaConf.merge(cfg_obj, OmegaConf.create(override_dict))

    device = setup_device(cfg_obj.meta.device)
    setup_seed(cfg_obj.meta.seed)

    data_cfg = cfg_obj.data
    eval_set = DualDisentangleDataset(
        data_type=data_cfg.get("type", "2d"),
        num_samples=data_cfg.get("num_samples", 4096),
        path_a=data_cfg.get("path_a", None),
        path_b=data_cfg.get("path_b", None),
        manifold_noise_a=data_cfg.get("manifold_noise_a", None),
        manifold_noise_b=data_cfg.get("manifold_noise_b", None),
        asymmetric_noise_magnitude=data_cfg.get("asymmetric_noise_magnitude", None),
        asymmetric_noise_rate_a=data_cfg.get("asymmetric_noise_rate_a", None),
        asymmetric_noise_rate_b=data_cfg.get("asymmetric_noise_rate_b", None),
        external_noise_ratio=data_cfg.get("external_noise_ratio", None),
        noise_bbox_expansion=data_cfg.get("noise_bbox_expansion", 0.0),
        seed=cfg_obj.meta.seed,
    )
    effective_batch_size = int(batch_size) if batch_size is not None else int(data_cfg.get("batch_size", 128))
    eval_loader = DataLoader(eval_set, batch_size=effective_batch_size, shuffle=False,
                             num_workers=int(num_workers or data_cfg.get("num_workers", 0)))

    built = build_model_and_predictors(cfg_obj, device)
    dual_model = built["dual_model"]
    predictor_a2b = built["predictor_a2b"]
    predictor_b2a = built["predictor_b2a"]

    cm_val = str(cfg_obj.loss.get("congruence_mode", cfg_obj.loss.get("noise_reweighting", "none")))
    if cm_val in ("none", "off", "cm_off"):
        canon_cm = "none"
    elif cm_val in ("pred_only", "pred", "cm_pred"):
        canon_cm = "pred_only"
    elif cm_val in ("pred_and_sparse", "pred_sparse", "full", "cm_pred_sparse"):
        canon_cm = "pred_and_sparse"
    else:
        canon_cm = cm_val

    loss_type = cfg_obj.loss.get("type", "ebm")
    if loss_type == "ebm":
        loss_fn = EBMJEPALoss(
            predictor_a2b, predictor_b2a,
            lambda_jac=cfg_obj.loss.get("lambda_jac", 1.0),
            lambda_prior=cfg_obj.loss.get("lambda_prior", 0.5),
            lambda_pred=cfg_obj.loss.get("lambda_pred", 1.0),
            lambda_sparse=cfg_obj.loss.get("lambda_sparse", 0.1),
            prior_type=cfg_obj.loss.get("prior_type", 'l1'),
            pred_loss=cfg_obj.loss.get("pred_loss", 'l1'),
            congruence_mode=canon_cm,
            congruence_tau=cfg_obj.loss.get("congruence_tau", cfg_obj.loss.get("reweighting_tau", 0.5)),
        )
    else:
        loss_fn = SupervisedFactorLoss(
            dimensions_per_factor=[1, 1] if data_cfg.get("type", "2d") == "2d" else [1, 1, 1]
        )

    log_wandb_override = _to_bool_or_none(log_wandb)
    enabled_wandb = bool(cfg_obj.logging.get("log_wandb", False)) if log_wandb_override is None else bool(log_wandb_override)
    run_dir = folder_path if folder_path is not None else (checkpoint_path.parent if checkpoint_path is not None else cfg_path.parent)

    base_tags = ["sslda", "eval"]
    if cfg_obj.logging.get("log_seed_tag", False):
        base_tags.append(f"seed_{cfg_obj.meta.seed}")

    wandb_run = setup_wandb(
        project="eb_jepa", config=cfg_obj, run_dir=run_dir / "eval_wandb",
        run_name=f"{run_dir.name}_eval",
        tags=base_tags,
        group=cfg_obj.logging.get("wandb_group"),
        enabled=enabled_wandb, resume=False,
    )

    ckpts = [checkpoint_path] if checkpoint_path is not None else _discover_checkpoints(run_dir)
    if not ckpts:
        raise ValueError(f"No checkpoints found in {run_dir}")

    is_3d = str(data_cfg.get("type", "2d")).startswith("3d")

    for idx, ckpt in enumerate(ckpts):
        is_last = (idx == (len(ckpts) - 1))
        ckpt_meta = load_checkpoint(ckpt, dual_model, optimizer=None, device=device)
        ckpt_step = int(ckpt_meta.get("step", None) or _checkpoint_epoch(ckpt))

        metrics = evaluate_and_log_checkpoint(
            eval_set, eval_loader, dual_model, loss_fn, loss_type,
            {"a2b": predictor_a2b, "b2a": predictor_b2a},
            device, ckpt_step, wandb_run,
            checkpoint_name=ckpt.name, checkpoint_path=str(ckpt),
            max_batches=max_batches,
            log_interactive_3d=is_3d and is_last and log_interactive_3d,
            interactive_min_height=interactive_min_height,
            max_interactive_points=max_interactive_points,
            log_prefix="val", is_3d=is_3d,
        )
        logger.info("Eval %s | loss=%.4f | a2b=%.4f | b2a=%.4f",
                    ckpt.name, metrics["loss"], metrics["align_mse_a2b"], metrics["align_mse_b2a"])

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    import fire
    fire.Fire(run)
