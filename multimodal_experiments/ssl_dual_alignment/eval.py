import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_device, setup_seed, setup_wandb, load_checkpoint
from multimodal_experiments.ssl_dual_alignment.dataset import PointType, build_dataset_from_config, DualDisentangleDataset
from multimodal_experiments.ssl_dual_alignment.model_builder import build_model_and_predictors
from multimodal_experiments.ssl_dual_alignment.losses import build_loss_from_config
from multimodal_experiments.ssl_dual_alignment.vis import log_plots_to_wandb, project_to_3d, to_numpy, get_point_sizes, get_point_colors, build_interactive_4way_html
from multimodal_experiments.ssl_dual_alignment.metrics import (
    linear_probe_r2,
    rankme_score,
    vicreg_variance,
    vicreg_covariance,
    pca_axis_alignment,
    masked_retrieval_accuracy,
    retrieval_accuracy,
    cca_score,
    compute_diagonality_ratio,
    compute_found_rank_metrics,
    compute_norm_diagnostics,
)

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








# Helper functions migrated to metrics.py and vis.py


def compute_geometry_metrics(
    dual_model: torch.nn.Module,
    dataset: DualDisentangleDataset,
    device: torch.device,
    max_points: int = 4096,
    predictor_a2b=None,
) -> dict:
    """Collect z_A, z_B from model; compute R2, per-dim disentanglement,
    PCA axis-alignment, retrieval, CCA, norm diagnostics, and found-common-rank suite.

    predictor_a2b: optional AffinePredictor/DiagonalPredictor — used to compute
        per-dim predictor R2 and weight spectra for the found-rank suite.

    Returns flat dict suitable for wandb.log.
    """
    dual_model.eval()
    data_a = dataset.data_a
    data_b = dataset.data_b
    param_values = to_numpy(dataset.param_values)
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
    # k_shared: number of shared latent factors, inferred from param_values which stores u_s
    k_shared = u.shape[1] if u.ndim == 2 else 1
    num_zdims = z_a.shape[1]

    # --- Pre-compute clean mask once (reused by flatness, predictor blocks) ---
    z_a_clean: np.ndarray | None = None
    z_b_clean: np.ndarray | None = None
    _pt_a_attr = getattr(dataset, "point_type_a", None)
    _pt_b_attr = getattr(dataset, "point_type_b", None)
    if _pt_a_attr is not None and _pt_b_attr is not None:
        _pt_a_np = to_numpy(_pt_a_attr)
        _pt_b_np = to_numpy(_pt_b_attr)
        if idxs is not None:
            _pt_a_np = _pt_a_np[idxs]
            _pt_b_np = _pt_b_np[idxs]
        _clean_mask = (_pt_a_np == PointType.MANIFOLD) & (_pt_b_np == PointType.MANIFOLD)
        if _clean_mask.any():
            z_a_clean = z_a[_clean_mask]
            z_b_clean = z_b[_clean_mask]

    # Filter out NaNs for metrics using param_values (u)
    if u.ndim == 2:
        valid_mask = ~np.isnan(u).any(axis=1)
    else:
        valid_mask = ~np.isnan(u)
    z_a_valid = z_a[valid_mask]
    z_b_valid = z_b[valid_mask]
    u_valid = u[valid_mask]

    # --- Joint + per-modality linear probe R2 ---
    for prefix, z in [('za', z_a_valid), ('zb', z_b_valid), ('zjoint', np.concatenate([z_a_valid, z_b_valid], axis=1))]:
        probe = linear_probe_r2(z, u_valid)
        for k, v in probe.items():
            metrics[f'geom/{prefix}/{k}'] = v

    # --- RankMe ---
    metrics['geom/rankme_a'] = rankme_score(z_a)
    metrics['geom/rankme_b'] = rankme_score(z_b)
    
    # --- VICReg Variance/Covariance/Invariance ---
    metrics['geom/vicreg_variance_a'] = vicreg_variance(z_a)
    metrics['geom/vicreg_variance_b'] = vicreg_variance(z_b)
    metrics['geom/vicreg_covariance_a'] = vicreg_covariance(z_a)
    metrics['geom/vicreg_covariance_b'] = vicreg_covariance(z_b)
    metrics['geom/vicreg_invariance'] = float(np.mean((z_a - z_b)**2))

    # --- Generalised Axis-Alignment / Diagonality Ratio ---
    if param_values.ndim == 2 and num_zdims >= 2:
        metrics['geom/za/diagonality_ratio'] = compute_diagonality_ratio(z_a_valid, u_valid, num_zdims, k_shared)

    # --- PCA axis-alignment ---
    n_active = k_shared if param_values.ndim == 2 else 1
    metrics['geom/pca_axis_align_a'] = pca_axis_alignment(z_a, n_active=n_active)
    metrics['geom/pca_axis_align_b'] = pca_axis_alignment(z_b, n_active=n_active)

    # --- Retrieval ---
    ret = retrieval_accuracy(z_a, z_b)
    for k, v in ret.items():
        metrics[f'geom/{k}'] = v
        
    # --- Masked Retrieval ---
    if predictor_a2b is not None and hasattr(predictor_a2b, 'weight'):
        w_np = predictor_a2b.weight.detach().cpu().numpy()
        mask = np.abs(w_np) > 0.5
        masked_ret = masked_retrieval_accuracy(z_a, z_b, mask)
        for k, v in masked_ret.items():
            metrics[f'geom/{k}'] = v

    # --- CCA ---
    cca = cca_score(z_a, z_b)
    for k, v in cca.items():
        metrics[f'geom/{k}'] = v
    cca_corr_spectrum = [cca.get(f'cca_corr_dim{i}', 0.0) for i in range(num_zdims)]

    # --- Found Common Rank Suite ---
    found_rank_metrics = compute_found_rank_metrics(
        z_a, z_b, num_zdims, cca_corr_spectrum, predictor_a2b, z_a_clean, z_b_clean
    )
    metrics.update(found_rank_metrics)

    # --- Norm diagnostics ---
    norm_metrics = compute_norm_diagnostics(z_a, z_b, dataset, idxs)
    metrics.update(norm_metrics)

    return metrics


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def _accumulate_partition_mse(
    mask: torch.Tensor,
    prefix: str,
    metrics: Dict[str, float],
    pred_a2b: torch.Tensor,
    z_b: torch.Tensor,
    pred_b2a: Optional[torch.Tensor] = None,
    z_a: Optional[torch.Tensor] = None,
) -> None:
    """Computes MSE for a given mask and accumulates it in metrics."""
    if not mask.any():
        return
    
    count = int(mask.sum().item())
    metrics[f"count_{prefix}"] += count
    
    metrics[f"align_mse_a2b_{prefix}"] += F.mse_loss(
        pred_a2b[mask], z_b[mask], reduction="sum"
    ).item()
    
    if pred_b2a is not None and z_a is not None:
        metrics[f"align_mse_b2a_{prefix}"] += F.mse_loss(
            pred_b2a[mask], z_a[mask], reduction="sum"
        ).item()


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
        "align_mse_a2b_asym": 0.0,
        "align_mse_b2a_asym": 0.0,
        "align_mse_a2b_external": 0.0,
        "count_manifold": 0,
        "count_asym": 0,
        "count_external": 0,
        "num_batches": 0,
    }

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
            data_a = batch["data_a"].to(device, non_blocking=True)
            data_b = batch["data_b"].to(device, non_blocking=True)
            corr_target = batch["corr_target"].to(device, non_blocking=True)

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
                    pt_a = pt_a.to(device, non_blocking=True)
                    pt_b = pt_b.to(device, non_blocking=True)
                    external_mask = (pt_a == PointType.EXTERNAL) | (pt_b == PointType.EXTERNAL)
                    manifold_mask = (pt_a == PointType.MANIFOLD) & (pt_b == PointType.MANIFOLD)
                    asym_mask = (~external_mask) & (~manifold_mask)

                    _accumulate_partition_mse(manifold_mask, "manifold", metrics, pred_a2b, z_b, pred_b2a, z_a)
                    _accumulate_partition_mse(asym_mask, "asym", metrics, pred_a2b, z_b, pred_b2a, z_a)
                    _accumulate_partition_mse(external_mask, "external", metrics, pred_a2b, z_b)

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
        # Note the key mapping from _eval_loop metrics to returned dict:
        # metrics["align_mse_a2b_asym"] (accumulated by helper with prefix "asym") is mapped
        # to key "align_mse_a2b_asym_corrupt" to match downstream code.
        out["align_mse_a2b_asym_corrupt"] = metrics["align_mse_a2b_asym"] / metrics["count_asym"]
        out["align_mse_b2a_asym_corrupt"] = metrics["align_mse_b2a_asym"] / metrics["count_asym"]
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
    point_size_min: float = 2.0,
    point_size_max: float = 16.0,
    point_size_default: float = 5.0,
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
        log_plots_to_wandb(dual_model, eval_set, device, step, wandb_run,
                           point_size_min=point_size_min,
                           point_size_max=point_size_max,
                           point_size_default=point_size_default)

        # Geometry metrics: R2, per-dim disentanglement, PCA axis-align, retrieval, CCA, norms
        geom_metrics = compute_geometry_metrics(
                dual_model, eval_set, device,
                predictor_a2b=predictors.get("a2b"),
            )
        wandb.log(geom_metrics, step=step)

        if is_3d is None:
            is_3d = (eval_set.data_a.shape[1] >= 3)

        if is_3d and log_interactive_3d:
            data_a = to_numpy(eval_set.data_a)
            data_b = to_numpy(eval_set.data_b)
            param_values = to_numpy(eval_set.param_values)
            idxs = None
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
            else:
                pt_a = getattr(eval_set, "point_type_a", None)
                pt_b = getattr(eval_set, "point_type_b", None)
                if pt_a is not None:
                    pt_a = to_numpy(pt_a)
                    pt_b = to_numpy(pt_b)
                    
                    if idxs is not None:
                        pt_a = pt_a[idxs]
                        pt_b = pt_b[idxs]

                html = build_interactive_4way_html(
                    data_a, data_b, out_a, out_b, np.asarray(param_values),
                    point_type_a=pt_a,
                    point_type_b=pt_b,
                    min_height_px=int(interactive_min_height),
                    predictor_a2b=predictors.get("a2b"),
                    point_size_min=point_size_min,
                    point_size_max=point_size_max,
                    point_size_default=point_size_default,
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

    cfg = load_config(str(cfg_path), cli_overrides=overrides)
    vis_cfg = cfg.get("visualization", {})
    point_size_min = vis_cfg.get("point_size_min", 2.0)
    point_size_max = vis_cfg.get("point_size_max", 16.0)
    point_size_default = vis_cfg.get("point_size_default", 5.0)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    data_cfg = cfg.data
    eval_num_samples = int(data_cfg.get('eval_num_samples', 4096))
    eval_seed = int(data_cfg.get('eval_seed', cfg.meta.seed + 1000))
    from omegaconf import OmegaConf
    eval_data_cfg_overrides = OmegaConf.create({'num_samples': eval_num_samples})
    eval_cfg = OmegaConf.merge(cfg, OmegaConf.create({'data': eval_data_cfg_overrides}))
    eval_set = build_dataset_from_config(eval_cfg, seed=eval_seed)
    effective_batch_size = int(batch_size) if batch_size is not None else int(data_cfg.get("batch_size", 128))
    pin_memory = (device.type == 'cuda')
    eval_loader = DataLoader(
        eval_set,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else data_cfg.get("num_workers", 0)),
        pin_memory=pin_memory
    )

    built = build_model_and_predictors(cfg, device)
    full_model = built["full_model"]
    dual_model = built["dual_model"]
    predictor_a2b = built["predictor_a2b"]
    predictor_b2a = built["predictor_b2a"]

    loss_type = cfg.loss.get("type", "ebm")
    loss_fn = build_loss_from_config(cfg, predictor_a2b, predictor_b2a)

    log_wandb_override = _to_bool_or_none(log_wandb)
    enabled_wandb = bool(cfg.logging.get("log_wandb", False)) if log_wandb_override is None else bool(log_wandb_override)
    run_dir = folder_path if folder_path is not None else (checkpoint_path.parent if checkpoint_path is not None else cfg_path.parent)

    base_tags = ["sslda", "eval"]
    if cfg.logging.get("log_seed_tag", False):
        base_tags.append(f"seed_{cfg.meta.seed}")

    wandb_run = setup_wandb(
        project="eb_jepa", config=cfg, run_dir=run_dir / "eval_wandb",
        run_name=f"{run_dir.name}_eval",
        tags=base_tags,
        group=cfg.logging.get("wandb_group"),
        enabled=enabled_wandb, resume=False,
    )

    ckpts = [checkpoint_path] if checkpoint_path is not None else _discover_checkpoints(run_dir)
    if not ckpts:
        raise ValueError(f"No checkpoints found in {run_dir}")

    is_3d = (eval_set.data_a.shape[1] >= 3)

    for idx, ckpt in enumerate(ckpts):
        is_last = (idx == (len(ckpts) - 1))
        ckpt_meta = load_checkpoint(ckpt, full_model, optimizer=None, device=device)
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
            point_size_min=point_size_min,
            point_size_max=point_size_max,
            point_size_default=point_size_default,
        )
        logger.info("Eval %s | loss=%.4f | a2b=%.4f | b2a=%.4f",
                    ckpt.name, metrics["loss"], metrics["align_mse_a2b"], metrics["align_mse_b2a"])

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    import fire
    fire.Fire(run)
