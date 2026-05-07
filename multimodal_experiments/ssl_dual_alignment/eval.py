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
    if param_values.ndim == 1:
        vals = param_values
    else:
        vals = param_values[:, 0]
    denom = (vals.max() - vals.min()) + 1e-8
    return (vals - vals.min()) / denom


def _build_interactive_4way_html(
    data_a: np.ndarray,
    data_b: np.ndarray,
    out_a: np.ndarray,
    out_b: np.ndarray,
    param_values: np.ndarray,
    min_height_px: int = 420,
) -> Optional[str]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:
        logger.warning("Plotly not available; skipping interactive 3D plot: %s", exc)
        return None

    color_vals = _get_color_values(param_values)

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scatter3d"}] * 4],
        subplot_titles=("Input Space A", "Output Space A", "Output Space B", "Input Space B"),
    )

    def _scatter(xyz, name, show_scale=False):
        return go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=color_vals,
                colorscale="Turbo",
                showscale=show_scale,
                colorbar=dict(thickness=15, len=0.7),
            ),
            name=name,
        )

    fig.add_trace(_scatter(data_a, "Input A"), row=1, col=1)
    fig.add_trace(_scatter(out_a, "Output A"), row=1, col=2)
    fig.add_trace(_scatter(out_b, "Output B"), row=1, col=3)
    fig.add_trace(_scatter(data_b, "Input B", show_scale=True), row=1, col=4)

    fig.update_layout(
        autosize=True,
        height=min_height_px,
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=False,
        hovermode="closest",
    )
    scene_aspect = dict(aspectmode="manual", aspectratio=dict(x=1.6, y=1.0, z=0.9))
    fig.update_layout(scene=scene_aspect, scene2=scene_aspect, scene3=scene_aspect, scene4=scene_aspect)

    html_body = fig.to_html(full_html=False, include_plotlyjs="cdn", default_width="100%", default_height="100%")
    wrapped = f"<div style='width:100%;height:100%;min-height:{int(min_height_px)}px'>{html_body}</div>"
    return wrapped


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
                err_a2b = F.mse_loss(predictors["a2b"](z_a), z_b).item()
                err_b2a = F.mse_loss(predictors["b2a"](z_b), z_a).item()
                metrics["align_mse_a2b"] += err_a2b
                metrics["align_mse_b2a"] += err_b2a

            metrics["loss"] += float(loss.item())
            metrics["num_batches"] += 1

            if max_batches is not None and (bi + 1) >= int(max_batches):
                break

    denom = max(metrics["num_batches"], 1)
    return {
        "loss": metrics["loss"] / denom,
        "align_mse_a2b": metrics["align_mse_a2b"] / denom,
        "align_mse_b2a": metrics["align_mse_b2a"] / denom,
    }


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
        logger.info("Applied %d CLI override(s)", len(overrides))

    device = setup_device(cfg_obj.meta.device)
    setup_seed(cfg_obj.meta.seed)

    # Dataset (reuse train data)
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
    )
    effective_batch_size = int(batch_size) if batch_size is not None else int(data_cfg.get("batch_size", 128))
    effective_num_workers = int(num_workers) if num_workers is not None else int(data_cfg.get("num_workers", 0))
    eval_loader = DataLoader(
        eval_set,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
    )

    # Model + predictors
    built = build_model_and_predictors(cfg_obj, device)
    dual_model = built["dual_model"]
    predictor_a2b = built["predictor_a2b"]
    predictor_b2a = built["predictor_b2a"]

    loss_type = cfg_obj.loss.get("type", "ebm")
    if loss_type == "ebm":
        loss_fn = EBMJEPALoss(
            predictor_a2b,
            predictor_b2a,
            lambda_jac=cfg_obj.loss.get("lambda_jac", 1.0),
            lambda_prior=cfg_obj.loss.get("lambda_prior", 0.5),
            lambda_sparse=cfg_obj.loss.get("lambda_sparse", 0.1),
            use_l1=cfg_obj.loss.get("use_l1", False),
        )
    else:
        loss_fn = SupervisedFactorLoss(
            dimensions_per_factor=[1, 1] if data_cfg.get("type", "2d") == "2d" else [1, 1, 1]
        )

    # W&B
    log_wandb_override = _to_bool_or_none(log_wandb)
    enabled_wandb = bool(cfg_obj.logging.get("log_wandb", False)) if log_wandb_override is None else bool(log_wandb_override)
    run_dir = folder_path if folder_path is not None else (checkpoint_path.parent if checkpoint_path is not None else cfg_path.parent)
    run_name = f"{run_dir.name}_eval"

    wandb_run = setup_wandb(
        project="eb_jepa",
        config=cfg_obj,
        run_dir=run_dir / "eval_wandb",
        run_name=run_name,
        tags=["dual_disentangle", "eval", f"seed_{cfg_obj.meta.seed}"],
        group=cfg_obj.logging.get("wandb_group"),
        enabled=enabled_wandb,
        resume=False,
    )

    # Checkpoints
    if checkpoint_path is not None:
        ckpts = [checkpoint_path]
    else:
        ckpts = _discover_checkpoints(run_dir)

    if not ckpts:
        raise ValueError(f"No checkpoints found in {run_dir}")

    data_type = str(data_cfg.get("type", "2d"))
    is_3d = data_type.startswith("3d")

    for idx, ckpt in enumerate(ckpts):
        is_last = (idx == (len(ckpts) - 1))
        ckpt_meta = load_checkpoint(ckpt, dual_model, optimizer=None, device=device)
        ckpt_step = ckpt_meta.get("step", None)
        if ckpt_step is None:
            ckpt_step = int(_checkpoint_epoch(ckpt))
        ckpt_step = int(ckpt_step)

        metrics = _eval_loop(
            eval_loader,
            dual_model,
            loss_fn,
            loss_type,
            {"a2b": predictor_a2b, "b2a": predictor_b2a},
            device,
            max_batches=max_batches,
        )

        logs = {
            "val/loss": metrics["loss"],
            "val/align_mse_a2b": metrics["align_mse_a2b"],
            "val/align_mse_b2a": metrics["align_mse_b2a"],
            "eval/checkpoint_name": ckpt.name,
            "eval/checkpoint_path": str(ckpt),
            "eval/checkpoint_step": int(ckpt_step),
        }

        if wandb_run:
            import wandb

            wandb.log(logs, step=ckpt_step)
            # Static 4-way plots in media
            log_plots_to_wandb(dual_model, eval_set, device, ckpt_step, wandb_run)

            # Interactive HTML (3D only, last checkpoint)
            if is_3d and is_last and log_interactive_3d:
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

                html = _build_interactive_4way_html(
                    data_a,
                    data_b,
                    out_a,
                    out_b,
                    np.asarray(param_values),
                    min_height_px=int(interactive_min_height),
                )
                if html is not None:
                    wandb.log({"interactive_3d_4way_html": wandb.Html(html)}, step=ckpt_step)

        logger.info(
            "Eval %s | loss=%.4f | a2b=%.4f | b2a=%.4f",
            ckpt.name,
            metrics["loss"],
            metrics["align_mse_a2b"],
            metrics["align_mse_b2a"],
        )

    if wandb_run:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(run)
