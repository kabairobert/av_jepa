import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


def setup_device(device: str = "auto") -> torch.device:
    """Set up the compute device. Options: 'auto', 'cuda', or 'cpu'."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    return device


def setup_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_wandb(
    project: str,
    config: Union[Dict, DictConfig],
    run_dir: Union[str, Path],
    run_name: Optional[str] = None,
    resume: bool = True,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    enabled: bool = True,
    sweep_id: Optional[str] = None,
):
    """Initialize W&B with safe resume (preserves existing run metadata on resume)."""
    # Respect WANDB_DISABLED environment variable (used by wandb itself)
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        logger.info("W&B logging disabled via WANDB_DISABLED environment variable")
        return None

    if not enabled:
        logger.info("W&B logging disabled")
        return None

    import wandb

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id_file = run_dir / "wandb_run_id.txt"

    # Convert OmegaConf to dict if needed
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Handle wandb sweep registration via environment variables
    # This is how wandb associates runs with sweeps
    if sweep_id:
        os.environ["WANDB_SWEEP_ID"] = sweep_id
        logger.info(f"Registering run with wandb sweep: {sweep_id}")
        if tags:
            tags = list(tags) + [f"sweep_{sweep_id}"]
        else:
            tags = [f"sweep_{sweep_id}"]

    # Check if we should resume an existing run
    if resume and run_id_file.exists():
        with open(run_id_file, "r") as f:
            existing_run_id = f.read().strip()

        # For sweep runs, use environment variables for resume
        if sweep_id:
            os.environ["WANDB_RUN_ID"] = existing_run_id
            os.environ["WANDB_RESUME"] = "allow"
            wandb_config = {
                "project": project,
                "dir": str(run_dir),
                "config": config,
            }
            if run_name:
                wandb_config["name"] = run_name
            if tags:
                wandb_config["tags"] = tags
            if group:
                wandb_config["group"] = group
            run = wandb.init(**wandb_config)
            logger.info(f"Resumed W&B run: {existing_run_id} in sweep {sweep_id}")
            return run

        # SAFE RESUME: Only pass id and resume flag - do NOT pass name/config/tags
        # This prevents overwriting existing run metadata on W&B
        wandb_config = {
            "project": project,
            "dir": str(run_dir),
            "id": existing_run_id,
            "resume": "must",  # "must" = fail if run doesn't exist (safer than "allow")
        }
        if group:
            wandb_config["group"] = group

        try:
            run = wandb.init(**wandb_config)
            logger.info(
                f"Resumed W&B run: {existing_run_id} (existing config preserved)"
            )
            return run
        except wandb.errors.UsageError:
            # Run doesn't exist anymore on W&B, create new one
            logger.warning(f"W&B run {existing_run_id} not found, creating new run")
            run_id_file.unlink()  # Remove stale run ID file

    # NEW RUN: Pass all configuration
    wandb_config = {
        "project": project,
        "dir": str(run_dir),
        "config": config,
    }
    if run_name:
        wandb_config["name"] = run_name
    if tags:
        wandb_config["tags"] = tags
    if group:
        wandb_config["group"] = group

    run = wandb.init(**wandb_config)
    with open(run_id_file, "w") as f:
        f.write(run.id)
    logger.info(f"Created W&B run: {run.id}")

    return run


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    scaler: Optional[Any] = None,
    **extra_state,
) -> None:
    """Save a training checkpoint (model, optimizer, scheduler, scaler, extra_state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    checkpoint.update(extra_state)

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint: {path}")


def save_config(cfg: DictConfig, exp_dir: Union[str, Path]) -> None:
    """Save a copy of the config to the experiment directory as config.yaml.

    Does NOT overwrite an existing config.yaml so the original config that
    started the experiment is always preserved, even on resume.
    """
    config_path = Path(exp_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        OmegaConf.save(cfg, config_path)
        logger.info(f"Saved config to {config_path}")
    else:
        logger.debug(f"Config already exists at {config_path}, skipping save.")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint. Returns dict with epoch, step, and extra_state.

    The returned 'epoch' is the epoch to resume training from (0-indexed).
    If no checkpoint exists, returns epoch=0 to start fresh.
    If a checkpoint exists with epoch=N, returns epoch=N+1 to resume from the next epoch.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {path}")
        return {"epoch": 0, "step": 0, "resumed": False}

    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Handle compiled model state dicts
    state_dict = checkpoint.get("model_state_dict", {})
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded model state from: {path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Restored optimizer state")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored scheduler state")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("Restored scaler state")

    return {
        "epoch": checkpoint.get("epoch", 0) + 1,  # Resume from next epoch
        "step": checkpoint.get("step", 0),
        "resumed": True,
        **{
            k: v
            for k, v in checkpoint.items()
            if k
            not in [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "epoch",
                "step",
            ]
        },
    }


def load_config(
    config_path: Union[str, Path],
    cli_overrides: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> DictConfig:
    """Load YAML config with optional dot-notation overrides (e.g., 'model.lr': 0.001)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if not quiet:
        logger.info(f"Loaded config from {config_path}")

    if cli_overrides:
        # Convert dot notation to nested dict
        override_dict = {}
        for key, value in cli_overrides.items():
            keys = key.split(".")
            current = override_dict
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value

        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_dict))
        if not quiet:
            logger.info(f"Applied {len(cli_overrides)} config overrides")

    return cfg


def get_checkpoints_dir() -> Path:
    """Get the base checkpoints directory from EBJEPA_CKPTS env variable."""
    return Path(os.environ.get("EBJEPA_CKPTS", "checkpoints"))


def get_unified_experiment_dir(
    example_name: str,
    sweep_name: str,
    exp_name: str,
    seed: int,
    base_dir: Union[str, Path, None] = None,
    create: bool = True,
) -> Path:
    """Create experiment dir: {base_dir}/{example_name}/{sweep_name}/{exp_name}_seed{seed}."""
    if base_dir is None:
        base_dir = get_checkpoints_dir()

    # Convert to absolute path to avoid issues when cwd changes (e.g., after os.chdir)
    exp_dir = (
        Path(base_dir) / example_name / sweep_name / f"{exp_name}_seed{seed}"
    ).absolute()

    if create:
        exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def get_default_sweep_name() -> str:
    return datetime.now().strftime("sweep_%Y-%m-%d_%H-%M")


def get_default_dev_name() -> str:
    return datetime.now().strftime("dev_%Y-%m-%d_%H-%M")


def get_exp_name(example_name: str, cfg) -> str:
    """Get short experiment name encoding key hyperparameters (seed appended separately)."""
    if example_name == "image_jepa":
        # Use same compact naming convention as video_jepa:
        # model -> batch -> loss -> proj{HxO} -> loss-params -> lr
        def _fmt_num(x):
            try:
                xf = float(x)
            except Exception:
                return str(x)
            if xf.is_integer():
                return str(int(xf))
            return str(x)

        parts = [str(cfg.model.type), f"bs{cfg.data.batch_size}"]

        # loss type
        try:
            loss = cfg.loss.get("type", "vcreg")
        except Exception:
            loss = getattr(cfg.loss, "type", "vcreg")
        parts.append(loss)

        # projector dims (always include projHxO)
        ph = getattr(cfg.model, "proj_hidden_dim", None)
        po = getattr(cfg.model, "proj_output_dim", None)
        if ph is None and hasattr(cfg.model, "proj_hidden_dim"):
            ph = getattr(cfg.model, "proj_hidden_dim")
        if po is None and hasattr(cfg.model, "proj_output_dim"):
            po = getattr(cfg.model, "proj_output_dim")
        # fallback to dstc if available
        if ph is None and hasattr(cfg.model, "dstc"):
            ph = cfg.model.dstc * 4
        if po is None and hasattr(cfg.model, "dstc"):
            po = cfg.model.dstc
        if ph is None:
            ph = 0
        if po is None:
            po = 0
        parts.append(f"proj{_fmt_num(ph)}x{_fmt_num(po)}")

        # loss params
        if loss in ("vcreg", "vc", "vicreg"):
            std = None
            cov = None
            try:
                std = cfg.loss.get("std_coeff", None)
            except Exception:
                std = getattr(cfg.loss, "std_coeff", None)
            try:
                cov = cfg.loss.get("cov_coeff", None)
            except Exception:
                cov = getattr(cfg.loss, "cov_coeff", None)
            if std is not None:
                parts.append(f"std{_fmt_num(std)}")
            if cov is not None:
                parts.append(f"cov{_fmt_num(cov)}")
        elif loss in ("bcs", "sigreg"):
            lmbd = None
            try:
                lmbd = cfg.loss.get("lmbd", None)
            except Exception:
                lmbd = getattr(cfg.loss, "lmbd", None)
            if lmbd is not None:
                parts.append(f"lmbd{_fmt_num(lmbd)}")

        # learning rate
        if hasattr(cfg.optim, "lr"):
            lr = cfg.optim.lr
            try:
                lr_f = float(lr)
                if lr_f >= 1e-2:
                    lr_str = f"{lr_f:.3g}"
                else:
                    lr_str = f"{lr_f:.0e}"
            except Exception:
                lr_str = str(lr)
            parts.append(f"lr{lr_str}")

        return "_".join(str(p) for p in parts)
    elif example_name == "video_jepa":
        # Build a consistent, compact run name:
        # model -> batch -> loss -> proj{HxO} -> loss-params -> lr
        def _fmt_num(x):
            try:
                xf = float(x)
            except Exception:
                return str(x)
            if xf.is_integer():
                return str(int(xf))
            return str(x)

        parts = ["resnet"]
        parts.append(f"bs{cfg.data.batch_size}")

        # Determine loss type robustly; default to 'vcreg' when unspecified
        try:
            loss = cfg.loss.get("type", "vcreg")
        except Exception:
            try:
                loss = cfg.loss.type
            except Exception:
                loss = "vcreg"
        parts.append(loss)

        # Projector dims: always include proj{HxO}; use 0x0 when absent
        ph = getattr(cfg.model, "proj_hidden_dim", None)
        po = getattr(cfg.model, "proj_output_dim", None)

        # Determine loss type defaults for multipliers
        try:
            loss_type = cfg.loss.get("type", "vcreg")
        except Exception:
            try:
                loss_type = cfg.loss.type
            except Exception:
                loss_type = "vcreg"

        _BCS_TYPES = ("bcs", "bcs-euler")
        if loss_type in _BCS_TYPES:
            default_h_mult = 4
            default_o_mult = 1
        else:
            default_h_mult = 4
            default_o_mult = 4

        # fallback to dstc-derived dims for older configs; allow multipliers under cfg.loss
        if ph is None and hasattr(cfg.model, "dstc"):
            h_mult = cfg.loss.get("proj_hidden_mult", default_h_mult)
            ph = cfg.model.dstc * h_mult
        if po is None and hasattr(cfg.model, "dstc"):
            o_mult = cfg.loss.get("proj_out_mult", default_o_mult)
            po = cfg.model.dstc * o_mult
        if ph is None:
            ph = 0
        if po is None:
            po = 0
        parts.append(f"proj{_fmt_num(ph)}x{_fmt_num(po)}")

        # Loss-specific params
        if loss in ("vcreg", "vc", "vicreg"):
            std = None
            cov = None
            try:
                std = cfg.loss.get("std_coeff", None)
            except Exception:
                std = getattr(cfg.loss, "std_coeff", None)
            try:
                cov = cfg.loss.get("cov_coeff", None)
            except Exception:
                cov = getattr(cfg.loss, "cov_coeff", None)
            if std is not None:
                parts.append(f"std{_fmt_num(std)}")
            if cov is not None:
                parts.append(f"cov{_fmt_num(cov)}")
        elif loss in ("bcs", "bcs-euler", "sigreg"):
            # number of slices and lambda
            try:
                ns = cfg.loss.get("num_slices", None)
            except Exception:
                ns = getattr(cfg.loss, "num_slices", None)
            try:
                lmbd = cfg.loss.get("lmbd", None)
            except Exception:
                lmbd = getattr(cfg.loss, "lmbd", None)
            if ns is not None:
                parts.append(f"ns{_fmt_num(ns)}")
            if lmbd is not None:
                parts.append(f"lmbd{_fmt_num(lmbd)}")
            # For Euler variants, encode the ECF sync mode in the run name
            # so distributed vs local ablations produce distinct checkpoint dirs
            if loss == "bcs-euler":
                try:
                    ecf_distributed_sync = cfg.loss.get("ecf_distributed_sync", None)
                except Exception:
                    ecf_distributed_sync = getattr(
                        cfg.loss, "ecf_distributed_sync", None
                    )

                # Backward compatibility for old configs using ecf_sync string
                if ecf_distributed_sync is None:
                    try:
                        legacy_sync = cfg.loss.get("ecf_sync", "distributed")
                    except Exception:
                        legacy_sync = getattr(cfg.loss, "ecf_sync", "distributed")
                    ecf_distributed_sync = legacy_sync == "distributed"
                # Abbreviate: distributed -> dist, local -> loc
                sync_abbrev = "dist" if ecf_distributed_sync else "loc"
                parts.append(f"sync{sync_abbrev}")

        # learning rate
        if hasattr(cfg.optim, "lr"):
            # format lr compactly (1e-3 -> 1e-3, 0.001 -> 1e-3)
            lr = cfg.optim.lr
            try:
                lr_f = float(lr)
                if lr_f >= 1e-2:
                    lr_str = f"{lr_f:.3g}"
                else:
                    lr_str = f"{lr_f:.0e}"
            except Exception:
                lr_str = str(lr)
            parts.append(f"lr{lr_str}")

        return "_".join(str(p) for p in parts)
    elif example_name == "ac_video_jepa":
        return (
            f"{cfg.model.encoder_architecture}"
            f"_cov{cfg.model.regularizer.cov_coeff}"
            f"_std{cfg.model.regularizer.std_coeff}"
            f"_simt{cfg.model.regularizer.get('sim_coeff_t')}"
            f"_idm{cfg.model.regularizer.get('idm_coeff')}"
        )
    else:
        return "exp"


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dict as 'loss=0.1234 | acc=95.12'."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def log_epoch(
    epoch: int,
    metrics: Dict[str, float],
    total_epochs: Optional[int] = None,
    elapsed_time: Optional[float] = None,
) -> None:
    """Log epoch summary: 📊 [Epoch 001/100] metric1=val1 | metric2=val2 | time=123.4s."""
    if total_epochs:
        prefix = f"[Epoch {epoch:03d}/{total_epochs}]"
    else:
        prefix = f"[Epoch {epoch:03d}]"

    metrics_str = format_metrics(metrics)

    if elapsed_time is not None:
        logger.info(f"📊 {prefix} {metrics_str} | time={elapsed_time:.1f}s")
    else:
        logger.info(f"📊 {prefix} {metrics_str}")


def log_model_info(model: nn.Module, param_counts: Dict[str, int]) -> None:
    """Log model structure and parameter counts."""
    logger.info(f"🧠 Model:\n{model}")
    param_str = " | ".join(f"{k}={v:,}" for k, v in param_counts.items())
    logger.info(f"🔢 Parameters: {param_str}")


def log_data_info(
    dataset_name: str,
    num_batches: int,
    batch_size: int,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
) -> None:
    """Log dataset information."""
    if train_samples is not None and val_samples is not None:
        logger.info(
            f"📦 Data: {dataset_name} | {num_batches} batches x {batch_size} samples | "
            f"train={train_samples:,} | val={val_samples:,}"
        )
    else:
        logger.info(
            f"📦 Data: {dataset_name} | {num_batches} batches x {batch_size} samples"
        )


def log_config(cfg: Union[Dict, DictConfig], title: str = "Run Configuration") -> None:
    """Log configuration in a readable format."""
    logger.info("=" * 60)
    logger.info(f"⚙️  {title}:")
    logger.info("=" * 60)

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    for section, values in cfg.items():
        if isinstance(values, dict):
            for key, value in values.items():
                logger.info(f"  {section}.{key}={value}")
        else:
            logger.info(f"  {section}={values}")
    logger.info("=" * 60)
