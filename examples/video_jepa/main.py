"""
Video JEPA Training Script

Train a self-supervised video prediction model on Moving MNIST using
Joint Embedding Predictive Architecture (JEPA) with VC regularization.
"""

import gc
import os
from pathlib import Path

import fire
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
try:
    import psutil
except Exception:
    psutil = None
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    save_config,
    setup_device,
    setup_seed,
    setup_wandb,
    _get_process_rss_mb,
    _get_cuda_mem_mb,
    _get_param_mem_breakdown,
    _get_param_dtype_counts,
    _log_memory_snapshot,
    _log_tensor_shapes,
)
from examples.video_jepa.diagnostics import DiagnosticsManager
from examples.video_jepa.eval import validation_loop
from examples.video_jepa.model_builder import build_video_jepa_and_probes
from examples.video_jepa.vis import assemble_geometry_viz_videos

logger = get_logger(__name__)
# Memory/dtype probe helpers have been centralized in `eb_jepa.training_utils`.


def _post_diagnostics_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _int_or_none(value):
    if value in (None, "", 0):
        return None
    return int(value)


def _resolve_epoch_interval(configured_value, fallback):
    if configured_value in (None, ""):
        return max(1, int(fallback))
    return max(1, int(configured_value))


def _probe_loader_and_batches(mode, val_loader, train_probe_loader, diagnostics_cfg):
    mode = str(mode).lower()
    if mode == "train_subset":
        return train_probe_loader, int(diagnostics_cfg.get("train_subset_num_batches", 2)), "train"
    if mode == "val_subset":
        return val_loader, int(diagnostics_cfg.get("val_subset_num_batches", 4)), "val"
    return val_loader, None, "val"


def _geometry_cfg_for_event(geometry_cfg, enable_geometry):
    cfg_dict = OmegaConf.to_container(geometry_cfg, resolve=True) if not isinstance(geometry_cfg, dict) else dict(geometry_cfg)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}
    if not enable_geometry:
        cfg_dict = {**cfg_dict, "enabled": False}
    return cfg_dict


def _capture_shapes(sample_x, encoder, projector, device, force_runtime_shapes: bool = False):
    """Capture raw input, encoder output, and projector output shapes safely.

    Returns a dict with stringified shapes for keys: raw_input, encoder_output, projector_output.
    """
    try:
        with torch.no_grad():
            x = sample_x.to(device)

            # Normalize possible single-sample shapes to have a batch dim.
            # Common dataset single-sample shapes: (C, T, H, W) or (C, H, W).
            try:
                if x.dim() == 4:
                    # Heuristic: if first dim looks like channels (small) and second dim > 1 (time),
                    # then treat as (C, T, H, W) and add batch dim -> (1, C, T, H, W).
                    if x.shape[0] <= 4 and x.shape[1] > 1:
                        x = x.unsqueeze(0)
                elif x.dim() == 3:
                    # (C, H, W) -> (1, C, H, W)
                    x = x.unsqueeze(0)
            except Exception:
                logger.exception("Failed normalizing sample shape for capture")

            # encoder: prefer full input, fall back to single frame if needed
            try:
                enc_out = encoder(x)
            except Exception:
                logger.exception("Encoder failed on full input when capturing shapes")
                try:
                    # Try single-frame fallback if time dim exists
                    if x.dim() == 5:
                        enc_out = encoder(x[:, :, 0])
                    else:
                        enc_out = encoder(x)
                except Exception:
                    logger.exception("Encoder single-frame fallback also failed when capturing shapes")
                    enc_out = None

            enc_shape = str(list(enc_out.shape)) if enc_out is not None else None

            proj_str = None
            if enc_out is not None:
                # Encoder outputs are typically [B, C, T, H, W] (or [B, C, H, W]).
                try:
                    proj_str = None
                    # Infer sample count without materializing the full flattened tensor
                    if enc_out.dim() == 5:
                        b, c, t, h, w = enc_out.shape
                        n_samples = int(b * t * h * w)
                    elif enc_out.dim() == 4:
                        b, c, h, w = enc_out.shape
                        n_samples = int(b * h * w)
                        t = None
                    elif enc_out.dim() == 2:
                        n_samples = int(enc_out.size(0))
                        c = enc_out.size(1)
                        t = None
                    else:
                        # Fallback: try to infer from flattened view
                        try:
                            proj_in_tmp = enc_out.reshape(enc_out.size(0), -1)
                            n_samples = int(proj_in_tmp.size(0))
                            c = proj_in_tmp.size(1)
                            t = None
                        except Exception:
                            n_samples = None

                    # Use memory-free reporting by default; optionally run a lightweight
                    # runtime forward pass only when explicitly requested.
                    if force_runtime_shapes:
                        # Materialize a small runtime sample for accurate runtime shapes
                        try:
                            if enc_out.dim() == 5:
                                proj_in = enc_out.permute(0, 2, 3, 4, 1).reshape(-1, c)
                            elif enc_out.dim() == 4:
                                proj_in = enc_out.permute(0, 2, 3, 1).reshape(-1, c)
                            else:
                                proj_in = enc_out

                            try:
                                dev = next(projector.parameters()).device
                                proj_in = proj_in.to(dev)
                            except Exception:
                                pass
                            proj_in = proj_in.contiguous().float()

                            # sample small number of rows to avoid heavy compute
                            if proj_in.size(0) > 8:
                                proj_in = proj_in[:8]

                            proj_str = projector.shape_str(input_tensor=proj_in)
                        except Exception:
                            logger.exception("Projector.shape_str runtime evaluation failed")
                            proj_str = None
                    else:
                        try:
                            proj_str = projector.shape_str(input_tensor=None, n_samples=n_samples)
                        except Exception:
                            logger.exception("Projector.shape_str inference failed")
                            proj_str = None
                except Exception:
                    logger.exception("Failed preparing projector input for shape capture")
                    proj_str = None

            return {
                "raw_input": str(list(sample_x.shape)),
                "encoder_output": enc_shape,
                "projector_output": proj_str,
            }
    except Exception:
        logger.exception("Failed capturing tensor shapes")
        return {"raw_input": str(list(sample_x.shape))}


def _collect_memory_snapshot_metrics(step, modules, prefix="train", dtype=None):
    rss = _get_process_rss_mb()
    cuda = _get_cuda_mem_mb()
    params = _get_param_mem_breakdown(modules)
    dtype_counts = _get_param_dtype_counts(modules)
    metrics = {f"{prefix}/mem/rss_mb": rss if rss is not None else -1}
    metrics.update({f"{prefix}/{k}": v for k, v in cuda.items()})
    metrics.update(params)
    for key, value in dtype_counts.items():
        if isinstance(value, dict):
            for dtype_name, count in value.items():
                metrics[f"{key}/{dtype_name}"] = count
        else:
            metrics[key] = -1
    metrics[f"{prefix}/dtype"] = str(dtype) if dtype is not None else "unknown"
    metrics[f"{prefix}/probe_step"] = int(step)
    return metrics


def run(
    fname: str = "examples/video_jepa/cfgs/default.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train a Video JEPA model on Moving MNIST.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., model.lr=0.001)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Mitigate CUDA memory fragmentation under interleaved train/eval diagnostics.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Setup
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            folder_name = exp_dir.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("video_jepa", cfg)
            exp_dir = get_unified_experiment_dir(
                example_name="video_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
                base_dir=cfg.meta.get("checkpoint_dir", None),
            )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        # Extract exp_name from folder name by removing _seed{seed} suffix
        folder_name = exp_dir.name  # e.g., "resnet_std10.0_cov100.0_seed1"
        exp_name = folder_name.rsplit("_seed", 1)[0]  # e.g., "resnet_std10.0_cov100.0"

    # Save config next to checkpoints (skipped if already exists, e.g. on resume)
    save_config(cfg, exp_dir)

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(wandb_cfg, dict):
        wandb_cfg = {}

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "video_jepa", **wandb_cfg},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["video_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    # Load datasets
    train_set = MovingMNISTDet(split="train")
    val_set = MovingMNISTDet(split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    train_probe_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    log_data_info(
        "MovingMNIST",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_set),
        val_samples=len(val_set),
    )
    steps_per_epoch = len(train_loader)

    # Initialize Video JEPA model and probes
    logger.info("Initializing model...")
    loss_type = cfg.loss.get("type", "vcreg")
    logger.info(f"Using regularizer: {loss_type}")
    built = build_video_jepa_and_probes(cfg, device)
    jepa = built["jepa"]
    pixel_decoder = built["pixel_decoder"]
    detection_head = built["detection_head"]
    encoder = built["encoder"]
    predictor = built["predictor"]
    projector = built["projector"]
    regularizer = built["regularizer"]

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    # One-time probe flags
    first_train_probe_done = False
    first_val_probe_done = False

    # Modules to include in param memory breakdown (add probes for decoder/head/jepa)
    probe_modules = {
        "jepa": jepa,
        "encoder": encoder,
        "predictor": predictor,
        "projector": projector,
        "regularizer": regularizer,
        "pixel_decoder": pixel_decoder,
        "detection_head": detection_head,
    }

    # Mixed precision setup
    train_cfg = cfg.get("training", {})
    geometry_cfg = cfg.logging.get("geometry_viz", {})
    diagnostics_cfg = cfg.logging.get("diagnostics", {})
    use_amp = train_cfg.get("use_amp", False)
    dtype_str = train_cfg.get("dtype", "float32").lower()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    # Guard AMP usage on CPU-only setups to avoid unsupported float16 paths
    if use_amp and not torch.cuda.is_available():
        logger.warning("AMP requested but CUDA not available — disabling AMP for safety on CPU")
        use_amp = False

    scaler = GradScaler(enabled=use_amp)
    logger.info(f"Using AMP: {use_amp} with dtype: {dtype}")

    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", True))
    diagnostics_manager = DiagnosticsManager(
        run_dir=exp_dir,
        wandb_run=wandb_run,
        enabled=diagnostics_enabled,
        upload_artifacts=bool(diagnostics_cfg.get("upload_artifacts", True)),
        flush_interval_sec=float(diagnostics_cfg.get("artifact_flush_interval_sec", 3.0)),
        run_kind="train",
    )
    # Read cadences from top-level logging (new structure)
    # diagnostics_every_epochs can be inherited from save_every if unset
    diag_every_epochs_cfg = cfg.logging.get("diagnostics_every_epochs", None)
    canonical_every_epochs = _resolve_epoch_interval(diag_every_epochs_cfg, cfg.logging.save_every)
    
    # Fast diagnostics cadences (independent, both optional)
    fast_every_epochs = _int_or_none(cfg.logging.get("fast_diagnostics_every_epochs", None))
    fast_every_steps = _int_or_none(cfg.logging.get("fast_diagnostics_every_steps", None))
    
    # Read modes from logging.diagnostics (renamed fields)
    canonical_mode = str(diagnostics_cfg.get("full_diagnostics_mode", "full_val")).lower()
    fast_mode = str(diagnostics_cfg.get("fast_diagnostics_mode", "val_subset")).lower()

    def _run_diagnostics_event(
        *,
        event_type,
        probe_mode,
        step,
        epoch_value,
        metrics_prefix="",
        enable_geometry=False,
        persist=False,
        event_metadata=None,
    ):
        probe_loader, max_batches, diagnostics_phase = _probe_loader_and_batches(
            probe_mode,
            val_loader=val_loader,
            train_probe_loader=train_probe_loader,
            diagnostics_cfg=diagnostics_cfg,
        )
        return validation_loop(
            val_loader=probe_loader,
            jepa=jepa,
            detection_head=detection_head,
            pixel_decoder=pixel_decoder,
            steps=cfg.model.steps,
            device=device,
            use_amp=use_amp,
            dtype=dtype,
            geometry_cfg=_geometry_cfg_for_event(geometry_cfg, enable_geometry),
            epoch=epoch_value,
            exp_dir=exp_dir,
            max_batches=max_batches,
            log_step=step,
            steps_per_epoch=steps_per_epoch,
            diagnostics_manager=diagnostics_manager,
            diagnostics_event_type=event_type,
            diagnostics_phase=diagnostics_phase,
            diagnostics_metadata={
                "probe_mode": probe_mode,
                **(event_metadata or {}),
            },
            metrics_prefix=metrics_prefix,
            emit_media=enable_geometry,
            persist_diagnostics=persist,
        )

    # Set learning rates for different components
    # Lower learning rate for pixel decoder to prevent overfitting
    optimizer = Adam(
        [
            {"params": jepa.parameters(), "lr": cfg.optim.lr},
            {"params": pixel_decoder.head.parameters(), "lr": cfg.optim.lr / 10},
            {"params": detection_head.head.parameters(), "lr": cfg.optim.lr},
        ]
    )

    # Log configuration
    log_config(cfg)

    # Load checkpoint if requested
    start_epoch = 0
    global_step = 0
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path, jepa, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step", 0)

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")

    for epoch in range(start_epoch, cfg.optim.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["video"]
            loc_map = batch["digit_location"]

            optimizer.zero_grad()
            with autocast(device.type, dtype=dtype, enabled=use_amp):
                _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                    x,
                    actions=None,
                    nsteps=cfg.model.steps,
                    unroll_mode="parallel",
                    compute_loss=True,
                    return_all_steps=False,
                )
                recon_loss = pixel_decoder(x, x)
                det_loss = detection_head(x, loc_map)
                total_loss = jepa_loss + recon_loss + det_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # One-time post-first-step memory probe (simplified)
            if not first_train_probe_done:
                try:
                    shapes = {}
                    _log_memory_snapshot(global_step, probe_modules, wandb_run=wandb_run, prefix="train")
                    if cfg.logging.get("log_tensor_shapes", True):
                        shapes = _capture_shapes(
                            x,
                            encoder,
                            projector,
                            device,
                            cfg.logging.get("projector_force_runtime_shapes", False),
                        )
                        _log_tensor_shapes(global_step, shapes, wandb_run=wandb_run, prefix="train")
                    diagnostics_manager.record_event(
                        event_type="train_health_probe",
                        phase="train",
                        step=global_step,
                        epoch=epoch,
                        metrics={
                            **_collect_memory_snapshot_metrics(global_step, probe_modules, prefix="train", dtype=dtype),
                            **({f"train/shape/{k}": v for k, v in shapes.items()} if cfg.logging.get("log_tensor_shapes", True) else {}),
                        },
                        raw_payloads={"train_health_probe": {"shapes": shapes if cfg.logging.get("log_tensor_shapes", True) else {}}},
                        metadata={"trigger": "first_train_probe"},
                    )
                except Exception:
                    logger.exception("Failed running post-first-step memory probe")
                finally:
                    first_train_probe_done = True

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{jepa_loss.item():.4f}",
                    "vc": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                }
            )

            global_step += 1

            if diagnostics_enabled and fast_every_steps is not None and global_step > 0 and global_step % fast_every_steps == 0:
                fast_logs = _run_diagnostics_event(
                    event_type="fast_diagnostics",
                    probe_mode=fast_mode,
                    step=global_step,
                    epoch_value=epoch,
                    metrics_prefix="fast/",
                    enable_geometry=False,
                    persist=True,
                    event_metadata={"trigger": "step"},
                )
                _post_diagnostics_cleanup()
                if wandb_run:
                    import wandb

                    wandb.log(fast_logs, step=global_step)

        # Validation and diagnostics
        # Validation and diagnostics
        should_run_canonical = diagnostics_enabled and epoch > 0 and epoch % canonical_every_epochs == 0
        should_run_fast_epoch = diagnostics_enabled and fast_every_epochs is not None and epoch > 0 and epoch % fast_every_epochs == 0 and not should_run_canonical
        should_run_fast_step = diagnostics_enabled and fast_every_steps is not None and global_step > 0 and global_step % fast_every_steps == 0 and not should_run_canonical
        
        val_logs = {}
        if should_run_canonical:
            val_logs = _run_diagnostics_event(
                event_type="canonical_diagnostics",
                probe_mode=canonical_mode,
                step=global_step,
                epoch_value=epoch,
                metrics_prefix="",
                enable_geometry=True,
                persist=True,
                event_metadata={"trigger": "checkpoint_save"},
            )
            _post_diagnostics_cleanup()
        elif should_run_fast_epoch:
            val_logs = _run_diagnostics_event(
                event_type="fast_diagnostics",
                probe_mode=fast_mode,
                step=global_step,
                epoch_value=epoch,
                metrics_prefix="",
                enable_geometry=False,
                persist=False,
                event_metadata={"trigger": "fast_diagnostics_every_epochs"},
            )
            _post_diagnostics_cleanup()
        elif should_run_fast_step:
            val_logs = _run_diagnostics_event(
                event_type="fast_diagnostics",
                probe_mode=fast_mode,
                step=global_step,
                epoch_value=epoch,
                metrics_prefix="",
                enable_geometry=False,
                persist=False,
                event_metadata={"trigger": "fast_diagnostics_every_steps"},
            )
            _post_diagnostics_cleanup()

        if should_run_canonical or should_run_fast_epoch or should_run_fast_step:
            # One-time post-first-validation memory probe (simplified)
            if not first_val_probe_done:
                try:
                    shapes = {}
                    _log_memory_snapshot(global_step, probe_modules, wandb_run=wandb_run, prefix="val")
                    if cfg.logging.get("log_tensor_shapes", True):
                        # Safely attempt to fetch a single validation sample without nested try/excepts
                        sample_batch = next(iter(val_loader), None)
                        if sample_batch is not None:
                            sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
                            x_val = sample_batch.get("video")
                        else:
                            x_val = None

                        if x_val is not None:
                            shapes = _capture_shapes(
                                x_val,
                                encoder,
                                projector,
                                device,
                                cfg.logging.get("projector_force_runtime_shapes", False),
                            )
                        else:
                            shapes = {"raw_input": None, "encoder_output": None, "projector_output": None}

                        _log_tensor_shapes(global_step, shapes, wandb_run=wandb_run, prefix="train")
                    shapes_metrics = {f"train/shape/{k}": v for k, v in shapes.items()} if cfg.logging.get("log_tensor_shapes", True) else {}
                    diagnostics_manager.record_event(
                        event_type="train_health_probe",
                        phase="train",
                        step=global_step,
                        epoch=epoch,
                        metrics={
                            **_collect_memory_snapshot_metrics(global_step, probe_modules, prefix="train", dtype=dtype),
                            **shapes_metrics,
                        },
                        raw_payloads={"train_health_probe": {"shapes": shapes if cfg.logging.get("log_tensor_shapes", True) else {}}},
                        metadata={"trigger": "first_train_probe"},
                    )
                except Exception:
                    logger.exception("Failed running post-first-validation memory probe")
                finally:
                    first_val_probe_done = True

            train_metrics = {
                "epoch": epoch,
                "train/loss": jepa_loss.item(),
                "train/vc_loss": regl.item(),
                "train/pred_loss": pl.item(),
                "train/recon_loss": recon_loss.item(),
                "train/det_loss": det_loss.item(),
            }
            for k, v in regldict.items():
                train_metrics[f"train/{k}"] = float(v)

            all_metrics = {**train_metrics, **val_logs}

            if wandb_run:
                import wandb

                wandb.log(all_metrics, step=global_step)

            log_epoch(
                epoch,
                {
                    "loss": jepa_loss.item(),
                    "vc": regl.item(),
                    "pred": pl.item(),
                    "val_recon": val_logs.get("val/recon_loss", 0),
                },
                total_epochs=cfg.optim.epochs,
            )

        # Save checkpoint
        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model=jepa,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            steps_per_epoch=steps_per_epoch,
            pixel_decoder_state_dict=pixel_decoder.state_dict(),
            detection_head_state_dict=detection_head.state_dict(),
        )
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=jepa,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                steps_per_epoch=steps_per_epoch,
                pixel_decoder_state_dict=pixel_decoder.state_dict(),
                detection_head_state_dict=detection_head.state_dict(),
            )

    geometry_enabled = bool(geometry_cfg.get("enabled", False))
    if geometry_enabled and wandb_run:
        try:
            import wandb

            evo_logs = assemble_geometry_viz_videos(
                exp_dir=exp_dir,
                fps=int(geometry_cfg.get("evolution_fps", 2)),
                wandb_prefix="geometry_viz",
            )
            if evo_logs:
                wandb.log(evo_logs, step=global_step)
        except Exception:
            logger.exception("Failed assembling/logging geometry evolution videos")

    if wandb_run:
        diagnostics_manager.close()
        import wandb

        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    fire.Fire(run)
