import sys
from pathlib import Path
# Support running from subdirectories without PYTHONPATH overrides
sys.path.append(str(Path(__file__).resolve().parents[2]))

from datetime import datetime

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.training_utils import (
    load_config,
    setup_device,
    setup_seed,
    setup_wandb,
    get_default_dev_name,
    get_unified_experiment_dir,
    save_config,
    save_checkpoint,
    load_checkpoint
)
from multimodal_experiments.ssl_dual_alignment.dataset import build_dataset_from_config
from multimodal_experiments.ssl_dual_alignment.model_builder import build_model_and_predictors
from multimodal_experiments.ssl_dual_alignment.losses import build_loss_from_config
from multimodal_experiments.ssl_dual_alignment.eval import evaluate_and_log_checkpoint
from multimodal_experiments.ssl_dual_alignment.vis import log_plots_to_wandb


def _parse_tags(wandb_tags):
    """Normalize wandb_tags: Fire may pass str, tuple, or list."""
    if not wandb_tags:
        return []
    if isinstance(wandb_tags, (list, tuple)):
        return [str(t).strip() for t in wandb_tags if str(t).strip()]
    return [t.strip() for t in str(wandb_tags).split(",") if t.strip()]


def _estimate_vram_footprint(cfg, device) -> None:
    """Estimate GPU VRAM footprint and print warning if usage is predicted high."""
    if device.type != 'cuda':
        return
    try:
        total_mem = torch.cuda.get_device_properties(device).total_memory
        s_count = cfg.model.get('stage_count', 6)
        n_dims = cfg.model.get('num_dims', 2)
        h_units = cfg.model.get('hidden_units', 128)
        b_size = cfg.data.get('batch_size', 128)
        
        # Estimate activation memory in GB (4 coupling layers per stage, float32)
        est_act_mem_gb = (4 * s_count * b_size * (3 * n_dims + 2 * h_units) * 4) / 1024**3
        # Estimate compile overhead for deep models
        compile_overhead = 8.0 if (hasattr(torch, "compile") and s_count >= 12 and n_dims >= 256) else 0.0
        
        total_needed_gb = est_act_mem_gb + compile_overhead
        total_avail_gb = total_mem / 1024**3
        
        print(
            f"[INFO] GPU VRAM footprint estimates:\n"
            f"  Estimated Activation Memory: {est_act_mem_gb:.2f} GB\n"
            f"  Estimated Compile Overhead: {compile_overhead:.2f} GB\n"
            f"  Total Estimated GPU VRAM Required: {total_needed_gb:.2f} GB\n"
            f"  GPU total capacity: {total_avail_gb:.2f} GB"
        )
        
        if total_needed_gb > 0.50 * total_avail_gb:
            print(
                f"⚠️ WARNING: High GPU VRAM usage predicted! (> 50% capacity).\n"
                f"  If you experience OOM, consider reducing batch_size, stage_count, or disabling torch.compile."
            )
    except Exception:
        pass


def run(
    fname: str = "multimodal_experiments/ssl_dual_alignment/cfgs/paired_factors_2D.yaml",
    config: str = None,
    cfg=None,
    folder=None,
    wandb_tags=None,
    quickrun=False,
    **overrides
):
    import wandb
    # --config is an alias for --fname (used by sweep.py)
    if config is not None:
        fname = config

    # --- 1. Config & Env ---
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Apply quickrun shortcut
    if quickrun:
        valid_opts = ["cpu-nolog", "cpu-log", "gpu-nolog", "gpu-log"]
        
        if isinstance(quickrun, bool) and quickrun is True:
            quickrun_opt = "cpu-nolog"
        else:
            quickrun_opt = str(quickrun).strip().lower()
            
        if quickrun_opt not in valid_opts:
            print(f"❌ Invalid quickrun option: '{quickrun}'. Valid options: {', '.join(valid_opts)}")
            import sys
            sys.exit(1)
            
        cfg.optim.epochs = 1
        if not hasattr(cfg, "training"):
            from omegaconf import OmegaConf
            cfg.training = OmegaConf.create({})
        cfg.training.max_train_batches = 1
        
        if "nolog" in quickrun_opt:
            cfg.logging.log_wandb = False
            
        if "cpu" in quickrun_opt:
            cfg.meta.device = "cpu"
            cfg.data.batch_size = 4

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    torch.set_default_dtype(torch.float32)

    # --- 2. Exp Dir Setup ---
    two_stage_legacy = cfg.training.get('two_stage', False) if hasattr(cfg, 'training') else False
    if two_stage_legacy:
        print("⚠️ WARNING: Config specified two_stage=True, but two-stage training has been deprecated. Running in standard one-stage joint mode instead.")
    loss_type = cfg.loss.get("type", "ebm")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    exp_name = f"{Path(fname).stem}_{timestamp}"

    if folder is None:
        sweep_name = get_default_dev_name()
        exp_dir = get_unified_experiment_dir(
            example_name="sslda",
            sweep_name=sweep_name,
            exp_name=exp_name,
            seed=cfg.meta.seed,
            base_dir=cfg.meta.get("checkpoint_dir", None),
        )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_name = exp_dir.name.rsplit("_seed", 1)[0]
    save_config(cfg, exp_dir)

    # --- 3. W&B Logging ---
    base_tags = ["sslda"]
    if cfg.logging.get("log_seed_tag", False):
        base_tags.append(f"seed_{cfg.meta.seed}")
    extra_tags = _parse_tags(wandb_tags)
    all_tags = base_tags + extra_tags

    wandb_run = setup_wandb(
        project="eb_jepa",
        config=cfg,
        run_dir=exp_dir,
        run_name=exp_name,
        tags=all_tags,
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
    )

    train_set = build_dataset_from_config(cfg)

    # Keep dataset on CPU to save VRAM and avoid OOM on large configurations
    # train_set.to(device)
    
    # Use config value or default to 0
    num_workers = cfg.data.get('num_workers', 0)
    pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.get('batch_size', 128),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # --- 5. Model Init ---
    built = build_model_and_predictors(cfg, device)
    full_model = built["full_model"]
    dual_model = built["dual_model"]
    predictor_a2b = built["predictor_a2b"]
    predictor_b2a = built["predictor_b2a"]

    # VRAM safety check
    _estimate_vram_footprint(cfg, device)

    if hasattr(torch, "compile") and device.type == 'cuda':
        print("Compiling model with torch.compile...")
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
        except Exception:
            pass
        dual_model = torch.compile(dual_model)

    loss_fn = build_loss_from_config(cfg, predictor_a2b, predictor_b2a)

    # --- 7. Optimizers ---
    start_epoch = 0
    global_step = 0
    epochs = cfg.optim.get("epochs", 500)
    all_params = list(dual_model.parameters())
    if predictor_a2b is not None:
        all_params += list(predictor_a2b.parameters()) + list(predictor_b2a.parameters())
    optimizer = torch.optim.Adam(all_params, lr=cfg.optim.get("lr", 0.001))

    # --- 8. Resume Checkpoint ---
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path, full_model, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step", 0)

    print(f"Starting training for {epochs} epochs...")
    if wandb_run:
        log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)

    # Mixed precision setup matching video_jepa's elegant and robust pattern
    train_cfg = cfg.get("training", {})
    if train_cfg is None:
        train_cfg = {}
    
    use_amp = train_cfg.get("use_amp", True)
    dtype_str = str(train_cfg.get("dtype", "bfloat16")).lower()
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    if use_amp and device.type != 'cuda':
        print("AMP requested but device is CPU — disabling AMP for safety")
        use_amp = False
        
    scaler_device = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_amp and dtype == torch.float16)
    print(f"Using AMP: {use_amp} with dtype: {dtype}")

    # --- 9. Training Loop ---
    max_train_batches = cfg.training.get("max_train_batches") if hasattr(cfg, "training") else None

    def _train_epoch(epoch_idx, active_optimizer):
        """Single training epoch, returns (avg_loss, avg_align_a2b, avg_align_b2a)."""
        nonlocal global_step
        dual_model.train()
        epoch_loss = 0.0
        epoch_align_a2b = 0.0
        epoch_align_b2a = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch_idx+1}/{epochs}",
            disable=cfg.logging.get("tqdm_silent", False)
        )
        for batch_idx, batch in enumerate(pbar, start=1):
            data_a = batch["data_a"].to(device, non_blocking=True)
            data_b = batch["data_b"].to(device, non_blocking=True)
            corr_target = batch["corr_target"].to(device, non_blocking=True)
            active_optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                outputs = dual_model(data_a, data_b)
                loss = loss_fn(outputs) if loss_type == "ebm" else loss_fn(corr_target, outputs)
            scaler.scale(loss).backward()
            scaler.step(active_optimizer)
            scaler.update()
            # Calculate alignment metrics under no_grad
            d = (outputs.shape[1] - 2) // 2
            z_a, z_b = outputs[:, :d], outputs[:, d:2*d]
            with torch.no_grad():
                # Only log MSE alignment if prediction task is actually enabled
                has_pred = getattr(loss_fn, 'lambda_pred', 1.0) > 0
                if has_pred and predictor_a2b is not None and predictor_b2a is not None:
                    epoch_align_a2b += torch.nn.functional.mse_loss(predictor_a2b(z_a), z_b).item()
                    epoch_align_b2a += torch.nn.functional.mse_loss(predictor_b2a(z_b), z_a).item()
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
        nb = batch_idx
        return epoch_loss / nb, epoch_align_a2b / nb, epoch_align_b2a / nb

    def _maybe_eval_and_save(epoch_idx, active_optimizer):
        save_checkpoint(
            exp_dir / f"epoch_{epoch_idx+1}.pth.tar",
            model=full_model,
            optimizer=active_optimizer,
            epoch=epoch_idx,
            step=global_step,
            axis_box=getattr(train_set, 'axis_box', None),
        )
        if wandb_run:
            evaluate_and_log_checkpoint(
                train_set, train_loader, dual_model, loss_fn, loss_type,
                {"a2b": predictor_a2b, "b2a": predictor_b2a},
                device, global_step, wandb_run,
                checkpoint_name=f"epoch_{epoch_idx+1}.pth.tar",
                checkpoint_path=str(exp_dir / f"epoch_{epoch_idx+1}.pth.tar"),
                log_interactive_3d=bool(train_set.data_a.shape[1] >= 3),
                log_prefix="val",
                is_3d=bool(train_set.data_a.shape[1] >= 3),
            )

    save_every = cfg.logging.get("save_every", 50)
    for epoch_idx in range(start_epoch, epochs):
        avg_loss, avg_a2b, avg_b2a = _train_epoch(epoch_idx, optimizer)
        if wandb_run:
            wandb.log({
                "train/loss": avg_loss,
                "train/align_mse_a2b": avg_a2b,
                "train/align_mse_b2a": avg_b2a,
            }, step=global_step)
        if (epoch_idx + 1) % save_every == 0:
            _maybe_eval_and_save(epoch_idx, optimizer)
    final_optimizer = optimizer

    # --- 10. Final checkpoint ---
    save_checkpoint(
        exp_dir / "latest.pth.tar",
        model=full_model,
        optimizer=final_optimizer,
        epoch=epochs - 1,
        step=global_step,
        axis_box=getattr(train_set, 'axis_box', None),
    )

    if wandb_run:
        evaluate_and_log_checkpoint(
            train_set, train_loader, dual_model, loss_fn, loss_type,
            {"a2b": predictor_a2b, "b2a": predictor_b2a},
            device, global_step, wandb_run,
            checkpoint_name="latest.pth.tar",
            checkpoint_path=str(exp_dir / "latest.pth.tar"),
            log_interactive_3d=bool(train_set.data_a.shape[1] >= 3),
            log_prefix="val",
            is_3d=bool(train_set.data_a.shape[1] >= 3),
        )
        log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    fire.Fire(run)
