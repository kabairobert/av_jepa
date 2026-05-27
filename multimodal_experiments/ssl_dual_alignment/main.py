import os
from pathlib import Path

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
    get_exp_name,
    get_unified_experiment_dir,
    save_config,
    save_checkpoint,
    load_checkpoint
)
from multimodal_experiments.ssl_dual_alignment.dataset import DualDisentangleDataset
from multimodal_experiments.ssl_dual_alignment.model_builder import build_model_and_predictors
from multimodal_experiments.ssl_dual_alignment.losses import EBMJEPALoss
from multimodal_experiments.initial_trials.ssl_disentangling import SupervisedFactorLoss
from multimodal_experiments.ssl_dual_alignment.eval import evaluate_and_log_checkpoint
from multimodal_experiments.ssl_dual_alignment.vis import log_plots_to_wandb


def _parse_tags(wandb_tags):
    """Normalize wandb_tags: Fire may pass str, tuple, or list."""
    if not wandb_tags:
        return []
    if isinstance(wandb_tags, (list, tuple)):
        return [str(t).strip() for t in wandb_tags if str(t).strip()]
    return [t.strip() for t in str(wandb_tags).split(",") if t.strip()]


def _save_optimizer_only(path: Path, optimizer, epoch: int, step: int) -> None:
    """Save only optimizer state (no model) for Stage-2 predictor checkpoints.

    save_checkpoint() requires a non-None model; use this helper when only the
    optimizer state needs to be persisted (e.g. predictor optimizer in two-stage).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "step": step, "optimizer_state_dict": optimizer.state_dict()},
        path,
    )


def run(
    fname: str = "multimodal_experiments/ssl_dual_alignment/cfgs/paired_factors_2D.yaml",
    config: str = None,
    cfg=None,
    folder=None,
    wandb_tags=None,
    quickrun: bool = False,
    **overrides
):
    # --config is an alias for --fname (used by sweep.py)
    if config is not None:
        fname = config

    # --- 1. Config & Env ---
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Apply quickrun shortcut
    if quickrun:
        cfg.optim.epochs = 1
        if not hasattr(cfg, "training"):
            from omegaconf import OmegaConf
            cfg.training = OmegaConf.create({})
        cfg.training.max_train_batches = 1
        cfg.logging.log_wandb = False

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    torch.set_default_dtype(torch.float32)

    # --- 2. Exp Dir Setup ---
    two_stage = cfg.training.get('two_stage', False) if hasattr(cfg, 'training') else False
    loss_type = cfg.loss.get("type", "ebm")

    data_type_str = str(cfg.data.get('type', '2d')).replace('3d-2f-common', '3D2F').replace('3d-av-1f-common', '3D1F')
    pred_type_raw = str(cfg.model.get('predictor_type', 'none'))
    pred_type_str = "aff" if pred_type_raw == "affine" else pred_type_raw

    cm_val = str(cfg.loss.get("congruence_mode", cfg.loss.get("noise_reweighting", "none")))
    if cm_val in ("none", "off"):
        canon_cm = "none"
        cm_str = "off"
    elif cm_val in ("pred_only", "pred"):
        canon_cm = "pred_only"
        cm_str = "pred"
    elif cm_val in ("pred_and_sparse", "pred_sparse", "full"):
        canon_cm = "pred_and_sparse"
        cm_str = "pred_sparse"
    else:
        canon_cm = cm_val
        cm_str = cm_val

    # EBM-only naming tags are only meaningful when loss.type == "ebm"
    if loss_type == "ebm":
        pred_loss_str = str(cfg.loss.get('pred_loss', 'l1'))
        if cfg.loss.get('lambda_pred', 1.0) == 0.0:
            pred_loss_str = "none"

        ds_dims = ""
        if data_type_str == "nd-kf-mlp":
            ds_dims = f"-k{cfg.data.get('k_shared')}_m{cfg.data.get('m_unique')}_d{cfg.data.get('d_out')}"

        asy_a = cfg.data.get('asymmetric_noise_rate_a', 0.0)
        ext = cfg.data.get('external_noise_ratio', 0.0)
        noise_str = f"-nz_a{asy_a}_e{ext}"

        exp_name = (
            f"sslda-{data_type_str}{ds_dims}-"
            f"l_{loss_type}-"
            f"pre_{pred_type_str}_{pred_loss_str}-"
            f"pri_{cfg.loss.get('prior_type', 'l1')}-"
            f"cm_{cm_str}-"
            f"sp_{cfg.loss.get('lambda_sparse', 0.0)}"
            f"{noise_str}-"
            f"{'2stg' if two_stage else '1stg'}"
        )
    else:
        ds_dims = ""
        if data_type_str == "nd-kf-mlp":
            ds_dims = f"-k{cfg.data.get('k_shared')}_m{cfg.data.get('m_unique')}_d{cfg.data.get('d_out')}"
        exp_name = (
            f"sslda-{data_type_str}{ds_dims}-"
            f"l_{loss_type}-"
            f"pre_{pred_type_str}"
        )

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

    # --- 4. Dataset ---
    train_set = DualDisentangleDataset(
        data_type=cfg.data.get('type', '2d'),
        num_samples=cfg.data.get('num_samples', 4096),
        path_a=cfg.data.get('path_a', None),
        path_b=cfg.data.get('path_b', None),
        manifold_noise_a=cfg.data.get('manifold_noise_a', None),
        manifold_noise_b=cfg.data.get('manifold_noise_b', None),
        asymmetric_noise_magnitude=cfg.data.get('asymmetric_noise_magnitude', None),
        asymmetric_noise_rate_a=cfg.data.get('asymmetric_noise_rate_a', None),
        asymmetric_noise_rate_b=cfg.data.get('asymmetric_noise_rate_b', None),
        external_noise_ratio=cfg.data.get('external_noise_ratio', None),
        noise_bbox_expansion=cfg.data.get('noise_bbox_expansion', 0.0),
        seed=cfg.meta.seed,
        # nd-kf-mlp dataset shape params (ignored for other data_types)
        k_shared=cfg.data.get('k_shared', 2),
        m_unique=cfg.data.get('m_unique', 2),
        d_out=cfg.data.get('d_out', 16),
        u3a_scale=cfg.data.get('u3a_scale', 0.2),
        u3b_scale=cfg.data.get('u3b_scale', 0.3),
        turns=cfg.data.get('turns', 1.0),
        wave_amplitude=cfg.data.get('wave_amplitude', 1.0),
        embed_dim=cfg.data.get('embed_dim', None),
    )

    # Move entire dataset to GPU for significant training speedup (if using CUDA)
    train_set.to(device)
    
    # Use 0 workers for GPU-resident datasets (multiprocessing with CUDA tensors is complex/slow)
    num_workers = 0 if device.type == 'cuda' else cfg.data.get('num_workers', 0)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.get('batch_size', 128),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Redundant for GPU-resident data
    )

    # --- 5. Model Init ---
    built = build_model_and_predictors(cfg, device)
    full_model = built["full_model"]
    dual_model = built["dual_model"]
    predictor_a2b = built["predictor_a2b"]
    predictor_b2a = built["predictor_b2a"]

    # --- 6. Loss ---
    if loss_type == "ebm":
        loss_fn = EBMJEPALoss(
            predictor_a2b,
            predictor_b2a,
            lambda_jac=cfg.loss.get("lambda_jac", 1.0),
            lambda_prior=cfg.loss.get("lambda_prior", 0.5),
            lambda_pred=cfg.loss.get("lambda_pred", 1.0),
            lambda_sparse=cfg.loss.get("lambda_sparse", 0.1),
            prior_type=cfg.loss.get("prior_type", 'l1'),
            pred_loss=cfg.loss.get("pred_loss", 'l1'),
            congruence_mode=canon_cm,
            congruence_tau=cfg.loss.get("congruence_tau", cfg.loss.get("reweighting_tau", 0.5)),
        )
    else:
        loss_fn = SupervisedFactorLoss(
            dimensions_per_factor=[1, 1] if cfg.data.get('type', '2d') == '2d' else [1, 1, 1]
        )

    # --- 7. Optimizers (one-stage or two-stage) ---
    start_epoch = 0
    global_step = 0
    if two_stage:
        stage1_epochs = cfg.training.get('stage1_epochs', 150)
        stage2_epochs = cfg.training.get('stage2_epochs', 150)
        epochs = stage1_epochs + stage2_epochs
        opt_flow = torch.optim.Adam(dual_model.parameters(), lr=cfg.optim.get("lr", 0.001))
        pred_params = []
        if predictor_a2b is not None:
            pred_params += list(predictor_a2b.parameters()) + list(predictor_b2a.parameters())
        opt_pred = torch.optim.Adam(pred_params, lr=cfg.optim.get("lr", 0.001)) if pred_params else None
    else:
        epochs = cfg.optim.get("epochs", 500)
        all_params = list(dual_model.parameters())
        if predictor_a2b is not None:
            all_params += list(predictor_a2b.parameters()) + list(predictor_b2a.parameters())
        optimizer = torch.optim.Adam(all_params, lr=cfg.optim.get("lr", 0.001))

    # --- 8. Resume Checkpoint ---
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        if not two_stage:
            ckpt_info = load_checkpoint(ckpt_path, full_model, optimizer, device=device)
        else:
            ckpt_info = load_checkpoint(ckpt_path, full_model, opt_flow, device=device)
            # If resuming into Stage 2, also restore opt_pred state if a separate
            # pred_optimizer checkpoint is available alongside the main checkpoint.
            # load_checkpoint returns epoch+1, so after saving at stage1_epochs-1 we
            # get back stage1_epochs here — the condition is therefore exact (no off-by-one).
            start_epoch_probe = ckpt_info.get("epoch", 0)
            if start_epoch_probe >= stage1_epochs and opt_pred is not None:
                pred_ckpt = Path(str(ckpt_path).replace(".pth.tar", "_pred.pth.tar"))
                if pred_ckpt.exists():
                    raw = torch.load(pred_ckpt, map_location=device)
                    if "optimizer_state_dict" in raw:
                        opt_pred.load_state_dict(raw["optimizer_state_dict"])
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step", 0)

    print(f"Starting training for {epochs} epochs"
          f"{' (two-stage: '+str(stage1_epochs)+'+'+str(stage2_epochs)+')' if two_stage else ''}...")
    if wandb_run:
        log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)

    # --- 9. Training Loop ---
    max_train_batches = cfg.training.get("max_train_batches") if hasattr(cfg, "training") else None

    def _train_epoch(epoch_idx, active_optimizer, stage=None):
        """Single training epoch, returns (avg_loss, avg_align_a2b, avg_align_b2a).

        stage: 1 = flow-only (Stage 1), 2 = predictor-only (Stage 2), None = joint.
        In Stage 1, predictor .grad is zeroed after backward to prevent silent accumulation.
        """
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
            data_a = batch["data_a"].to(device)
            data_b = batch["data_b"].to(device)
            corr_target = batch["corr_target"].to(device)
            active_optimizer.zero_grad()
            outputs = dual_model(data_a, data_b)
            loss = loss_fn(outputs) if loss_type == "ebm" else loss_fn(corr_target, outputs)
            loss.backward()
            active_optimizer.step()
            # FIX #8: In Stage 1, predictor params receive gradients (from pred loss path)
            # but are not in opt_flow. Zero them explicitly to prevent stale grad accumulation.
            if stage == 1 and predictor_a2b is not None:
                for p in list(predictor_a2b.parameters()) + list(predictor_b2a.parameters()):
                    if p.grad is not None:
                        p.grad.zero_()
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

    def _maybe_eval_and_save(epoch_idx, active_optimizer, current_stage=None):
        save_checkpoint(
            exp_dir / f"epoch_{epoch_idx+1}.pth.tar",
            model=full_model,
            optimizer=active_optimizer,
            epoch=epoch_idx,
            step=global_step,
            axis_box=getattr(train_set, 'axis_box', None),
        )
        if wandb_run:
            import wandb
            # FIX Bug1: evaluate_and_log_checkpoint has no extra_logs param.
            # Log training_stage directly before the eval call instead.
            if current_stage is not None:
                wandb.log({"eval/training_stage": current_stage}, step=global_step)
            evaluate_and_log_checkpoint(
                train_set, train_loader, dual_model, loss_fn, loss_type,
                {"a2b": predictor_a2b, "b2a": predictor_b2a},
                device, global_step, wandb_run,
                checkpoint_name=f"epoch_{epoch_idx+1}.pth.tar",
                checkpoint_path=str(exp_dir / f"epoch_{epoch_idx+1}.pth.tar"),
                log_interactive_3d=str(cfg.data.get("type", "2d")).startswith("3d"),
                log_prefix="val",
                is_3d=str(cfg.data.get("type", "2d")).startswith("3d"),
            )

    save_every = cfg.logging.get("save_every", 50)
    if two_stage:
        # Stage 1: train flows only
        print(f"=== Stage 1: flow training ({stage1_epochs} epochs) ===")

        # --- TEMPORARY CHANGE FOR STAGE 1 ---
        # Set lambda_pred to 0 to ensure encoders only learn from Jacobian/Prior.
        # This prevents the cross-modal prediction task from influencing encoders in Stage 1.
        original_lambda_pred = getattr(loss_fn, 'lambda_pred', 1.0)
        if hasattr(loss_fn, 'lambda_pred'):
            loss_fn.lambda_pred = 0.0

        for epoch_idx in range(start_epoch, stage1_epochs):
            avg_loss, avg_a2b, avg_b2a = _train_epoch(epoch_idx, opt_flow, stage=1)
            if wandb_run:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/align_mse_a2b": avg_a2b,
                    "train/align_mse_b2a": avg_b2a,
                    "train/stage": 1,
                }, step=global_step)
            if (epoch_idx + 1) % save_every == 0:
                _maybe_eval_and_save(epoch_idx, opt_flow, current_stage=1)

        # --- RESTORE CHANGE FOR STAGE 2 ---
        if hasattr(loss_fn, 'lambda_pred'):
            loss_fn.lambda_pred = original_lambda_pred
            print(f"Stage 1 complete. Restored loss_fn.lambda_pred to {original_lambda_pred}")

        # Freeze flows, unfreeze predictors for Stage 2
        dual_model.requires_grad_(False)
        if predictor_a2b is not None:
            predictor_a2b.requires_grad_(True)
            predictor_b2a.requires_grad_(True)

        # Stage 2: train predictors only
        print(f"=== Stage 2: predictor training ({stage2_epochs} epochs) ===")
        if opt_pred is None:
            print("WARNING: no predictor params — Stage 2 is a no-op")
        for epoch_idx in range(stage1_epochs, epochs):
            if opt_pred is not None:
                avg_loss, avg_a2b, avg_b2a = _train_epoch(epoch_idx, opt_pred, stage=2)
            else:
                avg_loss, avg_a2b, avg_b2a = 0.0, 0.0, 0.0
            if wandb_run:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/align_mse_a2b": avg_a2b,
                    "train/align_mse_b2a": avg_b2a,
                    "train/stage": 2,
                }, step=global_step)
            if (epoch_idx + 1) % save_every == 0:
                _maybe_eval_and_save(epoch_idx, opt_pred or opt_flow, current_stage=2)
                # FIX Bug2: save_checkpoint requires a non-None model.
                # Persist only opt_pred state with the dedicated helper.
                if opt_pred is not None:
                    _save_optimizer_only(
                        exp_dir / f"epoch_{epoch_idx+1}_pred.pth.tar",
                        opt_pred, epoch=epoch_idx, step=global_step,
                    )

        # Restore flow grad for final save
        dual_model.requires_grad_(True)
        final_optimizer = opt_pred if opt_pred is not None else opt_flow

    else:
        for epoch_idx in range(start_epoch, epochs):
            avg_loss, avg_a2b, avg_b2a = _train_epoch(epoch_idx, optimizer)
            if wandb_run:
                import wandb
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
        import wandb
        evaluate_and_log_checkpoint(
            train_set, train_loader, dual_model, loss_fn, loss_type,
            {"a2b": predictor_a2b, "b2a": predictor_b2a},
            device, global_step, wandb_run,
            checkpoint_name="latest.pth.tar",
            checkpoint_path=str(exp_dir / "latest.pth.tar"),
            log_interactive_3d=str(cfg.data.get("type", "2d")).startswith("3d"),
            log_prefix="val",
            is_3d=str(cfg.data.get("type", "2d")).startswith("3d"),
        )
        log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    fire.Fire(run)
