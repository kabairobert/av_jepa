import os
from pathlib import Path
import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.training_utils import (
    load_config, setup_device, setup_seed, setup_wandb,
    get_default_dev_name, get_exp_name, get_unified_experiment_dir,
    save_config, save_checkpoint, load_checkpoint
)
from multimodal_experiments.ssl_dual_alignment.dataset import DualDisentangleDataset
from multimodal_experiments.ssl_dual_alignment.model_builder import build_model_and_predictors
from multimodal_experiments.ssl_dual_alignment.losses import EBMJEPALoss
from multimodal_experiments.initial_trials.ssl_disentangling import SupervisedFactorLoss
from multimodal_experiments.ssl_dual_alignment.eval import evaluate_and_log_checkpoint
from multimodal_experiments.ssl_dual_alignment.vis import log_plots_to_wandb


def run(fname: str = "multimodal_experiments/ssl_dual_alignment/cfgs/paired_factors_2D.yaml", cfg=None, folder=None, **overrides):
    # --- 1. Config & Env ---
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    torch.set_default_dtype(torch.float32)

    # --- 2. Exp Dir Setup ---
    two_stage = cfg.training.get('two_stage', False) if hasattr(cfg, 'training') else False
    exp_name = (
        f"dalign_{cfg.data.get('type', '2d')}_"
        f"pred_{cfg.model.get('predictor_type', 'none')}_"
        f"prior_{cfg.loss.get('prior_type', 'l1')}_"
        f"pred_loss_{cfg.loss.get('pred_loss', 'l1')}_"
        f"rw_{cfg.loss.get('noise_reweighting', 'none')}_"
        f"sparse_{cfg.loss.get('lambda_sparse', 0.0)}_"
        f"{'2stage' if two_stage else '1stage'}"
    )
    if folder is None:
        sweep_name = get_default_dev_name()
        exp_dir = get_unified_experiment_dir(
            example_name="dual_disentangle",
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
    wandb_run = setup_wandb(
        project="eb_jepa",
        config=cfg,
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["dual_disentangle", f"seed_{cfg.meta.seed}", "multimodal_initial"],
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
        seed=cfg.meta.seed,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.data.get('batch_size', 128), shuffle=True, num_workers=cfg.data.get('num_workers', 0))

    # --- 5. Model Init ---
    built = build_model_and_predictors(cfg, device)
    dual_model = built["dual_model"]
    predictor_a2b = built["predictor_a2b"]
    predictor_b2a = built["predictor_b2a"]

    # --- 6. Loss ---
    loss_type = cfg.loss.get("type", "ebm")
    if loss_type == "ebm":
        loss_fn = EBMJEPALoss(
            predictor_a2b, predictor_b2a,
            lambda_jac=cfg.loss.get("lambda_jac", 1.0),
            lambda_prior=cfg.loss.get("lambda_prior", 0.5),
            lambda_sparse=cfg.loss.get("lambda_sparse", 0.1),
            prior_type=cfg.loss.get("prior_type", 'l1'),
            pred_loss=cfg.loss.get("pred_loss", 'l1'),
            noise_reweighting=cfg.loss.get("noise_reweighting", 'none'),
            reweighting_tau=cfg.loss.get("reweighting_tau", 0.5),
        )
    else:
        loss_fn = SupervisedFactorLoss(dimensions_per_factor=[1, 1] if cfg.data.get('type', '2d') == '2d' else [1, 1, 1])

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
            ckpt_info = load_checkpoint(ckpt_path, dual_model, optimizer, device=device)
        else:
            ckpt_info = load_checkpoint(ckpt_path, dual_model, opt_flow, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step", 0)

    print(f"Starting training for {epochs} epochs{'  (two-stage: '+str(stage1_epochs)+'+'+str(stage2_epochs)+')' if two_stage else ''}...")

    if wandb_run:
        log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)

    # --- 9. Training Loop ---
    def _train_epoch(epoch_idx, active_optimizer):
        """Single training epoch, returns (avg_loss, avg_align_a2b, avg_align_b2a)."""
        nonlocal global_step
        dual_model.train()
        epoch_loss = 0.0
        epoch_align_a2b = 0.0
        epoch_align_b2a = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{epochs}", disable=cfg.logging.get("tqdm_silent", False))
        for batch in pbar:
            data_a = batch["data_a"].to(device)
            data_b = batch["data_b"].to(device)
            corr_target = batch["corr_target"].to(device)

            active_optimizer.zero_grad()
            outputs = dual_model(data_a, data_b)
            loss = loss_fn(outputs) if loss_type == "ebm" else loss_fn(corr_target, outputs)
            loss.backward()
            active_optimizer.step()

            d = (outputs.shape[1] - 2) // 2
            z_a, z_b = outputs[:, :d], outputs[:, d:2*d]
            with torch.no_grad():
                if predictor_a2b is not None and predictor_b2a is not None:
                    epoch_align_a2b += torch.nn.functional.mse_loss(predictor_a2b(z_a), z_b).item()
                    epoch_align_b2a += torch.nn.functional.mse_loss(predictor_b2a(z_b), z_a).item()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        nb = len(train_loader)
        return epoch_loss / nb, epoch_align_a2b / nb, epoch_align_b2a / nb

    def _maybe_eval_and_save(epoch_idx, active_optimizer):
        save_checkpoint(
            exp_dir / f"epoch_{epoch_idx+1}.pth.tar",
            model=dual_model, optimizer=active_optimizer,
            epoch=epoch_idx, step=global_step,
            axis_box=getattr(train_set, 'axis_box', None),
        )
        if wandb_run:
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
        for epoch_idx in range(start_epoch, stage1_epochs):
            avg_loss, avg_a2b, avg_b2a = _train_epoch(epoch_idx, opt_flow)
            if wandb_run:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/align_mse_a2b": avg_a2b,
                           "train/align_mse_b2a": avg_b2a, "train/stage": 1,
                           "epoch": epoch_idx+1, "step": global_step}, step=global_step)
            if (epoch_idx + 1) % save_every == 0:
                _maybe_eval_and_save(epoch_idx, opt_flow)

        # Stage 2: freeze flows, train predictor only
        print(f"=== Stage 2: predictor training ({stage2_epochs} epochs) ===")
        for p in dual_model.parameters():
            p.requires_grad_(False)
        for epoch_idx in range(stage1_epochs, stage1_epochs + stage2_epochs):
            if opt_pred is not None:
                avg_loss, avg_a2b, avg_b2a = _train_epoch(epoch_idx, opt_pred)
                if wandb_run:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/align_mse_a2b": avg_a2b,
                               "train/align_mse_b2a": avg_b2a, "train/stage": 2,
                               "epoch": epoch_idx+1, "step": global_step}, step=global_step)
                if (epoch_idx + 1) % save_every == 0:
                    _maybe_eval_and_save(epoch_idx, opt_pred)
        # Re-enable all grads for final save
        for p in dual_model.parameters():
            p.requires_grad_(True)
        final_optimizer = opt_pred if opt_pred is not None else opt_flow

    else:
        # One-stage: joint training
        for epoch_idx in range(start_epoch, epochs):
            avg_loss, avg_a2b, avg_b2a = _train_epoch(epoch_idx, optimizer)
            if wandb_run:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/align_mse_a2b": avg_a2b,
                           "train/align_mse_b2a": avg_b2a,
                           "epoch": epoch_idx+1, "step": global_step}, step=global_step)
            if (epoch_idx + 1) % save_every == 0:
                _maybe_eval_and_save(epoch_idx, optimizer)
        final_optimizer = optimizer

    # --- 10. Final Save & Eval ---
    save_checkpoint(exp_dir / "latest.pth.tar", model=dual_model, optimizer=final_optimizer,
                    epoch=epochs, step=global_step, axis_box=getattr(train_set, 'axis_box', None))

    if wandb_run and epochs % save_every != 0:
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

    if wandb_run:
        import wandb
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    fire.Fire(run)
