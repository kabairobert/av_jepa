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
from multimodal_experiments.ssl_dual_alignment.vis import log_plots_to_wandb

def run(fname: str = "multimodal_experiments/ssl_dual_alignment/cfgs/paired_factors_2D.yaml", cfg=None, folder=None, **overrides):
    # --- 1. Config & Env ---
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    
    # Notebooks used double by default
    torch.set_default_dtype(torch.float32)

    # --- 2. Exp Dir Setup ---
    exp_name = (
        f"dalign_{cfg.data.get('type', '2d')}_"
        f"pred_{cfg.model.get('predictor_type', 'none')}_"
        f"loss_{cfg.loss.get('type', 'ebm')}_"
        f"l1_{cfg.loss.get('use_l1', False)}_"
        f"sparse_{cfg.loss.get('lambda_sparse', 0.0)}"
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
        path_b=cfg.data.get('path_b', None)
    )
    train_loader = DataLoader(train_set, batch_size=cfg.data.get('batch_size', 128), shuffle=True, num_workers=cfg.data.get('num_workers', 0))

    # --- 5. Model Init ---
    built = build_model_and_predictors(cfg, device)
    dual_model = built["dual_model"]
    predictor_a2b = built["predictor_a2b"]
    predictor_b2a = built["predictor_b2a"]

    # --- 6. Loss & Optim ---
    loss_type = cfg.loss.get("type", "ebm")
    if loss_type == "ebm":
        loss_fn = EBMJEPALoss(
            predictor_a2b, predictor_b2a, 
            lambda_jac=cfg.loss.get("lambda_jac", 1.0), 
            lambda_prior=cfg.loss.get("lambda_prior", 0.5), 
            lambda_sparse=cfg.loss.get("lambda_sparse", 0.1),
            use_l1=cfg.loss.get("use_l1", False)
        )
        params = list(dual_model.parameters()) + list(predictor_a2b.parameters()) + list(predictor_b2a.parameters())
    else:
        loss_fn = SupervisedFactorLoss(dimensions_per_factor=[1, 1] if cfg.data.get('type', '2d') == '2d' else [1, 1, 1])
        params = list(dual_model.parameters())

    optimizer = torch.optim.Adam(params, lr=cfg.optim.get("lr", 0.001))

    # --- 7. Resume Checkpoint ---
    start_epoch = 0
    global_step = 0
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path, dual_model, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step", 0)

    epochs = cfg.optim.get("epochs", 500)
    print(f"Starting training for {epochs} epochs...")

    # Log initial state
    if wandb_run:
        log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)

    # --- 8. Training Loop ---
    for epoch_idx in range(start_epoch, epochs):
        dual_model.train()
        epoch_loss = 0.0
        epoch_align_a2b = 0.0
        epoch_align_b2a = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{epochs}", disable=cfg.logging.get("tqdm_silent", False))
        for batch in pbar:
            data_a = batch["data_a"].to(device)
            data_b = batch["data_b"].to(device)
            corr_target = batch["corr_target"].to(device)

            optimizer.zero_grad()
            
            # Forward & Loss
            outputs = dual_model(data_a, data_b)
            if loss_type == "ebm":
                loss = loss_fn(outputs)
            else:
                loss = loss_fn(corr_target, outputs)

            # Backprop & Step
            loss.backward()
            optimizer.step()

            # Math alignment metrics
            d = (outputs.shape[1] - 2) // 2
            z_a, z_b = outputs[:, :d], outputs[:, d:2*d]
            with torch.no_grad():
                if predictor_a2b and predictor_b2a:
                    err_a2b = torch.nn.functional.mse_loss(predictor_a2b(z_a), z_b).item()
                    err_b2a = torch.nn.functional.mse_loss(predictor_b2a(z_b), z_a).item()
                    epoch_align_a2b += err_a2b
                    epoch_align_b2a += err_b2a

            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 9. Log & Save ---
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_align_a2b = epoch_align_a2b / num_batches
        avg_align_b2a = epoch_align_b2a / num_batches

        if wandb_run:
            import wandb
            wandb.log({
                "train/loss": avg_loss, 
                "train/align_mse_a2b": avg_align_a2b,
                "train/align_mse_b2a": avg_align_b2a,
                "epoch": epoch_idx+1, 
                "step": global_step
            }, step=global_step)

        if (epoch_idx + 1) % cfg.logging.get("save_every", 50) == 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch_idx+1}.pth.tar",
                model=dual_model,
                optimizer=optimizer,
                epoch=epoch_idx,
                step=global_step,
            )
            if wandb_run:
                log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)
            
    save_checkpoint(exp_dir / "latest.pth.tar", model=dual_model, optimizer=optimizer, epoch=epochs, step=global_step)
    
    if wandb_run and epochs % cfg.logging.get("save_every", 50) != 0:
        log_plots_to_wandb(dual_model, train_set, device, global_step, wandb_run)
    
    if wandb_run:
        import wandb
        wandb.finish()
        
    print("Training complete!")

if __name__ == "__main__":
    fire.Fire(run)
