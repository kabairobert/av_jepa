import collections
from pathlib import Path
import re

import fire
import numpy as np
import torch
from torch.amp import autocast
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_device, setup_seed, setup_wandb

from examples.video_jepa.model_builder import build_video_jepa_and_probes
from examples.video_jepa.vis import (
    assemble_geometry_viz_videos,
    geometry_visualization_loop,
    log_and_save_geometry_viz,
)

logger = get_logger(__name__)
_LONG_SEQUENCE_NOTICE_LOGGED = False


def add_label_to_video(video, label):
    """Add a text label overlay on each frame of a video.

    Args:
        video: numpy array of shape (T, H, W, C) in uint8
        label: text string to add

    Returns:
        numpy array of shape (T, H, W, C)
    """
    font = ImageFont.load_default()
    T, H, W, C = video.shape

    labeled_frames = []
    for t in range(T):
        frame = Image.fromarray(video[t])
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.rectangle([0, 0, W, 20], fill=(40, 40, 40, 200))
        draw.text((4, 4), label, fill=(255, 255, 255), font=font)
        labeled_frames.append(np.array(frame))
    return np.stack(labeled_frames, axis=0)


def visualize_videos(
    batch,
    jepa,
    pixel_decoder,
    detection_head,
    num_samples,
):
    """Create visualization videos for wandb logging.

    Returns a list of videos, each with 3 vertically stacked rows:
    1. Ground truth video
    2. Predicted rollout reconstruction
    3. Digit detection overlay
    """

    x = batch["video"]
    x_jepa = jepa.encoder(x)

    T = x.shape[2]
    preds, _ = jepa.unroll(
        x,
        actions=None,
        nsteps=T - 2,
        unroll_mode="parallel",
        compute_loss=False,
        return_all_steps=True,
    )

    # One step predictions                                             # my: [S1, S2', S3', ..., S{T-1}'],        Out Shape: [B, D, T-1, H', W'] (B=batch, D=latent dim, T-1 frames, spatial).
    one_step_pred = x_jepa[:, :, 1:].clone()                           # my: [S1, S2, S3, ..., S{T-1}],           In  Shape: [B, D, T, H', W'] -> Out Shape: [B, D, T-1, H', W'] (start with the original context)(we skip t=0 to align predicted vs GT)
    one_step_pred[:, :, 1:] = preds[0]                                 # my: [S1, S2', S3', ..., S{T-1}'],        Shape: [B, D, T-1, H', W'] (replace future frames with 1-step preds)
    one_step_reconstruction = pixel_decoder.head(one_step_pred)        # my: latent [B, D, T-1, H', W'] -> decode to pixel [B, C, T-1, H, W] for visualization

    # Multi-step rollouts                                              # my: [S1, S2'(1), S3'(2), …, S{T-1}'(T-2)],  where S{t}'{t-1} is the t-th step prediction (shape [B, D, T-1, H', W'] or compatible). We iteratively overwrite future timesteps with each step's predictions to build a multi-step rollout in latent space.
    rollout = x_jepa[:, :, 1:].clone()                                 # my: [S1, S2, S3, ..., S{T-1}],           Same context slice Shape: [B, D, T-1, H', W']
    for t in range(1, T - 1):
        rollout[:, :, t:] = preds[t - 1][:, :, t - 1 :]                # my: iteratively overwrite future timesteps: preds[t-1] is the t-th step prediction (shape [B, D, T-1, H', W'] or compatible). Slicing [:, :, t-1:] aligns the predicted frames so that as t increases more of the right-hand future is replaced -> builds a multi-step rollout in latent space using each step's predictions
    rollout_reconstruction = pixel_decoder.head(rollout)               # my: decode the assembled multi-step latent rollout into pixels [B, C, T-1, H, W] for visualization and detection overlay

    # Location predictions overlaid over rollout as blue heatmap
    loc_prediction = detection_head.head(rollout)
    loc_prediction = F.interpolate(
        loc_prediction, (x.shape[-2], x.shape[-1]), mode="nearest"
    )
    loc_prediction = repeat(loc_prediction, "b t h w -> b c t h w", c=3).clone()
    loc_prediction[:, :2].fill_(0)

    # Overlay rollout reconstruction and location predictions
    detection_overlay = 0.2 * rollout_reconstruction + 0.8 * loc_prediction

    # Ground truth (skip first frame to align with predictions)
    gt = x[:, :, 1:]

    # Helper function to scale and convert pixel decoder outputs
    # to uint8 RGB and return as numpy array for video logging
    def scale_and_convert_to_uint8(tensor):
        tensor = F.interpolate(tensor, (100, 100), mode="bilinear")
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1, 1)
        tensor = torch.clamp(tensor * 255, 0, 255).to(torch.uint8)
        tensor = rearrange(tensor, "c t h w -> t h w c").cpu().numpy()
        return tensor

    rows = [gt, rollout_reconstruction, detection_overlay]
    labels = ["Ground truth", "Predicted rollout", "Digit detections"]

    viz_videos = []
    for b in range(num_samples):
        videos = [row[b] for row in rows]
        videos = [scale_and_convert_to_uint8(video) for video in videos]
        videos = [
            add_label_to_video(video, label) for video, label in zip(videos, labels)
        ]
        videos = [video.transpose(0, 3, 1, 2) for video in videos]
        viz_videos.append(np.concatenate(videos, axis=2))  # (T, C, 3*H, W)

    return viz_videos


# Run full loop over validation set and compute metrics
@torch.inference_mode()
def validation_loop(
    val_loader,
    jepa,
    detection_head,
    pixel_decoder,
    steps,
    device,
    use_amp=False,
    dtype=torch.float32,
    geometry_cfg=None,
    epoch=None,
    exp_dir=None,
    max_batches=None,
):

    # Set modules to eval mode
    jepa.eval()
    detection_head.eval()
    pixel_decoder.eval()

    metrics = collections.defaultdict(list)
    for bi, batch in enumerate(tqdm(val_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        x = batch["video"]
        loc_map = batch["digit_location"]

        with autocast(device.type, dtype=dtype, enabled=use_amp):
            recon_loss = pixel_decoder(x, x)
            det_loss = detection_head(x, loc_map)

            logs = {
                "val/recon_loss": float(recon_loss.item()),
                "val/det_loss": float(det_loss.item()),
            }
            for k, v in logs.items():
                metrics[k].append(v)

            T = x.shape[2]
            preds, _ = jepa.unroll(
                x,
                actions=None,
                nsteps=T - 2,
                unroll_mode="parallel",
                compute_loss=False,
                return_all_steps=True,
            )
            scores = detection_head.head.score(preds, loc_map[:, 2:])
            
        for s, score in enumerate(scores):
            metrics[f"AP_{s}"].append(float(score))

        if max_batches is not None and (bi + 1) >= int(max_batches):
            break

    # Aggregate val results and visualize last batch
    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    videos = visualize_videos(
        batch, jepa, pixel_decoder, detection_head, num_samples=min(16, batch["video"].shape[0])
    )
    logs = {
        **metrics,
        "viz": [wandb.Video(video, fps=4, format="mp4") for video in videos],
    }

    geometry_vis_enabled = bool((geometry_cfg or {}).get("enabled", False)) if isinstance(geometry_cfg, dict) else bool(getattr(geometry_cfg, "enabled", False) if geometry_cfg is not None else False)
    if geometry_vis_enabled and exp_dir is not None and epoch is not None:
        try:
            figures, meta = geometry_visualization_loop(
                batch=batch,
                jepa=jepa,
                device=device,
                geometry_cfg=geometry_cfg,
                detection_targets=batch.get("digit_location"),
                epoch=epoch,
            )
            global _LONG_SEQUENCE_NOTICE_LOGGED
            if not _LONG_SEQUENCE_NOTICE_LOGGED:
                logger.info(
                    "Geometry viz long-sequence controls: enabled=%s mode=%s used=%s details=%s",
                    bool(meta.get("long_sequence_enabled", False)),
                    str(meta.get("long_sequence_mode", "uniform")),
                    bool(meta.get("long_sequence_used", False)),
                    str(meta.get("long_sequence_details", "none")),
                )
                _LONG_SEQUENCE_NOTICE_LOGGED = True
            logs.update(
                log_and_save_geometry_viz(
                    figures=figures,
                    exp_dir=exp_dir,
                    epoch=epoch,
                    wandb_prefix="geometry_viz",
                    include_epoch_in_filename=bool((geometry_cfg or {}).get("include_epoch_in_filename", True)) if isinstance(geometry_cfg, dict) else bool(getattr(geometry_cfg, "include_epoch_in_filename", True) if geometry_cfg is not None else True),
                    log_to_wandb=bool((geometry_cfg or {}).get("wandb_log_geometry", True)) if isinstance(geometry_cfg, dict) else bool(getattr(geometry_cfg, "wandb_log_geometry", True) if geometry_cfg is not None else True),
                )
            )
        except Exception as exc:
            print(f"[geometry_viz] Skipping geometry plots due to error: {exc}")

    print(metrics)

    # Set modules back to train mode
    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    return logs


def _checkpoint_epoch(path):
    m = re.search(r"epoch_(\d+)\.pth\.tar$", path.name)
    if m:
        return int(m.group(1))
    return -1


def _discover_checkpoints(run_dir):
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("epoch_*.pth.tar"), key=_checkpoint_epoch)
    latest = run_dir / "latest.pth.tar"
    if latest.exists():
        ckpts.append(latest)
    return ckpts


def _load_jepa_weights(jepa, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    epoch = None
    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", None)
        if "model_state_dict" in ckpt:
            jepa.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            jepa.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            jepa.load_state_dict(ckpt, strict=False)
    else:
        jepa.load_state_dict(ckpt, strict=False)
    return epoch


def _to_bool_or_none(val):
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


def run(
    folder=None,
    cfg=None,
    eval_cfg=None,
    checkpoint=None,
    log_wandb=None,
    max_batches=None,
    batch_size=None,
    num_workers=None,
    **overrides,
):
    """Evaluate a trained video JEPA model.

    Config priority (highest last):
      1. Run's saved config.yaml  (model architecture / data / optimizer)
      2. --eval_cfg YAML          (only logging.* is applied; all other keys ignored)
      3. CLI **overrides          (dot-notation, e.g. logging.geometry_viz.enabled=true)
    """
    if folder is None and checkpoint is None:
        raise ValueError("Provide at least one of: folder or checkpoint")

    folder_path = Path(folder) if folder is not None else None
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None

    cfg_path = None
    if cfg is not None:
        cfg_path = Path(cfg)
    elif folder_path is not None:
        cand = folder_path / "config.yaml"
        if cand.exists():
            cfg_path = cand

    if cfg_path is None:
        raise ValueError("Could not resolve config path. Pass --cfg or provide --folder containing config.yaml")

    # --- Phase 1: base config from the run's saved config.yaml ---
    cfg_obj = load_config(str(cfg_path))

    # --- Phase 2: merge only logging.* from the eval-specific YAML (if given) ---
    if eval_cfg is not None:
        _eval_cfg_path = Path(eval_cfg)
        if not _eval_cfg_path.exists():
            raise FileNotFoundError(f"eval_cfg not found: {_eval_cfg_path}")
        _eval_cfg_obj = OmegaConf.load(_eval_cfg_path)
        if "logging" in _eval_cfg_obj:
            cfg_obj = OmegaConf.merge(
                cfg_obj,
                OmegaConf.create({"logging": OmegaConf.to_container(_eval_cfg_obj.logging, resolve=True)}),
            )
            logger.info(f"Applied eval config logging overrides from {_eval_cfg_path}")
        else:
            logger.warning(f"eval_cfg {_eval_cfg_path} has no 'logging' key — nothing applied")

    # --- Phase 3: CLI dot-notation overrides (highest priority) ---
    if overrides:
        _override_dict: dict = {}
        for key, value in overrides.items():
            keys = key.split(".")
            current = _override_dict
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
        cfg_obj = OmegaConf.merge(cfg_obj, OmegaConf.create(_override_dict))
        logger.info(f"Applied {len(overrides)} CLI override(s)")

    device = setup_device("cpu")
    setup_seed(cfg_obj.meta.seed)

    if folder_path is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint path is required when folder is not provided")
        folder_path = checkpoint_path.parent

    val_set = MovingMNISTDet(split="val")
    effective_batch_size = int(batch_size) if batch_size is not None else int(cfg_obj.data.batch_size)
    effective_num_workers = int(num_workers) if num_workers is not None else int(cfg_obj.data.num_workers)
    val_loader = DataLoader(
        val_set,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
    )

    built = build_video_jepa_and_probes(cfg_obj, device)
    jepa = built["jepa"]
    pixel_decoder = built["pixel_decoder"]
    detection_head = built["detection_head"]

    geometry_cfg = cfg_obj.logging.get("geometry_viz", {})

    log_wandb_override = _to_bool_or_none(log_wandb)
    enabled_wandb = bool(cfg_obj.logging.log_wandb) if log_wandb_override is None else bool(log_wandb_override)

    run_name = f"{folder_path.name}_eval"
    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "video_jepa_eval",
            "eval_mode": "standalone",
            "eval_source_folder": str(folder_path),
            **OmegaConf.to_container(cfg_obj, resolve=True),
        },
        run_dir=folder_path / "eval_wandb",
        run_name=run_name,
        tags=["video_jepa", "eval", f"seed_{cfg_obj.meta.seed}"],
        group=cfg_obj.logging.get("wandb_group"),
        enabled=enabled_wandb,
        resume=False,
    )

    if checkpoint_path is not None:
        ckpts = [checkpoint_path]
    else:
        ckpts = _discover_checkpoints(folder_path)

    if not ckpts:
        raise ValueError(f"No checkpoints found in {folder_path}")

    for ckpt in ckpts:
        loaded_epoch = _load_jepa_weights(jepa, ckpt, device)
        epoch_step = int(loaded_epoch) if loaded_epoch is not None else _checkpoint_epoch(ckpt)
        if epoch_step < 0:
            epoch_step = 0

        logs = validation_loop(
            val_loader=val_loader,
            jepa=jepa,
            detection_head=detection_head,
            pixel_decoder=pixel_decoder,
            steps=cfg_obj.model.steps,
            device=device,
            use_amp=False,
            dtype=torch.float32,
            geometry_cfg=geometry_cfg,
            epoch=epoch_step,
            exp_dir=folder_path,
            max_batches=max_batches,
        )

        logs["eval/checkpoint_name"] = ckpt.name
        logs["eval/checkpoint_path"] = str(ckpt)
        logs["eval/epoch_step"] = int(epoch_step)

        if wandb_run:
            wandb.log(logs, step=epoch_step)

        logger.info("Evaluated checkpoint %s at step=%d", ckpt.name, epoch_step)

    geometry_enabled = bool((geometry_cfg or {}).get("enabled", False)) if isinstance(geometry_cfg, dict) else bool(getattr(geometry_cfg, "enabled", False) if geometry_cfg is not None else False)
    if geometry_enabled and wandb_run:
        try:
            evo_logs = assemble_geometry_viz_videos(
                exp_dir=folder_path,
                fps=int((geometry_cfg or {}).get("evolution_fps", 2)) if isinstance(geometry_cfg, dict) else int(getattr(geometry_cfg, "evolution_fps", 2)),
                wandb_prefix="geometry_viz",
            )
            if evo_logs:
                wandb.log(evo_logs)
        except Exception:
            logger.exception("Failed assembling/logging geometry evolution videos")

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(run)
