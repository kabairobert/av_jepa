import collections

import numpy as np
import torch
from torch.amp import autocast
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from eb_jepa.logging import get_logger

from examples.video_jepa.vis import (
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
):

    # Set modules to eval mode
    jepa.eval()
    detection_head.eval()
    pixel_decoder.eval()

    metrics = collections.defaultdict(list)
    for batch in tqdm(val_loader):
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

    # Aggregate val results and visualize last batch
    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    videos = visualize_videos(
        batch, jepa, pixel_decoder, detection_head, num_samples=min(16, batch["video"].shape[0])
    )
    logs = {
        **metrics,
        "viz": [wandb.Video(video, fps=4, format="mp4") for video in videos],
    }

    geometry_enabled = bool((geometry_cfg or {}).get("enabled", False)) if isinstance(geometry_cfg, dict) else bool(getattr(geometry_cfg, "enabled", False) if geometry_cfg is not None else False)
    if geometry_enabled and exp_dir is not None and epoch is not None:
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
