from pathlib import Path
import importlib
import importlib.util
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

umap: Any = None
_umap_spec = importlib.util.find_spec("umap")
if _umap_spec is not None:
    umap = importlib.import_module("umap")
    UMAP_AVAILABLE = True
else:
    UMAP_AVAILABLE = False


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _plot_enabled(geometry_cfg, key):
    plots_cfg = _cfg_get(geometry_cfg, "plots", {})
    val = _cfg_get(plots_cfg, key, True)
    return bool(val)


def _epoch_suffix(epoch, include_epoch_in_title):
    if include_epoch_in_title and epoch is not None:
        return f" | epoch={int(epoch)}"
    return ""


def _time_indices(length, max_frames, device):
    if max_frames is None or max_frames <= 0 or length <= max_frames:
        return None
    # Uniformly subsample time to keep plots readable/tractable for long sequences.
    idx = torch.linspace(0, length - 1, steps=max_frames, device=device)
    idx = idx.round().long().unique_consecutive()
    return idx


def _window_indices(length, window_size, window_stride, device):
    if window_size is None or window_size <= 0 or length <= window_size:
        return None, None

    stride = max(1, int(window_stride))
    last_start = max(0, length - int(window_size))
    starts = list(range(0, last_start + 1, stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    start = starts[len(starts) // 2]
    end = start + int(window_size)
    idx = torch.arange(start, end, device=device)
    detail = f"window[{start}:{end}]/{length},stride={stride}"
    return idx, detail


def _time_indices_by_mode(length, mode, max_frames, window_size, window_stride, device):
    mode_norm = str(mode).lower()
    if mode_norm == "windowed":
        idx, detail = _window_indices(length, window_size, window_stride, device)
        return idx, detail, "windowed"

    idx = _time_indices(length, max_frames, device)
    detail = None
    if idx is not None:
        detail = f"uniform:{length}->{idx.shape[0]}"
    return idx, detail, "uniform"


def _index_time(tensor, idx):
    if idx is None:
        return tensor
    return tensor.index_select(2, idx)


def _build_rollout_latents(batch, jepa):
    x = batch["video"]
    gt_latent = jepa.encoder(x)

    tdim = x.shape[2]
    preds, _ = jepa.unroll(
        x,
        actions=None,
        nsteps=tdim - 2,
        unroll_mode="parallel",
        compute_loss=False,
        return_all_steps=True,
    )

    pred_rollout = gt_latent[:, :, 1:].clone()
    for t in range(1, tdim - 1):
        pred_rollout[:, :, t:] = preds[t - 1][:, :, t - 1 :]

    return {
        "video": x,
        "gt_latent": gt_latent,
        "pred_rollout": pred_rollout,
        "pred_steps": preds,
    }


def _avgpool_bt(latent):
    # [B, D, T, H, W] -> [B, T, D]
    return latent.mean(dim=(-2, -1)).permute(0, 2, 1).contiguous()


def _spatial_flat_bt(latent):
    # [B, D, T, H, W] -> [B, T, D*H*W]
    b, d, t, h, w = latent.shape
    return latent.permute(0, 2, 1, 3, 4).reshape(b, t, d * h * w).contiguous()


def _pick_indices(batch_size, max_count):
    return list(range(min(batch_size, max_count)))


def _progressive_groups(idxs):
    return [idxs[:1], idxs[:2], idxs[:4]]


def _trajectory_panel(
    gt_bt,
    pred_bt,
    title,
    stage_label="encoder_rollout",
    epoch=None,
    include_epoch_in_title=False,
    n_show=12,
):
    # gt_bt: [B, T, F], pred_bt: [B, T-1, F]
    bsz = gt_bt.shape[0]
    idxs = _pick_indices(bsz, n_show)
    if not idxs:
        return None

    fit = np.concatenate([gt_bt[i].cpu().numpy() for i in idxs], axis=0)
    pca = PCA(n_components=2)
    pca.fit(fit)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

    groups = _progressive_groups(idxs)

    for panel_i in range(3):
        ax = axes[panel_i]
        group = groups[panel_i]
        if not group:
            ax.axis("off")
            continue
        for j, vid_idx in enumerate(group):
            color = colors[j % len(colors)]
            gt2 = pca.transform(gt_bt[vid_idx].cpu().numpy())
            pred2 = pca.transform(pred_bt[vid_idx].cpu().numpy())
            ax.plot(gt2[:, 0], gt2[:, 1], color=color, linewidth=2)
            ax.plot(pred2[:, 0], pred2[:, 1], color=color, linestyle="--", linewidth=2)
            ax.scatter(gt2[0, 0], gt2[0, 1], color=color, s=20)
        ax.set_title(f"{title} - first {len(group)} pair(s)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)

    ax = axes[3]
    for j, vid_idx in enumerate(idxs):
        color = colors[j % len(colors)]
        gt2 = pca.transform(gt_bt[vid_idx].cpu().numpy())
        pred2 = pca.transform(pred_bt[vid_idx].cpu().numpy())
        ax.plot(gt2[:, 0], gt2[:, 1], color=color, linewidth=1.5, alpha=0.85)
        ax.plot(pred2[:, 0], pred2[:, 1], color=color, linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_title(f"{title} - summary")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)

    style_handles = [
        Line2D([0], [0], color="black", linewidth=2, label="GT"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Pred"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=5, label="t0"),
    ]
    axes[0].legend(handles=style_handles, loc="best")

    fig.suptitle(
        f"{title} [{stage_label}]" + _epoch_suffix(epoch, include_epoch_in_title),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_latent_trajectories_avgpool(
    gt_latent,
    pred_rollout,
    num_samples=12,
    stage_label="encoder_rollout",
    epoch=None,
    include_epoch_in_title=False,
    aligned=False,
):
    gt_src = gt_latent if aligned else gt_latent[:, :, 1:]
    gt_bt = _avgpool_bt(gt_src)
    pred_bt = _avgpool_bt(pred_rollout)
    return _trajectory_panel(
        gt_bt,
        pred_bt,
        "Latent Trajectories (avgpool)",
        stage_label=stage_label,
        epoch=epoch,
        include_epoch_in_title=include_epoch_in_title,
        n_show=num_samples,
    )


def plot_latent_trajectories_spatialflat(
    gt_latent,
    pred_rollout,
    num_samples=12,
    stage_label="encoder_rollout",
    epoch=None,
    include_epoch_in_title=False,
    aligned=False,
):
    gt_src = gt_latent if aligned else gt_latent[:, :, 1:]
    gt_bt = _spatial_flat_bt(gt_src)
    pred_bt = _spatial_flat_bt(pred_rollout)
    return _trajectory_panel(
        gt_bt,
        pred_bt,
        "Latent Trajectories (spatial-flat)",
        stage_label=stage_label,
        epoch=epoch,
        include_epoch_in_title=include_epoch_in_title,
        n_show=num_samples,
    )


def plot_temporal_self_similarity(
    gt_latent,
    num_samples=2,
    stage_label="encoder_rollout",
    epoch=None,
    include_epoch_in_title=False,
    aligned=False,
):
    gt_src = gt_latent if aligned else gt_latent[:, :, 1:]
    bt = _avgpool_bt(gt_src)
    idxs = _pick_indices(bt.shape[0], num_samples)
    if not idxs:
        return None

    n = len(idxs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue
        x = bt[idxs[i]]
        x = F.normalize(x, dim=-1)
        sim = x @ x.transpose(0, 1)
        im = ax.imshow(sim.cpu().numpy(), vmin=-1, vmax=1, cmap="viridis")
        ax.set_title(f"sample_idx={idxs[i]}")
        ax.set_xlabel("t")
        ax.set_ylabel("t")

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(im, cax=cax)
    fig.suptitle(
        f"Temporal Self-Similarity [{stage_label}] | sample_indices={idxs}"
        + _epoch_suffix(epoch, include_epoch_in_title),
        fontsize=12,
    )
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.07, top=0.90, wspace=0.24, hspace=0.28)
    return fig


def plot_occupancy_spatial_embedding(
    gt_latent,
    digit_location,
    method="umap",
    max_points=10000,
    max_frames=None,
    stage_label="encoder_rollout",
    epoch=None,
    include_epoch_in_title=False,
):
    if digit_location is None:
        return None

    idx = _time_indices(gt_latent.shape[2], max_frames, gt_latent.device)
    gt_latent = _index_time(gt_latent, idx)
    if idx is not None:
        digit_location = digit_location.index_select(1, idx.to(digit_location.device))

    b, d, t, h, w = gt_latent.shape
    x = gt_latent.permute(0, 2, 3, 4, 1).reshape(-1, d).detach().cpu().numpy()

    # Align GT occupancy with latent spatial size: [B, T, Hm, Wm] -> [B, T, h, w]
    occ = digit_location[:, :t].float()
    b_occ, t_occ, hm, wm = occ.shape
    occ = occ.reshape(b_occ * t_occ, 1, hm, wm)
    occ = F.interpolate(occ, size=(h, w), mode="nearest")
    occ = occ.reshape(b_occ, t_occ, h, w)
    y = occ.reshape(-1).detach().cpu().numpy().astype(np.int32)

    n = x.shape[0]
    if n > max_points:
        sel = np.random.RandomState(2025).choice(n, size=max_points, replace=False)
        x = x[sel]
        y = y[sel]

    used = method.lower()
    fallback_note = ""
    if used == "umap" and UMAP_AVAILABLE:
        assert umap is not None
        embed = umap.UMAP(n_components=2, random_state=42).fit_transform(x)
        method_label = "UMAP"
    else:
        if used == "umap" and not UMAP_AVAILABLE:
            fallback_note = " (fallback)"
        embed = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(x)
        method_label = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 7))
    bg = y == 0
    fg = y == 1
    ax.scatter(embed[bg, 0], embed[bg, 1], s=4, alpha=0.25, c="#1f77b4", label="background")
    ax.scatter(embed[fg, 0], embed[fg, 1], s=8, alpha=0.7, c="#d62728", label="digit")
    ax.set_title(
        f"Occupancy Spatial Embedding ({method_label}{fallback_note}) [{stage_label}]"
        + _epoch_suffix(epoch, include_epoch_in_title)
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _get_projector_module(jepa):
    reg = getattr(jepa, "regularizer", None)
    if reg is None:
        return None
    proj = getattr(reg, "proj", None)
    if proj is None:
        proj = getattr(reg, "projector", None)
    return proj


def plot_cov_eig_spectrum(
    gt_latent,
    jepa,
    max_samples=2048,
    stage_label="projector",
    epoch=None,
    include_epoch_in_title=False,
):
    proj = _get_projector_module(jepa)
    x = gt_latent.permute(0, 2, 3, 4, 1).reshape(-1, gt_latent.shape[1])
    if x.shape[0] > max_samples:
        x = x[:max_samples]

    if proj is not None:
        with torch.no_grad():
            x = proj(x)

    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.t() @ x) / max(1, x.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    eigvals = torch.sort(eigvals, descending=True).values.cpu().numpy()

    k = np.arange(1, eigvals.shape[0] + 1)
    uniform = np.full_like(eigvals, eigvals.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k, eigvals, linewidth=2, label="spectrum")
    ax.plot(k, uniform, linestyle="--", linewidth=1.5, label="uniform ref")
    ax.set_yscale("log")
    ax.set_title(f"Covariance Eigenvalue Spectrum [{stage_label}]" + _epoch_suffix(epoch, include_epoch_in_title))
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue (log)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _autodetect_activation_layer(model, sample_x):
    outputs = {}
    handles = []

    def make_hook(name):
        def _hook(_m, _i, out):
            if isinstance(out, torch.Tensor) and out.dim() == 5 and out.shape[-1] > 1 and out.shape[-2] > 1:
                outputs[name] = tuple(out.shape)

        return _hook

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(sample_x)
    except Exception:
        pass

    for h in handles:
        h.remove()

    if not outputs:
        return None
    return list(outputs.keys())[-1]


def _resolve_module(root, dot_path):
    mod = root
    for attr in dot_path.split("."):
        mod = getattr(mod, attr)
    return mod


def plot_activation_overlays(
    batch,
    jepa,
    layer_name=None,
    num_samples=4,
    max_frames=None,
    stage_label="encoder_rollout",
    epoch=None,
    include_epoch_in_title=False,
):
    x = batch["video"]
    x = x[: min(num_samples, x.shape[0])]
    idx = _time_indices(x.shape[2], max_frames, x.device)
    x = _index_time(x, idx)
    model = jepa.encoder

    chosen = layer_name or _autodetect_activation_layer(model, x)
    if chosen is None:
        return None

    activations = {}

    def _hook(_m, _i, out):
        activations["feat"] = out.detach()

    layer = _resolve_module(model, chosen)
    handle = layer.register_forward_hook(_hook)
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    if "feat" not in activations:
        return None

    feat = activations["feat"]
    sal = feat.mean(dim=1)
    mins = sal.flatten(2).min(dim=2).values[:, :, None, None]
    maxs = sal.flatten(2).max(dim=2).values[:, :, None, None]
    sal = (sal - mins) / (maxs - mins + 1e-8)

    b, t, h, w = sal.shape
    fig, axes = plt.subplots(b, t, figsize=(1.8 * t, 1.8 * b), squeeze=False)
    cmap = cm.get_cmap("jet")

    for bi in range(b):
        for ti in range(t):
            ax = axes[bi, ti]
            frame = x[bi, 0, ti].detach().cpu().numpy()
            heat = sal[bi, ti].detach().cpu().numpy()
            heat = torch.from_numpy(heat)[None, None]
            heat = F.interpolate(heat, size=frame.shape, mode="bilinear", align_corners=False)
            heat = heat.squeeze().numpy()
            color = cmap(heat)[..., :3]
            gray = np.stack([frame, frame, frame], axis=-1)
            gray = np.clip(gray, 0, 1)
            overlay = np.clip(0.55 * gray + 0.45 * color, 0, 1)
            ax.imshow(overlay)
            ax.axis("off")
            if bi == 0:
                ax.set_title(f"t={ti}")
    fig.suptitle(
        f"Activation Overlays ({chosen}) [{stage_label}]" + _epoch_suffix(epoch, include_epoch_in_title),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_phase_space_portrait(
    gt_latent,
    pred_rollout,
    num_samples=4,
    stage_label="encoder_rollout",
    epoch=None,
    include_epoch_in_title=False,
    aligned=False,
):
    gt_src = gt_latent if aligned else gt_latent[:, :, 1:]
    gt_bt = _spatial_flat_bt(gt_src)
    pred_bt = _spatial_flat_bt(pred_rollout)
    idxs = _pick_indices(gt_bt.shape[0], num_samples)
    if not idxs:
        return None

    fig = plt.figure(figsize=(12, 9))
    groups = _progressive_groups(idxs)
    groups.append(idxs)

    for i, group in enumerate(groups[:4]):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        if not group:
            ax.set_axis_off()
            continue

        fit = np.concatenate(
            [
                np.concatenate([gt_bt[vid_idx].cpu().numpy(), pred_bt[vid_idx].cpu().numpy()], axis=0)
                for vid_idx in group
            ],
            axis=0,
        )
        pca = PCA(n_components=3)
        pca.fit(fit)

        colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
        for j, vid_idx in enumerate(group):
            color = colors[j % len(colors)]
            gt3 = pca.transform(gt_bt[vid_idx].cpu().numpy())
            pr3 = pca.transform(pred_bt[vid_idx].cpu().numpy())
            ax.plot(gt3[:, 0], gt3[:, 1], gt3[:, 2], color=color, linewidth=2)
            ax.plot(pr3[:, 0], pr3[:, 1], pr3[:, 2], color=color, linestyle="--", linewidth=2)

        if i < 3:
            ax.set_title(f"first {len(group)} pair(s)")
        else:
            ax.set_title("summary")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        if i == 0:
            style_handles = [
                Line2D([0], [0], color="black", linewidth=2, label="GT"),
                Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Pred"),
            ]
            ax.legend(handles=style_handles, loc="best")
    fig.suptitle(
        f"Phase Space Portrait (static 3D) [{stage_label}]" + _epoch_suffix(epoch, include_epoch_in_title)
    )
    # tight_layout is unreliable with 3D axes; adjust spacing manually.
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.90, wspace=0.22, hspace=0.26)
    return fig


def geometry_visualization_loop(
    batch,
    jepa,
    device,
    geometry_cfg=None,
    detection_targets=None,
    epoch=None,
):
    _ = device
    bundle = _build_rollout_latents(batch, jepa)
    gt_latent = bundle["gt_latent"]
    pred_rollout = bundle["pred_rollout"]

    num_samples = int(_cfg_get(geometry_cfg, "num_samples", 12))
    tsne_method = str(_cfg_get(geometry_cfg, "tsne_method", "umap"))
    act_layer = _cfg_get(geometry_cfg, "activation_layer", None)
    stage_label = str(_cfg_get(geometry_cfg, "stage_label", "encoder_rollout"))
    include_epoch_in_title = bool(_cfg_get(geometry_cfg, "include_epoch_in_title", True))
    tsne_max_points = int(_cfg_get(geometry_cfg, "spatial_tsne_max_points", 10000))
    cov_max_samples = int(_cfg_get(geometry_cfg, "covariance_max_samples", 2048))
    tss_num_samples = int(_cfg_get(geometry_cfg, "temporal_self_similarity_num_samples", 2))

    long_cfg = _cfg_get(geometry_cfg, "long_sequence", {})
    long_enabled = bool(_cfg_get(long_cfg, "enabled", True))
    time_subsample_mode = str(_cfg_get(long_cfg, "time_subsample_mode", "uniform")).lower()
    max_frames_for_plots = int(_cfg_get(long_cfg, "max_frames_for_plots", 96))
    max_frames_for_embedding = int(_cfg_get(long_cfg, "max_frames_for_embedding", 96))
    max_frames_activation_overlay = int(_cfg_get(long_cfg, "max_frames_activation_overlay", 24))
    window_size = int(_cfg_get(long_cfg, "window_size", 96))
    window_stride = int(_cfg_get(long_cfg, "window_stride", 48))

    long_used = False
    long_msgs = []

    gt_seq = gt_latent[:, :, 1:]
    pred_seq = pred_rollout
    if long_enabled:
        seq_window_size = min(window_size, max_frames_for_plots)
        idx_seq, seq_detail, mode_used = _time_indices_by_mode(
            length=gt_seq.shape[2],
            mode=time_subsample_mode,
            max_frames=max_frames_for_plots,
            window_size=seq_window_size,
            window_stride=window_stride,
            device=gt_seq.device,
        )
        if idx_seq is not None:
            gt_seq = _index_time(gt_seq, idx_seq)
            pred_seq = _index_time(pred_seq, idx_seq)
            long_used = True
            if seq_detail is None:
                seq_detail = f"seq:{gt_latent.shape[2]-1}->{gt_seq.shape[2]}"
            long_msgs.append(f"seq[{mode_used}]:{seq_detail}")

    figs = {}

    if _plot_enabled(geometry_cfg, "latent_trajectories_avgpool"):
        figs["latent_trajectories_avgpool"] = plot_latent_trajectories_avgpool(
            gt_seq,
            pred_seq,
            num_samples=num_samples,
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
        )

    if _plot_enabled(geometry_cfg, "latent_trajectories_spatialflat"):
        figs["latent_trajectories_spatialflat"] = plot_latent_trajectories_spatialflat(
            gt_seq,
            pred_seq,
            num_samples=num_samples,
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
        )

    if _plot_enabled(geometry_cfg, "temporal_self_similarity"):
        figs["temporal_self_similarity"] = plot_temporal_self_similarity(
            gt_seq,
            num_samples=tss_num_samples,
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
        )

    if _plot_enabled(geometry_cfg, "occupancy_spatial_embedding"):
        figs["occupancy_spatial_embedding"] = plot_occupancy_spatial_embedding(
            gt_latent,
            detection_targets,
            method=tsne_method,
            max_points=tsne_max_points,
            max_frames=(
                min(window_size, max_frames_for_embedding)
                if (long_enabled and time_subsample_mode == "windowed")
                else (max_frames_for_embedding if long_enabled else None)
            ),
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
        )
        if long_enabled and gt_latent.shape[2] > max_frames_for_embedding:
            long_used = True
            long_msgs.append(
                f"embed[{time_subsample_mode}]:{gt_latent.shape[2]}->{min(window_size, max_frames_for_embedding) if time_subsample_mode == 'windowed' else max_frames_for_embedding}"
            )

    if _plot_enabled(geometry_cfg, "cov_eig_spectrum"):
        figs["cov_eig_spectrum"] = plot_cov_eig_spectrum(
            gt_latent,
            jepa,
            max_samples=cov_max_samples,
            stage_label="projector_or_encoder",
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
        )

    if _plot_enabled(geometry_cfg, "activation_overlays"):
        figs["activation_overlays"] = plot_activation_overlays(
            batch,
            jepa,
            layer_name=act_layer,
            num_samples=min(4, batch["video"].shape[0]),
            max_frames=(
                min(window_size, max_frames_activation_overlay)
                if (long_enabled and time_subsample_mode == "windowed")
                else (max_frames_activation_overlay if long_enabled else None)
            ),
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
        )
        if long_enabled and batch["video"].shape[2] > max_frames_activation_overlay:
            long_used = True
            long_msgs.append(
                f"overlay[{time_subsample_mode}]:{batch['video'].shape[2]}->{min(window_size, max_frames_activation_overlay) if time_subsample_mode == 'windowed' else max_frames_activation_overlay}"
            )

    if _plot_enabled(geometry_cfg, "phase_space_portrait"):
        figs["phase_space_portrait"] = plot_phase_space_portrait(
            gt_seq,
            pred_seq,
            num_samples=min(4, batch["video"].shape[0]),
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
        )

    meta = {
        "long_sequence_enabled": long_enabled,
        "long_sequence_mode": time_subsample_mode,
        "long_sequence_used": long_used,
        "long_sequence_details": ", ".join(long_msgs) if long_msgs else "",
    }
    return {k: v for k, v in figs.items() if v is not None}, meta


def log_and_save_geometry_viz(
    figures,
    exp_dir,
    epoch,
    wandb_prefix="geometry_viz",
    include_epoch_in_filename=True,
):
    exp_dir = Path(exp_dir)
    epoch_dir = exp_dir / "geometry_viz" / f"epoch_{int(epoch):04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    logs = {}
    for key, fig in figures.items():
        if include_epoch_in_filename:
            out_path = epoch_dir / f"{key}_epoch_{int(epoch):04d}.png"
        else:
            out_path = epoch_dir / f"{key}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        logs[f"{wandb_prefix}/{key}"] = wandb.Image(str(out_path), caption=f"epoch={int(epoch)}")
    return logs


def assemble_geometry_viz_videos(exp_dir, fps=2, wandb_prefix="geometry_viz"):
    import imageio.v2 as imageio

    def _pad_frame_to_macroblock(frame, block=16):
        if block is None or block <= 1:
            return frame
        h, w = frame.shape[:2]
        nh = ((h + block - 1) // block) * block
        nw = ((w + block - 1) // block) * block
        if nh == h and nw == w:
            return frame

        if frame.ndim == 3:
            pad = ((0, nh - h), (0, nw - w), (0, 0))
        else:
            pad = ((0, nh - h), (0, nw - w))
        return np.pad(frame, pad, mode="edge")

    exp_dir = Path(exp_dir)
    base = exp_dir / "geometry_viz"
    if not base.exists():
        return {}

    epoch_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("epoch_")])
    if not epoch_dirs:
        return {}

    keys = set()
    for d in epoch_dirs:
        for p in d.glob("*.png"):
            stem = p.stem
            if "_epoch_" in stem:
                stem = stem.rsplit("_epoch_", 1)[0]
            keys.add(stem)
    keys = sorted(keys)
    logs = {}

    for key in keys:
        frames = []
        for d in epoch_dirs:
            # Prefer epoch-suffixed files when present, fallback to legacy plain key.png.
            cands = sorted(d.glob(f"{key}_epoch_*.png"))
            if cands:
                frames.append(imageio.imread(cands[0]))
                continue
            img_path = d / f"{key}.png"
            if img_path.exists():
                frames.append(imageio.imread(img_path))
        if not frames:
            continue

        video_path = base / f"{key}_evolution.mp4"
        with imageio.get_writer(video_path, fps=fps) as writer:
            for fr in frames:
                writer.append_data(_pad_frame_to_macroblock(fr))  # type: ignore[attr-defined]

        logs[f"{wandb_prefix}/{key}_evolution"] = wandb.Video(str(video_path), fps=fps, format="mp4")

    return logs
