from pathlib import Path
import importlib
import importlib.util
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
import torch.nn as nn
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


def _build_pred_rollout(gt_latent, preds, tdim):
    pred_rollout = gt_latent[:, :, 1:].clone()
    for t in range(1, tdim - 1):
        pred_rollout[:, :, t:] = preds[t - 1][:, :, t - 1 :]
    return pred_rollout


def _build_rollout_latents(batch, jepa, use_amp=False, dtype=torch.float32):
    x = batch["video"]
    with autocast(x.device.type, dtype=dtype, enabled=use_amp):
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

    pred_rollout = _build_pred_rollout(gt_latent, preds, tdim)

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


def _tss_feature_bt(latent, feature_mode):
    mode = str(feature_mode).lower()
    if mode == "avgpool":
        return _avgpool_bt(latent)
    return _spatial_flat_bt(latent)


def _fit_tss_pca_features(x, fixed_k, var_ratio, pca_max_k):
    # x: [T, F]
    t, f = x.shape
    max_k = max(1, min(int(pca_max_k), int(t), int(f)))
    fixed_k = max(1, min(int(fixed_k), max_k))

    pca = PCA(n_components=max_k)
    z_all = pca.fit_transform(x)

    z_fixed = z_all[:, :fixed_k]

    evr = pca.explained_variance_ratio_
    target = float(var_ratio)
    target = min(max(target, 1e-6), 1.0)
    if evr.size == 0:
        var_k = 1
    else:
        var_k = int(np.searchsorted(np.cumsum(evr), target) + 1)
    var_k = max(1, min(var_k, max_k))
    z_var = z_all[:, :var_k]

    return z_fixed, z_var, fixed_k, var_k


def _pairwise_time_matrix(x, distance):
    mode = str(distance).lower()
    xt = torch.from_numpy(x).float()
    if mode == "cosine":
        xt = F.normalize(xt, dim=-1)
        return (xt @ xt.transpose(0, 1)).cpu().numpy()
    return torch.cdist(xt, xt, p=2).cpu().numpy()


def _pick_indices(batch_size, max_count):
    return list(range(min(batch_size, max_count)))


def _progressive_groups(idxs):
    return [idxs[:1], idxs[:2], idxs[:4]]


def _offdiag_values(matrix):
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask].astype(np.float64, copy=False)


def _neighbor_values(matrix):
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    idx = np.arange(matrix.shape[0] - 1)
    vals = np.concatenate([matrix[idx, idx + 1], matrix[idx + 1, idx]])
    return vals.astype(np.float64, copy=False)


def _far_values(matrix):
    if matrix.ndim != 2 or matrix.shape[0] < 4:
        return _offdiag_values(matrix)
    t = matrix.shape[0]
    band = max(2, t // 4)
    rows, cols = np.indices(matrix.shape)
    mask = np.abs(rows - cols) >= band
    return matrix[mask].astype(np.float64, copy=False)


def _matrix_summary_metrics(matrix, prefix, distance_mode):
    offdiag = _offdiag_values(matrix)
    near = _neighbor_values(matrix)
    far = _far_values(matrix)
    if offdiag.size == 0:
        return {
            f"{prefix}/sample_count": 0.0,
        }
    metrics = {
        f"{prefix}/sample_count": float(offdiag.size),
        f"{prefix}/offdiag_mean": float(offdiag.mean()),
        f"{prefix}/offdiag_std": float(offdiag.std()),
        f"{prefix}/neighbor_mean": float(near.mean()) if near.size else float("nan"),
        f"{prefix}/far_mean": float(far.mean()) if far.size else float("nan"),
    }
    if near.size and far.size:
        if str(distance_mode).lower() == "cosine":
            metrics[f"{prefix}/temporal_contrast"] = float(near.mean() - far.mean())
        else:
            metrics[f"{prefix}/temporal_contrast"] = float(far.mean() - near.mean())
    return metrics


def _trajectory_summary_metrics(gt_coords, pred_coords, prefix):
    if gt_coords.size == 0 or pred_coords.size == 0:
        return {f"{prefix}/sample_count": 0.0}
    diffs = np.linalg.norm(gt_coords - pred_coords, axis=-1)
    return {
        f"{prefix}/sample_count": float(diffs.size),
        f"{prefix}/mean_divergence": float(diffs.mean()),
        f"{prefix}/final_divergence": float(diffs[:, -1].mean()),
        f"{prefix}/max_divergence": float(diffs.max()),
    }


def _embedding_summary_metrics(embed, labels, prefix):
    if embed.size == 0 or labels.size == 0:
        return {f"{prefix}/sample_count": 0.0}
    bg = embed[labels == 0]
    fg = embed[labels == 1]
    if bg.size == 0 or fg.size == 0:
        return {f"{prefix}/sample_count": float(embed.shape[0])}
    bg_centroid = bg.mean(axis=0)
    fg_centroid = fg.mean(axis=0)
    centroid_distance = float(np.linalg.norm(bg_centroid - fg_centroid))
    bg_spread = float(np.linalg.norm(bg - bg_centroid, axis=1).mean()) if bg.shape[0] else 0.0
    fg_spread = float(np.linalg.norm(fg - fg_centroid, axis=1).mean()) if fg.shape[0] else 0.0
    pooled = max(1e-12, 0.5 * (bg_spread + fg_spread))
    return {
        f"{prefix}/sample_count": float(embed.shape[0]),
        f"{prefix}/centroid_distance": centroid_distance,
        f"{prefix}/pooled_spread": pooled,
        f"{prefix}/separation_ratio": centroid_distance / pooled,
        f"{prefix}/foreground_fraction": float((labels == 1).mean()),
    }


def _activation_summary_metrics(saliency, prefix):
    if saliency.numel() == 0:
        return {f"{prefix}/sample_count": 0.0}
    flat = saliency.flatten(2)
    probs = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    topk = max(1, int(flat.shape[-1] * 0.1))
    top_mass = flat.topk(topk, dim=-1).values.sum(dim=-1) / flat.sum(dim=-1).clamp_min(1e-8)
    drift = torch.linalg.vector_norm(saliency[:, 1:] - saliency[:, :-1], dim=(-2, -1)) if saliency.shape[1] > 1 else torch.zeros_like(entropy[:, :1])
    return {
        f"{prefix}/sample_count": float(saliency.shape[0] * saliency.shape[1]),
        f"{prefix}/entropy_mean": float(entropy.mean().item()),
        f"{prefix}/top10_mass_mean": float(top_mass.mean().item()),
        f"{prefix}/temporal_drift_mean": float(drift.mean().item()),
    }


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
        return None, {}

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
    gt_coords = np.stack([pca.transform(gt_bt[vid_idx].cpu().numpy()) for vid_idx in idxs], axis=0)
    pred_coords = np.stack([pca.transform(pred_bt[vid_idx].cpu().numpy()) for vid_idx in idxs], axis=0)
    payload = {
        "sample_indices": np.asarray(idxs, dtype=np.int64),
        "gt_coords": gt_coords,
        "pred_coords": pred_coords,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "summary_metrics": _trajectory_summary_metrics(gt_coords, pred_coords, f"val/diag/trajectory/{stage_label}/{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}"),
    }
    return fig, payload


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
    feature_mode="spatialflat",
    distance="euclidean",
    pca_fixed_k=3,
    pca_var_ratio=0.95,
    pca_max_k=64,
    center_time=True,
):
    gt_src = gt_latent if aligned else gt_latent[:, :, 1:]
    bt = _tss_feature_bt(gt_src, feature_mode)
    idxs = _pick_indices(bt.shape[0], num_samples)
    if not idxs:
        return None, {}

    rows = []
    stats = []
    for idx in idxs:
        x = bt[idx].detach().float()
        if center_time:
            x = x - x.mean(dim=0, keepdim=True)
        x_np = x.cpu().numpy()

        z_fixed, z_var, used_fixed_k, used_var_k = _fit_tss_pca_features(
            x_np,
            fixed_k=pca_fixed_k,
            var_ratio=pca_var_ratio,
            pca_max_k=pca_max_k,
        )
        m_fixed = _pairwise_time_matrix(z_fixed, distance=distance)
        m_var = _pairwise_time_matrix(z_var, distance=distance)
        rows.append((m_fixed, m_var))
        stats.append((used_fixed_k, used_var_k))

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11.0, 4.2 * n), squeeze=False)

    mode = str(distance).lower()
    if mode == "cosine":
        vmin, vmax = -1.0, 1.0
    else:
        vmax = max(float(np.max(m)) for mf, mv in rows for m in (mf, mv))
        if vmax <= 0:
            vmax = 1.0
        vmin = 0.0

    im = None
    for i, ((m_fixed, m_var), (used_fixed_k, used_var_k)) in enumerate(zip(rows, stats)):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        im = ax_l.imshow(m_fixed, vmin=vmin, vmax=vmax, cmap="viridis")
        ax_l.set_title(f"sample_idx={idxs[i]} | Vis 2.1 fixed-k={used_fixed_k}")
        ax_l.set_xlabel("t")
        ax_l.set_ylabel("t")

        im = ax_r.imshow(m_var, vmin=vmin, vmax=vmax, cmap="viridis")
        ax_r.set_title(
            f"sample_idx={idxs[i]} | Vis 2.2 var{int(round(float(pca_var_ratio) * 100))}% k={used_var_k}"
        )
        ax_r.set_xlabel("t")
        ax_r.set_ylabel("t")

    cax = fig.add_axes((0.91, 0.12, 0.02, 0.76))
    assert im is not None
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("cosine similarity" if mode == "cosine" else "euclidean distance")
    fig.suptitle(
        "Temporal Time-Delay Matrices (Vis 2.1 fixed-k, Vis 2.2 var-ratio) "
        f"[{stage_label}] | feature_mode={str(feature_mode).lower()}"
        f" | metric={mode} | sample_indices={idxs}"
        + _epoch_suffix(epoch, include_epoch_in_title),
        fontsize=12,
    )
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.06, top=0.90, wspace=0.22, hspace=0.30)
    payload = {
        "sample_indices": np.asarray(idxs, dtype=np.int64),
        "fixed_matrices": np.stack([mf for mf, _ in rows], axis=0),
        "var_matrices": np.stack([mv for _, mv in rows], axis=0),
        "used_fixed_ks": np.asarray([s[0] for s in stats], dtype=np.int64),
        "used_var_ks": np.asarray([s[1] for s in stats], dtype=np.int64),
        "distance": mode,
        "feature_mode": str(feature_mode).lower(),
        "summary_metrics": {},
    }
    summary_metrics = {}
    for idx_value, (m_fixed, m_var) in zip(idxs, rows):
        summary_metrics.update(_matrix_summary_metrics(m_fixed, f"val/diag/temporal/fixed/sample_{idx_value}", mode))
        summary_metrics.update(_matrix_summary_metrics(m_var, f"val/diag/temporal/var/sample_{idx_value}", mode))
    if rows:
        payload["summary_metrics"] = summary_metrics
    return fig, payload


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
        return None, {}

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
    payload = {
        "embedding": embed,
        "labels": y,
        "method": method_label,
        "sample_count": int(embed.shape[0]),
        "summary_metrics": _embedding_summary_metrics(embed, y, f"val/diag/embedding/{stage_label}"),
    }
    return fig, payload


def _get_projector_module(jepa):
    reg = getattr(jepa, "regularizer", None)
    if reg is None:
        return None
    proj = getattr(reg, "proj", None)
    if proj is None:
        proj = getattr(reg, "projector", None)
    return proj


class StreamingCovarianceStats:
    def __init__(self):
        self.count = 0
        self.sum_vec = None
        self.sum_outer = None

    def update(self, x):
        if x is None:
            return
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if x.numel() == 0:
            return
        x_cpu = x.cpu().to(torch.float64)
        batch_count = int(x_cpu.shape[0])
        batch_sum = x_cpu.sum(dim=0)
        batch_outer = x_cpu.transpose(0, 1) @ x_cpu
        if self.sum_vec is None:
            self.sum_vec = batch_sum
            self.sum_outer = batch_outer
        else:
            self.sum_vec += batch_sum
            self.sum_outer += batch_outer
        self.count += batch_count

    def covariance(self):
        if self.count < 2 or self.sum_vec is None or self.sum_outer is None:
            return None
        mean = self.sum_vec / float(self.count)
        cov = (self.sum_outer - float(self.count) * torch.outer(mean, mean)) / float(max(1, self.count - 1))
        return cov

    def eigvals(self):
        cov = self.covariance()
        if cov is None:
            return None
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
        return torch.sort(eigvals, descending=True).values.cpu().numpy()


def _flatten_encoder_tokens(gt_latent):
    return gt_latent.permute(0, 2, 3, 4, 1).reshape(-1, gt_latent.shape[1]).contiguous()


def _projector_linear_outputs(projector, x):
    outputs = {}
    if projector is None:
        return outputs

    if hasattr(projector, "net") and isinstance(projector.net, nn.Sequential):
        current = x
        linear_idx = 0
        for module in projector.net:
            current = module(current)
            if isinstance(module, nn.Linear):
                outputs[f"projector_layer_{linear_idx}"] = current
                linear_idx += 1
        if linear_idx > 0:
            outputs["projector_output"] = current
        return outputs

    outputs["projector_output"] = projector(x)
    return outputs


def _covariance_source_mode(geometry_cfg):
    return str(_cfg_get(geometry_cfg, "covariance_source_mode", "encoder")).lower()


def iter_covariance_sources(gt_latent, jepa, source_mode="encoder"):
    mode = str(source_mode).lower()
    encoder_tokens = _flatten_encoder_tokens(gt_latent)

    if mode in ("encoder", "encoder+projector_output", "encoder+projector_all_linear_layers"):
        yield "encoder", encoder_tokens

    if mode == "encoder":
        return

    projector = _get_projector_module(jepa)
    if projector is None:
        return

    projector_outputs = _projector_linear_outputs(projector, encoder_tokens)
    if mode in ("projector_output", "encoder+projector_output"):
        if "projector_output" in projector_outputs:
            yield "projector_output", projector_outputs["projector_output"]
        return

    if mode in ("projector_all_linear_layers", "encoder+projector_all_linear_layers"):
        for key, value in projector_outputs.items():
            if key == "projector_output":
                continue
            yield key, value


def update_covariance_trackers(trackers, gt_latent, jepa, source_mode="encoder"):
    for key, values in iter_covariance_sources(gt_latent, jepa, source_mode=source_mode):
        tracker = trackers.setdefault(key, StreamingCovarianceStats())
        tracker.update(values)


def _covariance_scalar_metrics(eigvals, metric_prefix, sample_count):
    eigvals = np.asarray(eigvals, dtype=np.float64)
    total = float(np.sum(eigvals))
    if total <= 0:
        total = 1e-12
    probs = eigvals / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    eff_rank = float(np.exp(entropy))
    pr = float((total ** 2) / max(np.sum(eigvals ** 2), 1e-12))
    top1 = float(np.sum(eigvals[:1]) / total)
    top5 = float(np.sum(eigvals[: min(5, eigvals.shape[0])]) / total)
    return {
        f"{metric_prefix}/effective_rank": eff_rank,
        f"{metric_prefix}/participation_ratio": pr,
        f"{metric_prefix}/top1_frac": top1,
        f"{metric_prefix}/top5_frac": top5,
        f"{metric_prefix}/trace": float(np.sum(eigvals)),
        f"{metric_prefix}/sample_count": float(sample_count),
        f"{metric_prefix}/feature_dim": float(eigvals.shape[0]),
    }


def finalize_covariance_diagnostics(trackers):
    diagnostics = {}
    for key, tracker in trackers.items():
        eigvals = tracker.eigvals()
        if eigvals is None:
            continue
        metric_prefix = f"val/diag/cov/{key}"
        diagnostics[key] = {
            "eigvals": eigvals,
            "sample_count": tracker.count,
            "metrics": _covariance_scalar_metrics(eigvals, metric_prefix, tracker.count),
        }
    return diagnostics


def _covariance_plot_title(key):
    if key == "encoder":
        return "encoder"
    if key == "projector_output":
        return "projector output"
    if key.startswith("projector_layer_"):
        idx = key.rsplit("_", 1)[-1]
        return f"projector linear layer {idx}"
    return key.replace("_", " ")


def plot_cov_eig_spectrum_from_eigvals(
    eigvals,
    plot_label,
    stage_label="encoder",
    epoch=None,
    include_epoch_in_title=False,
):
    eigvals = np.asarray(eigvals, dtype=np.float64)

    k = np.arange(1, eigvals.shape[0] + 1)
    uniform = np.full_like(eigvals, eigvals.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k, eigvals, linewidth=2, label="spectrum")
    ax.plot(k, uniform, linestyle="--", linewidth=1.5, label="uniform ref")
    ax.set_yscale("log")
    ax.set_title(
        f"Covariance Eigenvalue Spectrum ({plot_label}) [{stage_label}]"
        + _epoch_suffix(epoch, include_epoch_in_title)
    )
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue (log)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    payload = {
        "eigvals": eigvals,
        "rank": k,
        "uniform_reference": uniform,
    }
    return fig, payload


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
        return None, {}

    activations = {}

    def _hook(_m, _i, out):
        activations["feat"] = out.detach()

    layer = _resolve_module(model, chosen)
    handle = layer.register_forward_hook(_hook)
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    if "feat" not in activations:
        return None, {}

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
    payload = {
        "layer_name": chosen,
        "saliency": sal.detach().cpu().numpy(),
        "summary_metrics": _activation_summary_metrics(sal, f"val/diag/activation/{chosen}"),
    }
    return fig, payload


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
        return None, {}

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
    payload = {
        "sample_indices": np.asarray(idxs, dtype=np.int64),
        "summary_metrics": {},
    }
    summary_metrics = {}
    for i, group in enumerate(groups[:4]):
        if not group:
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
        gt_coords = np.stack([pca.transform(gt_bt[vid_idx].cpu().numpy()) for vid_idx in group], axis=0)
        pred_coords = np.stack([pca.transform(pred_bt[vid_idx].cpu().numpy()) for vid_idx in group], axis=0)
        payload[f"group_{i}_gt_coords"] = gt_coords
        payload[f"group_{i}_pred_coords"] = pred_coords
        payload[f"group_{i}_explained_variance_ratio"] = pca.explained_variance_ratio_
        summary_metrics.update(_trajectory_summary_metrics(gt_coords, pred_coords, f"val/diag/trajectory/phase_space/group_{i}"))
    payload["summary_metrics"] = summary_metrics
    return fig, payload


def geometry_visualization_loop(
    batch,
    jepa,
    device,
    geometry_cfg=None,
    detection_targets=None,
    epoch=None,
    covariance_diagnostics=None,
    bundle=None,
    use_amp=False,
    dtype=torch.float32,
):
    _ = device
    if bundle is None:
        bundle = _build_rollout_latents(batch, jepa, use_amp=use_amp, dtype=dtype)
    gt_latent = bundle["gt_latent"]
    pred_rollout = bundle["pred_rollout"]

    num_samples = int(_cfg_get(geometry_cfg, "num_samples", 12))
    tsne_method = str(_cfg_get(geometry_cfg, "tsne_method", "umap"))
    act_layer = _cfg_get(geometry_cfg, "activation_layer", None)
    stage_label = str(_cfg_get(geometry_cfg, "stage_label", "encoder_rollout"))
    include_epoch_in_title = bool(_cfg_get(geometry_cfg, "include_epoch_in_title", True))
    tsne_max_points = int(_cfg_get(geometry_cfg, "spatial_tsne_max_points", 10000))
    cov_source_mode = _covariance_source_mode(geometry_cfg)
    tss_num_samples = int(_cfg_get(geometry_cfg, "temporal_self_similarity_num_samples", 2))
    tss_feature_mode = str(_cfg_get(geometry_cfg, "temporal_self_similarity_feature_mode", "spatialflat"))
    tss_distance = str(_cfg_get(geometry_cfg, "temporal_self_similarity_distance", "euclidean"))
    tss_pca_fixed_k = int(_cfg_get(geometry_cfg, "temporal_self_similarity_pca_fixed_k", 3))
    tss_pca_var_ratio = float(_cfg_get(geometry_cfg, "temporal_self_similarity_pca_var_ratio", 0.95))
    tss_pca_max_k = int(_cfg_get(geometry_cfg, "temporal_self_similarity_pca_max_k", 64))
    tss_center_time = bool(_cfg_get(geometry_cfg, "temporal_self_similarity_center_time", True))

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
    raw_payloads = {}
    scalar_metrics = {}

    if _plot_enabled(geometry_cfg, "latent_trajectories_avgpool"):
        fig, payload = plot_latent_trajectories_avgpool(
            gt_seq,
            pred_seq,
            num_samples=num_samples,
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
        )
        figs["latent_trajectories_avgpool"] = fig
        raw_payloads["latent_trajectories_avgpool"] = payload
        scalar_metrics.update(payload.get("summary_metrics", {}))

    if _plot_enabled(geometry_cfg, "latent_trajectories_spatialflat"):
        fig, payload = plot_latent_trajectories_spatialflat(
            gt_seq,
            pred_seq,
            num_samples=num_samples,
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
        )
        figs["latent_trajectories_spatialflat"] = fig
        raw_payloads["latent_trajectories_spatialflat"] = payload
        scalar_metrics.update(payload.get("summary_metrics", {}))

    if _plot_enabled(geometry_cfg, "temporal_self_similarity"):
        fig, payload = plot_temporal_self_similarity(
            gt_seq,
            num_samples=tss_num_samples,
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
            feature_mode=tss_feature_mode,
            distance=tss_distance,
            pca_fixed_k=tss_pca_fixed_k,
            pca_var_ratio=tss_pca_var_ratio,
            pca_max_k=tss_pca_max_k,
            center_time=tss_center_time,
        )
        figs["temporal_self_similarity"] = fig
        raw_payloads["temporal_self_similarity"] = payload
        scalar_metrics.update(payload.get("summary_metrics", {}))

    if _plot_enabled(geometry_cfg, "occupancy_spatial_embedding"):
        fig, payload = plot_occupancy_spatial_embedding(
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
        figs["occupancy_spatial_embedding"] = fig
        raw_payloads["occupancy_spatial_embedding"] = payload
        scalar_metrics.update(payload.get("summary_metrics", {}))
        if long_enabled and gt_latent.shape[2] > max_frames_for_embedding:
            long_used = True
            long_msgs.append(
                f"embed[{time_subsample_mode}]:{gt_latent.shape[2]}->{min(window_size, max_frames_for_embedding) if time_subsample_mode == 'windowed' else max_frames_for_embedding}"
            )

    if _plot_enabled(geometry_cfg, "cov_eig_spectrum"):
        cov_results = covariance_diagnostics
        if cov_results is None:
            trackers = {}
            update_covariance_trackers(trackers, gt_latent, jepa, source_mode=cov_source_mode)
            cov_results = finalize_covariance_diagnostics(trackers)
        for key, info in cov_results.items():
            fig, payload = plot_cov_eig_spectrum_from_eigvals(
                info["eigvals"],
                plot_label=_covariance_plot_title(key),
                stage_label=stage_label,
                epoch=epoch,
                include_epoch_in_title=include_epoch_in_title,
            )
            figs[f"cov_eig_spectrum_{key}"] = fig
            raw_payloads[f"cov_eig_spectrum_{key}"] = {**payload, "source_key": key, "sample_count": info.get("sample_count", 0)}

    if _plot_enabled(geometry_cfg, "activation_overlays"):
        fig, payload = plot_activation_overlays(
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
        figs["activation_overlays"] = fig
        raw_payloads["activation_overlays"] = payload
        scalar_metrics.update(payload.get("summary_metrics", {}))
        if long_enabled and batch["video"].shape[2] > max_frames_activation_overlay:
            long_used = True
            long_msgs.append(
                f"overlay[{time_subsample_mode}]:{batch['video'].shape[2]}->{min(window_size, max_frames_activation_overlay) if time_subsample_mode == 'windowed' else max_frames_activation_overlay}"
            )

    if _plot_enabled(geometry_cfg, "phase_space_portrait"):
        fig, payload = plot_phase_space_portrait(
            gt_seq,
            pred_seq,
            num_samples=min(4, batch["video"].shape[0]),
            stage_label=stage_label,
            epoch=epoch,
            include_epoch_in_title=include_epoch_in_title,
            aligned=True,
        )
        figs["phase_space_portrait"] = fig
        raw_payloads["phase_space_portrait"] = payload
        scalar_metrics.update(payload.get("summary_metrics", {}))

    meta = {
        "long_sequence_enabled": long_enabled,
        "long_sequence_mode": time_subsample_mode,
        "long_sequence_used": long_used,
        "long_sequence_details": ", ".join(long_msgs) if long_msgs else "",
    }
    return {k: v for k, v in figs.items() if v is not None}, meta, raw_payloads, scalar_metrics


def log_and_save_geometry_viz(
    figures,
    exp_dir,
    epoch,
    wandb_prefix="geometry_viz",
    include_epoch_in_filename=True,
    log_to_wandb=True,
):
    exp_dir = Path(exp_dir)
    epoch_dir = exp_dir / "diagnostics" / "media" / "geometry_viz" / f"epoch_{int(epoch):04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    logs = {}
    media_refs = {}
    for key, fig in figures.items():
        if include_epoch_in_filename:
            out_path = epoch_dir / f"{key}_epoch_{int(epoch):04d}.png"
        else:
            out_path = epoch_dir / f"{key}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        media_refs[f"{wandb_prefix}/{key}"] = {
            "local_path": str(out_path),
            "wandb_key": f"{wandb_prefix}/{key}",
            "kind": "image",
        }
        if log_to_wandb:
            logs[f"{wandb_prefix}/{key}"] = wandb.Image(str(out_path), caption=f"epoch={int(epoch)}")
    return logs, media_refs


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
    base = exp_dir / "diagnostics" / "media" / "geometry_viz"
    if not base.exists():
        legacy_base = exp_dir / "geometry_viz"
        if not legacy_base.exists():
            return {}
        base = legacy_base

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

        # Unify all frames to the same canvas size before writing.
        # Different checkpoints may produce plots with different heights/widths
        # (e.g. when the number of covariance sources changes over training).
        max_h = max(f.shape[0] for f in frames)
        max_w = max(f.shape[1] for f in frames)

        def _pad_to_canvas(frame, target_h, target_w):
            h, w = frame.shape[:2]
            if h == target_h and w == target_w:
                return frame
            pad = ((0, target_h - h), (0, target_w - w), (0, 0)) if frame.ndim == 3 else ((0, target_h - h), (0, target_w - w))
            return np.pad(frame, pad, mode="edge")

        video_path = base / f"{key}_evolution.mp4"
        with imageio.get_writer(video_path, fps=fps) as writer:
            for fr in frames:
                fr = _pad_to_canvas(fr, max_h, max_w)
                writer.append_data(_pad_frame_to_macroblock(fr))  # type: ignore[attr-defined]

        logs[f"{wandb_prefix}/{key}_evolution"] = wandb.Video(str(video_path), format="mp4")

    return logs
