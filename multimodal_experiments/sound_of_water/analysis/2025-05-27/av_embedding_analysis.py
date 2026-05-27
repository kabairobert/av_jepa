"""
AV Embedding Analysis — Sound of Water Dataset
=================================================
Grounded in B08/B09 findings:
  - k/m_unique ratio is the dominant difficulty parameter
  - L2/isotropic prior works better for MLP-warped (neural) representations
  - Predictor is necessary; marginal regularisation alone is insufficient

This script produces 3 plots:
  Plot 1: Joint PCA + CKA — do audio/video share structure?
           → estimates empirical k (shared dims) and k/d ratio
  Plot 2: Temporal alignment per video (Spearman ρ, audio PC1 ↔ video PC1)
           → tests whether k>0 is recoverable per-video/condition
  Plot 3: Eigenspectrum + Participation Ratio (PR)
           → diagnoses which D-regime (D0-D3) real AV encoders fall in

Models:
  Audio: EAT-base (worstchan/EAT-base_epoch30_pretrain) — layer 7 intermediate
  Video: V-JEPA 2.1 ViT-B (facebookresearch/vjepa2) — block 7 intermediate

Usage:
  source .venv/bin/activate
  python av_embedding_analysis.py [--skip-extraction] [--layer 7]

Outputs (to analysis/2025-05-27/):
  embeddings/          .npy cache files (audio_L7, video_L7 per video)
  plot1_cka_joint_pca.png
  plot1_cka_joint_pca.html
  plot2_temporal_alignment.png
  plot3_eigenspectrum_pr.png
  metrics_summary.json
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[4]  # eb_jepa_private/
VIDEO_DIR = REPO_ROOT / "multimodal_experiments/sound_of_water/videos"
OUT_DIR = Path(__file__).parent
EMB_DIR = OUT_DIR / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

LAYER_IDX = 7          # intermediate block index (0-based)
AUDIO_SR = 16_000      # EAT expects 16kHz
AUDIO_CHUNK_MS = 100   # 100ms chunks → 1600 samples per chunk
VIDEO_FPS = 10         # sample video at 10fps for visual embeddings

# 6 videos with their B09-relevant metadata (from localisation.csv + containers.yaml)
VIDEO_META = {
    "VID_20240116_230040_2.1_16.7.mp4": {
        "label": "V1",
        "container": "container_1",
        "material": "plastic",
        "shape": "cylindrical",
        "setting": "ws-kitchen",
        "flow_rate": "constant",
        "clean": True,
    },
    "VID_20240118_100817_2.8_23.6.mp4": {
        "label": "V2",
        "container": "container_2",
        "material": "plastic",
        "shape": "cylindrical",  # approx parabolic
        "setting": "ws-kitchen",
        "flow_rate": "constant",
        "clean": True,
    },
    "VID_20240118_221233_1.4_24.3.mp4": {
        "label": "V3",
        "container": "container_10",
        "material": "glass",
        "shape": "cylindrical",
        "setting": "ws-kitchen",
        "flow_rate": "constant",
        "clean": False,
    },
    "VID_20240122_001417_2.3_15.3.mp4": {
        "label": "V4",
        "container": "container_15",
        "material": "glass",
        "shape": "semiconical",
        "setting": "ws-kitchen",
        "flow_rate": "constant",
        "clean": True,
    },
    "VID_20240131_201458_3.7_24.4.mp4": {
        "label": "V5",
        "container": "container_18",
        "material": "glass",
        "shape": "cylindrical",
        "setting": "ws-room",   # ← different setting
        "flow_rate": "constant",
        "clean": True,
    },
    "VID_20240211_204115_2.6_15.0.mp4": {
        "label": "V6",
        "container": "container_29",
        "material": "plastic",
        "shape": "cylindrical",  # small chutney
        "setting": "ws-kitchen",
        "flow_rate": "constant",
        "clean": False,
    },
    "VID_20240211_204339_3.3_17.0.mp4": {
        "label": "V7",
        "container": "container_31",
        "material": "plastic_pet",
        "shape": "semiconical",
        "setting": "ws-kitchen",
        "flow_rate": "constant",
        "clean": True,
    },
}

# Matplotlib style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})

COLORS = {
    "glass":      "#4C8BF5",   # blue
    "plastic":    "#F5844C",   # orange
    "plastic_pet":"#9C6FDE",   # purple
}
MARKERS = {
    "cylindrical":  "o",
    "semiconical":  "^",
}
SETTINGS = {
    "ws-kitchen": "solid",
    "ws-room":    "dashed",
}


# ──────────────────────────────────────────────
# 1. Model Loading
# ──────────────────────────────────────────────

def load_eat_model():
    """EAT-base (audio ViT) from HuggingFace."""
    from transformers import AutoModel
    print("Loading EAT-base audio model...")
    model = AutoModel.from_pretrained(
        "worstchan/EAT-base_epoch30_pretrain",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


def load_vjepa_model():
    """V-JEPA 2.1 ViT-B from torch hub."""
    print("Loading V-JEPA 2.1 ViT-B model...")
    hub_model = torch.hub.load(
        'facebookresearch/vjepa2',
        'vjepa2_1_vit_base_384',
        pretrained=False,
        verbose=False,
    )
    vjepa_model = hub_model[0]

    checkpoint_url = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
    ckpt = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", weights_only=False)

    if 'encoder' in ckpt:
        raw = ckpt['encoder']
        sd = {k.replace("module.backbone.", ""): v for k, v in raw.items()}
        vjepa_model.load_state_dict(sd, strict=False)
    else:
        vjepa_model.load_state_dict(ckpt, strict=False)

    vjepa_model.eval()
    return vjepa_model


# ──────────────────────────────────────────────
# 2. Embedding Extraction — Audio (EAT-L7)
# ──────────────────────────────────────────────

def load_audio_torchaudio(video_path: Path, target_sr: int = 16_000):
    """Load audio from mp4 via torchaudio, resample to target_sr."""
    import torchaudio
    waveform, sr = torchaudio.load(str(video_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # [T]


def extract_eat_layer(video_path: Path, model, layer_idx: int = 7) -> np.ndarray:
    """
    Extract EAT intermediate representation at block `layer_idx`.
    Adapted from colab extract_layer_embeddings_v3.
    Input: 100ms chunks → [1, 1, 16, 100] for EAT's Conv2d local encoder.
    Returns: [N_chunks, D] where D = model hidden dim (768).
    """
    chunk_samples = AUDIO_CHUNK_MS * AUDIO_SR // 1000  # 1600

    waveform = load_audio_torchaudio(video_path)  # [T]
    core = model.model

    embeddings = []
    n_chunks = len(waveform) // chunk_samples
    for i in tqdm(range(0, len(waveform), chunk_samples),
                  total=n_chunks, desc=f"  EAT-L{layer_idx} {video_path.stem[:20]}", leave=False):
        chunk = waveform[i: i + chunk_samples]
        if len(chunk) < chunk_samples:
            continue

        x_in = chunk.float().view(1, 1, 16, 100)

        with torch.no_grad():
            # Patch embedding
            out = core.local_encoder(x_in)
            x = out[0] if isinstance(out, (tuple, list)) else out

            # Positional encoding
            if hasattr(core, 'fixed_positional_encoder'):
                pad_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool)
                pos_out = core.fixed_positional_encoder(x, padding_mask=pad_mask)
                x = pos_out[0] if isinstance(pos_out, (tuple, list)) else pos_out

            # Step through blocks up to layer_idx (inclusive)
            for j in range(layer_idx + 1):
                blk_out = core.blocks[j](x)
                x = blk_out[0] if isinstance(blk_out, (tuple, list)) else blk_out

            emb = x.mean(dim=1).squeeze(0).cpu().numpy()  # [D]
            embeddings.append(emb)

    return np.array(embeddings)  # [N, D]


# ──────────────────────────────────────────────
# 3. Embedding Extraction — Video (V-JEPA-L7)
# ──────────────────────────────────────────────

def extract_vjepa_layer(video_path: Path, model, layer_idx: int = 7,
                        fps_target: int = VIDEO_FPS) -> np.ndarray:
    """
    Extract V-JEPA intermediate representation at transformer block `layer_idx`.
    Uses torchvision VideoReader for frame sampling.
    Returns: [N_frames, D].
    """
    import torchvision.transforms.functional as TF
    from torchvision.io import VideoReader

    reader = VideoReader(str(video_path), "video")
    meta = reader.get_metadata()
    orig_fps = meta["video"]["fps"][0] if meta["video"]["fps"] else 30.0
    frame_interval = max(1, round(orig_fps / fps_target))

    # Preprocessing: resize to 224x224 (block 7 doesn't need full 384 spatial res)
    IMG_SIZE = 224
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    embeddings = []
    count = 0

    # Register a forward hook on block layer_idx to grab intermediate features
    captured = {}
    def hook_fn(module, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        captured['feat'] = x.detach()

    # Find the right blocks attribute
    if hasattr(model, 'blocks'):
        blocks = model.blocks
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'blocks'):
        blocks = model.encoder.blocks
    else:
        raise RuntimeError("Cannot find 'blocks' attribute on V-JEPA model")

    handle = blocks[layer_idx].register_forward_hook(hook_fn)

    try:
        reader.set_current_stream("video")
        pbar = tqdm(desc=f"  VJEPA-L{layer_idx} {video_path.stem[:20]}", leave=False)
        for frame in reader:
            if count % frame_interval == 0:
                img = frame["data"]  # [C, H, W] uint8
                img = img.float() / 255.0
                img = TF.resize(img, [IMG_SIZE, IMG_SIZE],
                                interpolation=TF.InterpolationMode.BILINEAR,
                                antialias=True)
                img = (img - mean) / std
                # V-JEPA expects [B, C, T, H, W]; T=1 for single frame
                inp = img.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]

                with torch.no_grad():
                    _ = model(inp)  # triggers hook

                feat = captured.get('feat')
                if feat is not None:
                    # [1, N_tokens, D] → mean pool over tokens
                    emb = feat.mean(dim=1).squeeze(0).cpu().numpy()
                    embeddings.append(emb)
                pbar.update(1)
            count += 1
        pbar.close()
    finally:
        handle.remove()

    return np.array(embeddings)  # [N, D]


# ──────────────────────────────────────────────
# 4. Cache I/O
# ──────────────────────────────────────────────

def cache_path(video_name: str, modality: str, layer: int) -> Path:
    stem = Path(video_name).stem
    return EMB_DIR / f"{stem}_{modality}_L{layer}.npy"


def load_or_extract_embeddings(videos, eat_model, vjepa_model, layer_idx):
    """Load from .npy cache or extract and save."""
    audio_embs, video_embs, labels, metas = [], [], [], []

    for vname, meta in videos.items():
        vpath = VIDEO_DIR / vname
        if not vpath.exists():
            print(f"  [SKIP] {vname} not found")
            continue

        # Audio
        a_path = cache_path(vname, "audio", layer_idx)
        if a_path.exists():
            a_emb = np.load(a_path)
            print(f"  [cache] audio {meta['label']}: {a_emb.shape}")
        else:
            print(f"  [extract] audio {meta['label']}...")
            a_emb = extract_eat_layer(vpath, eat_model, layer_idx)
            np.save(a_path, a_emb)
            print(f"    → saved {a_emb.shape}")

        # Video
        v_path = cache_path(vname, "video", layer_idx)
        if v_path.exists():
            v_emb = np.load(v_path)
            print(f"  [cache] video {meta['label']}: {v_emb.shape}")
        else:
            print(f"  [extract] video {meta['label']}...")
            v_emb = extract_vjepa_layer(vpath, vjepa_model, layer_idx)
            np.save(v_path, v_emb)
            print(f"    → saved {v_emb.shape}")

        audio_embs.append(a_emb)
        video_embs.append(v_emb)
        labels.append(meta['label'])
        metas.append(meta)

    return audio_embs, video_embs, labels, metas


# ──────────────────────────────────────────────
# 5. Metrics
# ──────────────────────────────────────────────

def participation_ratio(emb: np.ndarray) -> tuple[float, np.ndarray]:
    """
    PR = (sum λ_i)^2 / sum(λ_i^2)  where λ_i are PCA eigenvalues.
    Estimates effective dimensionality (number of active dimensions).
    Returns (PR, eigenvalues).
    """
    from sklearn.decomposition import PCA
    n_comp = min(emb.shape[0] - 1, emb.shape[1], 100)
    pca = PCA(n_components=n_comp)
    pca.fit(emb)
    lam = pca.explained_variance_
    pr = (lam.sum() ** 2) / (lam ** 2).sum()
    return float(pr), lam


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices X [N, Dx] and Y [N, Dy].
    Measures shared structure (0 = orthogonal, 1 = identical up to linear transform).
    Uses HSIC with linear kernel: HSIC(K,L) = <K_c, L_c>_F where K_c = X X^T centered.
    """
    # Centre
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T
    # Frobenius inner product (numerator)
    num = np.linalg.norm(K @ L, 'fro')  # equivalent to HSIC(K,L) up to 1/n^2
    denom = np.linalg.norm(K, 'fro') * np.linalg.norm(L, 'fro')
    return float(num / (denom + 1e-10))


def temporal_spearman(a_emb: np.ndarray, v_emb: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Spearman ρ between audio PC1 and video PC1, after temporal alignment (truncate).
    Returns (rho, audio_pc1_trajectory, video_pc1_trajectory).
    """
    from sklearn.decomposition import PCA
    min_len = min(len(a_emb), len(v_emb))
    a_sync = a_emb[:min_len]
    v_sync = v_emb[:min_len]

    a_pc1 = PCA(n_components=1).fit_transform(a_sync).flatten()
    v_pc1 = PCA(n_components=1).fit_transform(v_sync).flatten()

    rho, _ = spearmanr(a_pc1, v_pc1)
    return float(rho), a_pc1, v_pc1


# ──────────────────────────────────────────────
# 6. Plot 1 — Joint PCA + CKA
# ──────────────────────────────────────────────

def plot1_joint_pca_cka(audio_embs, video_embs, labels, metas, layer_idx, out_dir):
    """
    Project audio and video embeddings into a shared PCA space.
    Color = modality (audio/video). Shape = container shape. Border = material.
    Annotate with per-video CKA and global CKA.
    """
    from sklearn.decomposition import PCA

    print("\nPlot 1: Joint PCA + CKA...")

    # Align lengths: truncate each pair to shortest
    min_lens = [min(len(a), len(v)) for a, v in zip(audio_embs, video_embs)]
    a_aligned = [a[:n] for a, n in zip(audio_embs, min_lens)]
    v_aligned = [v[:n] for v, n in zip(video_embs, min_lens)]

    all_a = np.vstack(a_aligned)
    all_v = np.vstack(v_aligned)

    # Fit PCA on audio to project both (common reference frame)
    n_comp = 3
    pca = PCA(n_components=n_comp)
    pca.fit(all_a)
    a_pca = pca.transform(all_a)
    v_pca = pca.transform(all_v)

    # Per-video CKA
    per_vid_cka = []
    for a, v in zip(a_aligned, v_aligned):
        cka_val = linear_cka(a, v)
        per_vid_cka.append(cka_val)

    # Global CKA
    global_cka = linear_cka(all_a, all_v)

    var_exp = pca.explained_variance_ratio_

    # ── Static matplotlib figure ──
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_pc12 = fig.add_subplot(gs[0])
    ax_pc13 = fig.add_subplot(gs[1])
    ax_cka  = fig.add_subplot(gs[2])

    # Build offset arrays
    cumlen = np.concatenate([[0], np.cumsum(min_lens)])
    vid_colors_audio = "#4EABF5"
    vid_colors_video = "#F58B4E"

    for i, (label, meta) in enumerate(zip(labels, metas)):
        sl = slice(cumlen[i], cumlen[i+1])
        mark = MARKERS.get(meta['shape'], 'o')
        edge = COLORS.get(meta['material'], 'gray')
        alpha = 0.55

        # Audio points
        ax_pc12.scatter(a_pca[sl, 0], a_pca[sl, 1],
                        color=vid_colors_audio, marker=mark,
                        edgecolors=edge, linewidths=0.8,
                        alpha=alpha, s=18, label=f"{label}-audio" if i == 0 else "")
        ax_pc13.scatter(a_pca[sl, 0], a_pca[sl, 2],
                        color=vid_colors_audio, marker=mark,
                        edgecolors=edge, linewidths=0.8,
                        alpha=alpha, s=18)

        # Video points
        ax_pc12.scatter(v_pca[sl, 0], v_pca[sl, 1],
                        color=vid_colors_video, marker=mark,
                        edgecolors=edge, linewidths=0.8,
                        alpha=alpha, s=18, label=f"{label}-video" if i == 0 else "")
        ax_pc13.scatter(v_pca[sl, 0], v_pca[sl, 2],
                        color=vid_colors_video, marker=mark,
                        edgecolors=edge, linewidths=0.8,
                        alpha=alpha, s=18)

        # Label cluster centroid
        a_c = a_pca[sl].mean(0)
        v_c = v_pca[sl].mean(0)
        ax_pc12.annotate(label, (a_c[0], a_c[1]), fontsize=7.5, color="#4EABF5",
                         ha='center', va='bottom', fontweight='bold')
        ax_pc12.annotate(label, (v_c[0], v_c[1]), fontsize=7.5, color="#F58B4E",
                         ha='center', va='top', fontweight='bold')

    ax_pc12.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=10)
    ax_pc12.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=10)
    ax_pc12.set_title("Audio (blue) vs Video (orange)\nin Audio PCA space", fontsize=11, fontweight='bold')

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#4EABF5", markersize=8, label='Audio (EAT-L7)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#F58B4E", markersize=8, label='Video (VJEPA-L7)'),
        Line2D([0], [0], marker='o', color='gray', markeredgecolor='#4C8BF5', markerfacecolor='none', markersize=8, label='glass'),
        Line2D([0], [0], marker='o', color='gray', markeredgecolor='#F5844C', markerfacecolor='none', markersize=8, label='plastic'),
        Line2D([0], [0], marker='o', color='gray', markeredgecolor='#9C6FDE', markerfacecolor='none', markersize=8, label='plastic_pet'),
        Line2D([0], [0], marker='^', color='gray', markerfacecolor='none', markersize=8, label='semiconical'),
    ]
    ax_pc12.legend(handles=legend_elems, fontsize=7.5, loc='best', framealpha=0.5)

    ax_pc13.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=10)
    ax_pc13.set_ylabel(f"PC3 ({var_exp[2]*100:.1f}%)", fontsize=10)
    ax_pc13.set_title("PC1 vs PC3", fontsize=11, fontweight='bold')

    # CKA bar chart
    bar_colors = [COLORS.get(m['material'], 'gray') for m in metas]
    bars = ax_cka.bar(labels, per_vid_cka, color=bar_colors, edgecolor='white', linewidth=0.8)
    ax_cka.axhline(global_cka, color='black', linestyle='--', linewidth=1.5,
                   label=f'Global CKA = {global_cka:.3f}')
    for bar, val in zip(bars, per_vid_cka):
        ax_cka.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax_cka.set_ylim(0, max(per_vid_cka) * 1.25)
    ax_cka.set_xlabel("Video", fontsize=10)
    ax_cka.set_ylabel("Linear CKA (audio ↔ video)", fontsize=10)
    ax_cka.set_title(f"CKA per video\n(Global CKA = {global_cka:.3f})", fontsize=11, fontweight='bold')
    ax_cka.legend(fontsize=8.5)

    # Interpretation box
    if global_cka < 0.1:
        interp = "CKA < 0.1: modalities near-orthogonal\n→ alignment is D3-like (hardest regime)"
    elif global_cka < 0.3:
        interp = "CKA 0.1–0.3: weak shared structure\n→ D1/D3-like (sparse k)"
    else:
        interp = "CKA > 0.3: meaningful shared structure\n→ D0/D2-like regime"
    fig.text(0.5, -0.04, f"B09 context: {interp}", ha='center', fontsize=10,
             style='italic', color='#555')

    fig.suptitle(f"Plot 1 — Joint Audio-Video Embedding Space (EAT-L{layer_idx} + V-JEPA-L{layer_idx})",
                 fontsize=13, fontweight='bold', y=1.01)

    png_path = out_dir / "plot1_cka_joint_pca.png"
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  → saved {png_path}")

    # ── Interactive Plotly HTML ──
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig_p = make_subplots(rows=1, cols=2,
                              subplot_titles=["Audio PCA Space (PC1/PC2)", "CKA per Video"],
                              column_widths=[0.65, 0.35])

        mat_to_color = {"glass": "#4C8BF5", "plastic": "#F5844C", "plastic_pet": "#9C6FDE"}
        shape_to_symbol = {"cylindrical": "circle", "semiconical": "triangle-up"}

        for i, (label, meta) in enumerate(zip(labels, metas)):
            sl = slice(cumlen[i], cumlen[i+1])
            col = mat_to_color.get(meta['material'], 'gray')
            sym = shape_to_symbol.get(meta['shape'], 'circle')
            ts = np.arange(min_lens[i]) * 0.1  # time in seconds

            fig_p.add_trace(go.Scatter(
                x=a_pca[sl, 0], y=a_pca[sl, 1],
                mode='markers',
                name=f"{label} Audio",
                marker=dict(color=ts, colorscale='Blues', size=5,
                            symbol=sym, line=dict(color=col, width=1)),
                text=[f"{label} t={t:.1f}s" for t in ts],
                hoverinfo='text',
                legendgroup=label,
            ), row=1, col=1)

            fig_p.add_trace(go.Scatter(
                x=v_pca[sl, 0], y=v_pca[sl, 1],
                mode='markers',
                name=f"{label} Video",
                marker=dict(color=ts, colorscale='Oranges', size=5,
                            symbol=sym, line=dict(color=col, width=1)),
                text=[f"{label} t={t:.1f}s" for t in ts],
                hoverinfo='text',
                legendgroup=label,
            ), row=1, col=1)

        # CKA bars
        fig_p.add_trace(go.Bar(
            x=labels, y=per_vid_cka,
            marker_color=[mat_to_color.get(m['material'], 'gray') for m in metas],
            text=[f"{v:.3f}" for v in per_vid_cka],
            textposition='outside',
            name='CKA',
            showlegend=False,
        ), row=1, col=2)
        fig_p.add_hline(y=global_cka, line_dash='dash', line_color='black',
                        annotation_text=f"Global CKA={global_cka:.3f}", row=1, col=2)

        fig_p.update_xaxes(title_text=f"PC1 ({var_exp[0]*100:.1f}%)", row=1, col=1)
        fig_p.update_yaxes(title_text=f"PC2 ({var_exp[1]*100:.1f}%)", row=1, col=1)
        fig_p.update_xaxes(title_text="Video", row=1, col=2)
        fig_p.update_yaxes(title_text="Linear CKA", row=1, col=2)

        fig_p.update_layout(
            title=f"Joint AV Embedding Space — EAT-L{layer_idx} + V-JEPA-L{layer_idx} | Global CKA = {global_cka:.3f}",
            height=550, width=1100,
        )
        html_path = out_dir / "plot1_cka_joint_pca.html"
        fig_p.write_html(str(html_path))
        print(f"  → saved {html_path}")
    except Exception as e:
        print(f"  [WARN] Plotly HTML skipped: {e}")

    return {
        "global_cka": global_cka,
        "per_video_cka": dict(zip(labels, per_vid_cka)),
        "pca_var_explained_top3": list(var_exp[:3].tolist()),
    }


# ──────────────────────────────────────────────
# 7. Plot 2 — Temporal Alignment
# ──────────────────────────────────────────────

def plot2_temporal_alignment(audio_embs, video_embs, labels, metas, layer_idx, out_dir):
    """
    Spearman ρ (audio PC1 ↔ video PC1) per video.
    Also show time-series for best and worst video to illustrate.
    """
    print("\nPlot 2: Temporal alignment...")

    rhos, a_trajs, v_trajs = [], [], []
    for a, v in zip(audio_embs, video_embs):
        rho, a_t, v_t = temporal_spearman(a, v)
        rhos.append(rho)
        a_trajs.append(a_t)
        v_trajs.append(v_t)

    best_idx = int(np.argmax(rhos))
    worst_idx = int(np.argmin(rhos))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: Spearman bar chart ──
    ax = axes[0]
    bar_colors = [COLORS.get(m['material'], 'gray') for m in metas]
    bars = ax.bar(labels, rhos, color=bar_colors, edgecolor='white', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(np.mean(rhos), color='black', linestyle='--', linewidth=1.2,
               label=f'Mean ρ = {np.mean(rhos):.3f}')
    for bar, rho in zip(bars, rhos):
        ypos = rho + 0.01 if rho >= 0 else rho - 0.03
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{rho:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xlabel("Video", fontsize=10)
    ax.set_ylabel("Spearman ρ  (audio PC1 ↔ video PC1)", fontsize=10)
    ax.set_title("Cross-modal temporal alignment\nper video", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    # Condition markers (setting)
    for i, meta in enumerate(metas):
        if meta['setting'] == 'ws-room':
            ax.bar(labels[i], rhos[i], color=bar_colors[i],
                   edgecolor='black', linewidth=2.5)  # thick border = ws-room

    # ── Panel B: Best video time-series ──
    ax2 = axes[1]
    a_t_norm = (a_trajs[best_idx] - a_trajs[best_idx].mean()) / (a_trajs[best_idx].std() + 1e-8)
    v_t_norm = (v_trajs[best_idx] - v_trajs[best_idx].mean()) / (v_trajs[best_idx].std() + 1e-8)
    t = np.arange(len(a_t_norm)) * 0.1  # 100ms steps

    ax2.plot(t, a_t_norm, color='#4EABF5', lw=1.5, alpha=0.85, label='Audio PC1')
    ax2.plot(t, v_t_norm, color='#F58B4E', lw=1.5, alpha=0.85, label='Video PC1')
    ax2.fill_between(t, a_t_norm, v_t_norm, alpha=0.12, color='gray')
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Normalised PC1", fontsize=10)
    ax2.set_title(f"Best: {labels[best_idx]} (ρ={rhos[best_idx]:.3f})\n"
                  f"[{metas[best_idx]['material']}, {metas[best_idx]['shape']}, {metas[best_idx]['setting']}]",
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)

    # ── Panel C: Worst video time-series ──
    ax3 = axes[2]
    a_t_norm2 = (a_trajs[worst_idx] - a_trajs[worst_idx].mean()) / (a_trajs[worst_idx].std() + 1e-8)
    v_t_norm2 = (v_trajs[worst_idx] - v_trajs[worst_idx].mean()) / (v_trajs[worst_idx].std() + 1e-8)
    t2 = np.arange(len(a_t_norm2)) * 0.1

    ax3.plot(t2, a_t_norm2, color='#4EABF5', lw=1.5, alpha=0.85, label='Audio PC1')
    ax3.plot(t2, v_t_norm2, color='#F58B4E', lw=1.5, alpha=0.85, label='Video PC1')
    ax3.fill_between(t2, a_t_norm2, v_t_norm2, alpha=0.12, color='gray')
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("Normalised PC1", fontsize=10)
    ax3.set_title(f"Worst: {labels[worst_idx]} (ρ={rhos[worst_idx]:.3f})\n"
                  f"[{metas[worst_idx]['material']}, {metas[worst_idx]['shape']}, {metas[worst_idx]['setting']}]",
                  fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9)

    # B09 note
    if abs(np.mean(rhos)) < 0.15:
        note = "Mean |ρ| < 0.15: shared temporal dynamics weak → k very small, D3-like regime"
    elif abs(np.mean(rhos)) < 0.35:
        note = "Mean |ρ| 0.15–0.35: moderate shared dynamics → D1/D2-like regime"
    else:
        note = "Mean |ρ| > 0.35: strong shared dynamics → D0/D2-like regime; alignment tractable"
    fig.text(0.5, -0.03, f"B09 context: {note}", ha='center', fontsize=10, style='italic', color='#555')

    fig.suptitle(f"Plot 2 — Cross-Modal Temporal Alignment (EAT-L{layer_idx} ↔ V-JEPA-L{layer_idx})",
                 fontsize=13, fontweight='bold')

    png_path = out_dir / "plot2_temporal_alignment.png"
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  → saved {png_path}")

    return {
        "spearman_rho_per_video": dict(zip(labels, [round(r, 4) for r in rhos])),
        "mean_rho": round(float(np.mean(rhos)), 4),
        "best_video": labels[best_idx],
        "worst_video": labels[worst_idx],
    }


# ──────────────────────────────────────────────
# 8. Plot 3 — Eigenspectrum + Participation Ratio
# ──────────────────────────────────────────────

def plot3_eigenspectrum_pr(audio_embs, video_embs, labels, metas, layer_idx, out_dir):
    """
    Eigenvalue decay curves + Participation Ratio (PR) for both modalities.
    PR estimates effective dimensionality → maps to B09 D-regime (D0=3d, D1=10d, D2=10d, D3=20d).
    """
    print("\nPlot 3: Eigenspectrum + Participation Ratio...")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=0.38)

    ax_spec_a = axes[0]
    ax_spec_v = axes[1]
    ax_pr     = axes[2]

    pr_audio, pr_video = [], []

    # Reference B09 participation ratios (analytical): D0=3, D1=10, D2=10, D3=20
    # Real encoders d=768; PR gives effective k analogue
    B09_D_labels = {"D0\n(k/d=66%)": 3, "D1\n(k/d=20%)": 10,
                    "D2\n(k/d=50%)": 10, "D3\n(k/d=25%)": 20}

    for i, (label, meta, a_emb, v_emb) in enumerate(zip(labels, metas, audio_embs, video_embs)):
        col = COLORS.get(meta['material'], 'gray')
        lsv = SETTINGS.get(meta['setting'], 'solid')

        a_pr, a_lam = participation_ratio(a_emb)
        v_pr, v_lam = participation_ratio(v_emb)
        pr_audio.append(a_pr)
        pr_video.append(v_pr)

        # Normalise eigenvalues to [0,1] for shape comparison
        a_lam_n = a_lam / a_lam.sum()
        v_lam_n = v_lam / v_lam.sum()

        k = np.arange(1, len(a_lam_n) + 1)
        ax_spec_a.plot(k, np.cumsum(a_lam_n), color=col, linestyle=lsv,
                       linewidth=1.5, alpha=0.75, label=label)
        k2 = np.arange(1, len(v_lam_n) + 1)
        ax_spec_v.plot(k2, np.cumsum(v_lam_n), color=col, linestyle=lsv,
                       linewidth=1.5, alpha=0.75, label=label)

    ax_spec_a.axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax_spec_a.text(2, 0.82, '80% variance', fontsize=8, color='black', alpha=0.6)
    ax_spec_a.set_xlabel("Principal components (rank)", fontsize=10)
    ax_spec_a.set_ylabel("Cumulative variance explained", fontsize=10)
    ax_spec_a.set_title(f"Audio (EAT-L{layer_idx})\nEigenspectrum", fontsize=11, fontweight='bold')
    ax_spec_a.legend(fontsize=8.5, loc='lower right')
    ax_spec_a.set_xlim(1, 50)

    ax_spec_v.axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax_spec_v.text(2, 0.82, '80% variance', fontsize=8, color='black', alpha=0.6)
    ax_spec_v.set_xlabel("Principal components (rank)", fontsize=10)
    ax_spec_v.set_ylabel("Cumulative variance explained", fontsize=10)
    ax_spec_v.set_title(f"Video (V-JEPA-L{layer_idx})\nEigenspectrum", fontsize=11, fontweight='bold')
    ax_spec_v.legend(fontsize=8.5, loc='lower right')
    ax_spec_v.set_xlim(1, 50)

    # ── PR bar chart with B09 D-regime reference lines ──
    x = np.arange(len(labels))
    width = 0.35
    bars_a = ax_pr.bar(x - width/2, pr_audio, width, label='Audio PR',
                       color='#4EABF5', edgecolor='white', linewidth=0.8)
    bars_v = ax_pr.bar(x + width/2, pr_video, width, label='Video PR',
                       color='#F58B4E', edgecolor='white', linewidth=0.8)

    for bar, val in zip(list(bars_a) + list(bars_v), pr_audio + pr_video):
        ax_pr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=7.5)

    # B09 reference lines (scaled to d=768 encoder: D3 = 20 effective dims → k/d ≈ 20/768 = 2.6%)
    # We draw the B09 D-regime PRs as context lines
    b09_refs = [(20, "B09-D3 (PR=20, k/d≈2.6%)", '#888', '--'),
                (10, "B09-D1/D2 (PR=10, k/d≈1.3%)", '#aaa', ':')]
    for pr_ref, lbl, col, ls in b09_refs:
        ax_pr.axhline(pr_ref, color=col, linestyle=ls, linewidth=1.2, label=lbl)

    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels(labels)
    ax_pr.set_xlabel("Video", fontsize=10)
    ax_pr.set_ylabel("Participation Ratio (PR)", fontsize=10)
    ax_pr.set_title("Effective dimensionality\n(PR = effective # active dims)", fontsize=11, fontweight='bold')
    ax_pr.legend(fontsize=8.5, loc='upper left')

    mean_a = np.mean(pr_audio)
    mean_v = np.mean(pr_video)
    d_enc = 768
    note = (f"Mean PR: audio={mean_a:.1f}/768 ({100*mean_a/d_enc:.1f}%), "
            f"video={mean_v:.1f}/768 ({100*mean_v/d_enc:.1f}%)\n"
            f"B09 D3 analogy: k/d=25% of ambient dims = {0.25*d_enc:.0f}. "
            f"Real PR gives: harder than D3 if PR < {0.25*d_enc:.0f}")
    fig.text(0.5, -0.05, f"B09 context: {note}", ha='center', fontsize=9.5, style='italic', color='#555')

    fig.suptitle(f"Plot 3 — Eigenspectrum & Participation Ratio (EAT-L{layer_idx} + V-JEPA-L{layer_idx})",
                 fontsize=13, fontweight='bold')

    png_path = out_dir / "plot3_eigenspectrum_pr.png"
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  → saved {png_path}")

    return {
        "pr_audio_per_video": dict(zip(labels, [round(p, 2) for p in pr_audio])),
        "pr_video_per_video": dict(zip(labels, [round(p, 2) for p in pr_video])),
        "mean_pr_audio": round(mean_a, 2),
        "mean_pr_video": round(mean_v, 2),
        "pr_audio_over_d": round(mean_a / d_enc, 4),
        "pr_video_over_d": round(mean_v / d_enc, 4),
    }


# ──────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AV embedding analysis — Sound of Water")
    parser.add_argument("--layer", type=int, default=LAYER_IDX,
                        help=f"Transformer block index to extract (default={LAYER_IDX})")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip model loading — use cached .npy embeddings only")
    args = parser.parse_args()

    layer = args.layer
    print(f"\n{'='*60}")
    print(f"AV Embedding Analysis | Layer={layer} | Videos dir: {VIDEO_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"{'='*60}\n")

    # ── Load models (or skip if cached) ──
    eat_model, vjepa_model = None, None
    if not args.skip_extraction:
        # Check if all cache files exist
        all_cached = all(
            cache_path(vn, "audio", layer).exists() and
            cache_path(vn, "video", layer).exists()
            for vn in VIDEO_META
            if (VIDEO_DIR / vn).exists()
        )
        if all_cached:
            print("[INFO] All embeddings cached. Use --skip-extraction next time to skip model loading.")
        eat_model = load_eat_model()
        vjepa_model = load_vjepa_model()
    else:
        print("[INFO] --skip-extraction: loading from .npy cache only")

    # ── Load / extract embeddings ──
    audio_embs, video_embs, labels, metas = load_or_extract_embeddings(
        VIDEO_META, eat_model, vjepa_model, layer
    )

    if len(audio_embs) == 0:
        print("[ERROR] No embeddings extracted. Check video paths.")
        sys.exit(1)

    print(f"\nLoaded {len(audio_embs)} videos:")
    for lbl, a, v in zip(labels, audio_embs, video_embs):
        print(f"  {lbl}: audio {a.shape}, video {v.shape}")

    # ── Run plots ──
    metrics = {"layer": layer}

    m1 = plot1_joint_pca_cka(audio_embs, video_embs, labels, metas, layer, OUT_DIR)
    metrics.update({"plot1": m1})

    m2 = plot2_temporal_alignment(audio_embs, video_embs, labels, metas, layer, OUT_DIR)
    metrics.update({"plot2": m2})

    m3 = plot3_eigenspectrum_pr(audio_embs, video_embs, labels, metas, layer, OUT_DIR)
    metrics.update({"plot3": m3})

    # ── Save metrics JSON ──
    json_path = OUT_DIR / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {json_path}")

    # ── Print take-home summary ──
    print("\n" + "="*60)
    print("TAKE-HOME SUMMARY")
    print("="*60)
    cka = m1["global_cka"]
    rho = m2["mean_rho"]
    pr_a = m3["mean_pr_audio"]
    pr_v = m3["mean_pr_video"]
    d = 768

    print(f"\n🔑 Global CKA (audio ↔ video) = {cka:.3f}")
    if cka < 0.1:
        print("   → Near-orthogonal. Very sparse shared dims. Hardest alignment regime (like B09-D3).")
    elif cka < 0.3:
        print("   → Weak overlap. k small relative to d. D1/D3-like. L2+L2 pred is the right choice.")
    else:
        print("   → Meaningful overlap. Alignment tractable. D0/D2-like.")

    print(f"\n🔑 Mean temporal Spearman ρ = {rho:.3f}")
    if abs(rho) < 0.15:
        print("   → Temporal dynamics barely correlated. k≈0 in PC1 direction.")
    elif abs(rho) < 0.35:
        print("   → Moderate. Some shared events visible in PC1.")
    else:
        print("   → Strong. PC1 is likely a shared semantic dimension.")

    print(f"\n🔑 Participation Ratio: audio={pr_a:.1f}/{d} ({100*pr_a/d:.1f}%), "
          f"video={pr_v:.1f}/{d} ({100*pr_v/d:.1f}%)")
    print(f"   B09-D3 analogy (k=5,d=20 → 25%) would correspond to {0.25*d:.0f} here.")
    if max(pr_a, pr_v) < 0.25 * d:
        print(f"   → Both encoders are BELOW D3 density. Expect alignment harder than anything B09 tested.")
    else:
        print(f"   → One or both encoders above D3 density. Some structure to work with.")

    print("\n✅ All plots saved. Run with --skip-extraction next time to skip model loading.\n")


if __name__ == "__main__":
    main()
