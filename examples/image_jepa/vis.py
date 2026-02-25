"""
Visualization utilities for image_jepa (image/CIFAR-10 specific).

Designed to work both:
  - From the training loop in main.py (called periodically during training)
  - From a Jupyter/Colab notebook (import and call directly after loading a checkpoint)

All functions accept standard PyTorch objects and optionally log to wandb.

Usage from notebook:
    from examples.image_jepa.vis import visualization_loop
    figs = visualization_loop(model, linear_probe, val_loader, device,
                               save_dir="visualizations", wandb_run=None)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

# sklearn: required for t-SNE and confusion matrix
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Optional UMAP (graceful fallback to t-SNE if not installed)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Internal helper: single-pass feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_embeddings_and_preds(model, linear_probe, val_loader, device, use_amp=True):
    """
    Single forward pass over val_loader.

    Returns:
        embeddings: np.ndarray (N, D) — backbone features (float32)
        preds:      np.ndarray (N,)   — linear probe predicted class indices
        targets:    np.ndarray (N,)   — ground-truth class indices
    """
    model.eval()
    linear_probe.eval()

    all_embeddings, all_preds, all_targets = [], [], []

    for data, target in val_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp):
            features, _ = model(data)

        logits = linear_probe(features.float())
        preds = logits.argmax(dim=1)

        all_embeddings.append(features.float().cpu())
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return embeddings, preds, targets


# ---------------------------------------------------------------------------
# Internal helpers: multi-layer embedding extraction and layer discovery
# ---------------------------------------------------------------------------

# Default shorthand names → full dot-path inside ImageSSL (ResNet-18)
_RESNET_LAYER_PATHS = {
    "layer1": "backbone.backbone.layer1",
    "layer2": "backbone.backbone.layer2",
    "layer3": "backbone.backbone.layer3",
    "layer4": "backbone.backbone.layer4",
}

_DEFAULT_EMBEDDING_SOURCES = ["backbone"]
_DEFAULT_ACTIVATION_LAYERS = ["backbone.backbone.layer4"]
_DEFAULT_TSNE_METHODS       = ["tsne"]


def _resolve_module(model, dot_path):
    """Resolve a dot-path string to a submodule within model."""
    module = model
    for attr in dot_path.split("."):
        module = getattr(module, attr)
    return module


def _discover_spatial_layers(model, device):
    """
    Auto-discover all spatial (4-D output) layers via a dummy forward pass.
    Returns dot-paths ordered from shallowest to deepest.
    Works for any torchvision backbone — useful when switching to ResNet-50, ViT, etc.
    """
    found, handles = [], []

    def make_hook(name):
        def _h(mod, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                found.append(name)
        return _h

    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(make_hook(name)))

    dummy = torch.zeros(2, 3, 32, 32, device=device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass

    for h in handles:
        h.remove()

    # Deduplicate while preserving order
    seen, result = set(), []
    for name in found:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _resolve_embedding_sources(embedding_sources):
    """
    Normalize embedding_sources to a list of source labels.

    Valid labels:
      "backbone"          — GAP'd backbone output (post-layer4, post-GAP)
      "projector"         — projector MLP output
      "layer1".."layer4" — GAP'd intermediate ResNet feature maps

    Special values:
      None  → ["backbone"]                                         (default)
      "all" → ["layer1", "layer2", "layer3", "layer4", "backbone", "projector"]
    """
    if embedding_sources is None:
        return _DEFAULT_EMBEDDING_SOURCES[:]
    if embedding_sources == "all":
        return ["layer1", "layer2", "layer3", "layer4", "backbone", "projector"]
    return list(embedding_sources)


def _resolve_activation_layers(activation_layers, model=None, device=None):
    """
    Normalize activation_layers to a list of dot-paths.

    Special values:
      None  → ["backbone.backbone.layer4"]  (default)
      "all" → all spatial layers (auto-discovered via dummy forward pass)
    """
    if activation_layers is None:
        return _DEFAULT_ACTIVATION_LAYERS[:]
    if activation_layers == "all":
        if model is None or device is None:
            raise ValueError("model and device must be provided when activation_layers='all'")
        return _discover_spatial_layers(model, device)
    return list(activation_layers)


@torch.no_grad()
def _extract_all_embeddings(model, linear_probe, val_loader, device, sources, use_amp=True):
    """
    Extract embeddings from multiple sources in a single forward pass per batch.

    Intermediate spatial layers (layer1–4) are GAP'd to (N, C) vectors.
    "backbone" = post-GAP 512-d vector; "projector" = projector output.
    Linear probe predictions always come from backbone features.

    Args:
        sources: list of source labels (see _resolve_embedding_sources)

    Returns:
        embeddings_dict: dict {source_label: np.ndarray (N, D)}
        preds:           np.ndarray (N,)  — predicted class indices
        targets:         np.ndarray (N,)  — ground-truth class indices
    """
    model.eval()
    linear_probe.eval()

    hook_sources = [s for s in sources if s in _RESNET_LAYER_PATHS]
    buf          = {s: [] for s in hook_sources}
    all_backbone, all_projector, all_preds, all_targets = [], [], [], []

    for data, target in val_loader:
        data   = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Register per-batch hooks for intermediate layers
        batch_buf = {s: None for s in hook_sources}
        handles   = []
        for s in hook_sources:
            mod = _resolve_module(model, _RESNET_LAYER_PATHS[s])
            def _make_hook(src):
                def _h(m, i, o): batch_buf[src] = o.detach().cpu().float()
                return _h
            handles.append(mod.register_forward_hook(_make_hook(s)))

        with autocast(device.type, enabled=use_amp):
            features, projections = model(data)

        for h in handles:
            h.remove()

        # GAP spatial feature maps → (N, C) vectors
        for s in hook_sources:
            buf[s].append(batch_buf[s].mean(dim=(2, 3)))

        if "backbone"  in sources: all_backbone.append(features.float().cpu())
        if "projector" in sources: all_projector.append(projections.float().cpu())

        logits = linear_probe(features.float())
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(target.cpu())

    preds   = torch.cat(all_preds,   dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    embeddings_dict = {}
    for s in sources:
        if s in hook_sources:
            embeddings_dict[s] = torch.cat(buf[s], dim=0).numpy()
        elif s == "backbone":
            embeddings_dict[s] = torch.cat(all_backbone,  dim=0).numpy()
        elif s == "projector":
            embeddings_dict[s] = torch.cat(all_projector, dim=0).numpy()

    return embeddings_dict, preds, targets


# ---------------------------------------------------------------------------
# Vis 1: Latent Space Visualization (t-SNE / UMAP)
# ---------------------------------------------------------------------------

def plot_latent_tsne(
    embeddings,
    labels,
    class_names=None,
    save_path=None,
    wandb_run=None,
    method="tsne",
    title="Latent Space",
):
    """
    Visualize embeddings in 2D using t-SNE (default) or UMAP.

    Subsamples to 5000 points if dataset is large (t-SNE is O(N^2)).

    Args:
        embeddings:   np.ndarray (N, D)
        labels:       np.ndarray (N,) integer class indices
        class_names:  list of class name strings
        save_path:    optional PNG save path
        wandb_run:    optional wandb run object
        method:       "tsne" or "umap" (requires umap-learn)
        title:        plot title

    Returns:
        matplotlib Figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(labels)))]

    # Subsample if too many points (t-SNE is slow for N > 5000)
    max_samples = 5000
    if len(embeddings) > max_samples:
        idx = np.random.RandomState(42).choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # Dimensionality reduction
    if method == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        method_label = "UMAP"
    else:
        if method == "umap":
            print("[vis] umap-learn not installed, falling back to t-SNE.")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        reduced = tsne.fit_transform(embeddings)
        method_label = "t-SNE"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=[colors[i]], label=class_name, alpha=0.6, s=10,
        )

    ax.legend(markerscale=2, fontsize=9, loc="best")
    ax.set_title(f"{title} — {method_label}")
    ax.set_xlabel(f"{method_label}-1")
    ax.set_ylabel(f"{method_label}-2")
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if wandb_run is not None:
        import wandb
        key = f"vis/{title.lower().replace(' ', '_').replace('/', '_')}_{method_label.lower()}"
        wandb_run.log({key: wandb.Image(fig)}, commit=False)

    return fig


# ---------------------------------------------------------------------------
# Vis 2: Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    preds,
    targets,
    class_names=None,
    save_path=None,
    wandb_run=None,
    normalize=True,
    title="Confusion Matrix",
):
    """
    Plot a confusion matrix from precomputed predictions and targets.

    Args:
        preds:        np.ndarray (N,) predicted class indices
        targets:      np.ndarray (N,) ground-truth class indices
        class_names:  list of class name strings
        save_path:    optional PNG save path
        wandb_run:    optional wandb run object
        normalize:    if True, normalize rows to show per-class recall
        title:        plot title

    Returns:
        matplotlib Figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(targets)))]

    cm = confusion_matrix(targets, preds, normalize="true" if normalize else None)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if wandb_run is not None:
        import wandb
        key = f"vis/{title.lower().replace(' ', '_').replace('/', '_')}"
        wandb_run.log({key: wandb.Image(fig)}, commit=False)

    return fig


# ---------------------------------------------------------------------------
# Vis 3: Activation Maps (channel-averaged feature maps via forward hook)
# ---------------------------------------------------------------------------

def plot_activation_maps(
    model,
    val_loader,
    device,
    num_samples=8,
    layer_name="backbone.backbone.layer4",
    save_path=None,
    wandb_run=None,
    title="Activation Maps",
    use_amp=True,
):
    """
    Visualize spatial activation heatmaps from an intermediate ResNet layer,
    overlaid on the input image.

    Uses a forward hook to capture the last spatial feature map, averages
    across channels (simple saliency), and overlays the result as a
    jet-colormap heatmap on top of the original input image.

    Args:
        model:        ImageSSL model (ResNet backbone)
        val_loader:   validation DataLoader
        device:       torch device
        num_samples:  number of images to visualize (first N from first batch)
        layer_name:   dot-path to the backbone layer to hook
                      (default: "backbone.backbone.layer4" for ResNet-18)
        save_path:    optional PNG save path
        wandb_run:    optional wandb run object
        title:        plot title
        use_amp:      enable mixed precision for the forward pass

    Returns:
        matplotlib Figure
    """
    model.eval()
    activations = {}

    def _hook(module, input, output):
        # output: (N, C, H', W')
        activations["feat"] = output.detach().cpu().float()

    # Resolve the layer by dot-path
    layer = model
    for attr in layer_name.split("."):
        layer = getattr(layer, attr)
    handle = layer.register_forward_hook(_hook)

    # One forward pass on the first batch
    data, _ = next(iter(val_loader))
    data = data[:num_samples].to(device)

    with torch.no_grad():
        with autocast(device.type, enabled=use_amp):
            model(data)

    handle.remove()

    feat_maps = activations["feat"]           # (N, C, H', W')
    saliency = feat_maps.mean(dim=1)          # (N, H', W') — average across channels

    # Normalize each saliency map independently to [0, 1]
    mins = saliency.flatten(1).min(dim=1).values[:, None, None]
    maxs = saliency.flatten(1).max(dim=1).values[:, None, None]
    saliency = (saliency - mins) / (maxs - mins + 1e-8)

    # Un-normalize CIFAR-10 images for display
    # (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    data_cpu = data.cpu().float()
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    imgs = (data_cpu * std + mean).clamp(0, 1)  # (N, 3, H, W)

    # Plot: row 0 = input images, row 1 = activation overlay
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    axes = np.array(axes)

    for i in range(num_samples):
        img_np = imgs[i].permute(1, 2, 0).numpy()      # (H, W, 3)
        sal_np = saliency[i].numpy()                    # (H', W')

        # Upsample saliency map to match input image size
        sal_up = F.interpolate(
            torch.tensor(sal_np)[None, None],
            size=img_np.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        # Row 0: original input
        axes[0, i].imshow(img_np)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Input", fontsize=8)

        # Row 1: heatmap overlay
        axes[1, i].imshow(img_np)
        axes[1, i].imshow(sal_up, cmap="jet", alpha=0.5)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Activations", fontsize=8)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if wandb_run is not None:
        import wandb
        key = f"vis/{title.lower().replace(' ', '_').replace('/', '_')}"
        wandb_run.log({key: wandb.Image(fig)})

    return fig


# ---------------------------------------------------------------------------
# Main entry point: visualization_loop
# ---------------------------------------------------------------------------

def visualization_loop(
    model,
    linear_probe,
    val_loader,
    device,
    class_names=None,
    use_amp=True,
    save_dir=None,
    wandb_run=None,
    epoch=None,
    tsne_method="tsne",           # kept for backward compat; ignored when tsne_methods is set
    # --- Advanced options (all default to None = single default) ---
    embedding_sources=None,       # None→["backbone"], "all", or list of source names
                                  # valid names: "backbone", "projector",
                                  #              "layer1".."layer4"
    activation_layers=None,       # None→["backbone.backbone.layer4"], "all", or list
    tsne_methods=None,            # None→["tsne"], or ["tsne", "umap"]
):
    """
    Run all image_jepa visualizations.

    Default behaviour (all advanced options None) is identical to before:
      - one t-SNE plot of backbone features
      - one confusion matrix
      - one activation map from layer4
    Keys in the returned dict are also unchanged: "tsne", "confusion", "activation".

    Advanced options allow multi-layer / multi-method exploration from a notebook
    without affecting training runs:

        figs = visualization_loop(
            ...,
            embedding_sources=["layer2", "layer4", "backbone", "projector"],
            activation_layers=["backbone.backbone.layer2", "backbone.backbone.layer4"],
            tsne_methods=["tsne", "umap"],
        )
        # or use "all" to auto-discover:
        figs = visualization_loop(..., embedding_sources="all", activation_layers="all")

    When advanced options produce multiple items the dict keys encode what was used:
      "tsne_backbone_tsne", "tsne_layer2_umap", "activation_layer2", ...

    Args:
        model:              ImageSSL model
        linear_probe:       LinearProbe classifier
        val_loader:         validation DataLoader
        device:             torch device
        class_names:        list of class names (defaults to CIFAR-10)
        use_amp:            enable mixed precision
        save_dir:           directory to save PNGs (None = no file output)
        wandb_run:          wandb run object (None = no wandb logging)
        epoch:              current epoch (appended to filenames / wandb keys)
        tsne_method:        "tsne" or "umap" — backward-compat single-method param
        embedding_sources:  see above
        activation_layers:  see above
        tsne_methods:       see above

    Returns:
        dict of matplotlib Figures
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    # Resolve the three option lists
    sources    = _resolve_embedding_sources(embedding_sources)
    act_layers = _resolve_activation_layers(activation_layers, model=model, device=device)
    methods    = list(tsne_methods) if tsne_methods is not None else [tsne_method]

    # Detect default mode: controls backward-compat key + filename format
    is_default = (
        sources    == _DEFAULT_EMBEDDING_SOURCES and
        act_layers == _DEFAULT_ACTIVATION_LAYERS and
        methods    == _DEFAULT_TSNE_METHODS
    )

    suffix   = f"_epoch{epoch:04d}" if epoch is not None else ""
    save_dir = Path(save_dir) if save_dir is not None else None

    # --- Single pass: embeddings from all requested sources + preds + targets ---
    embeddings_dict, preds, targets = _extract_all_embeddings(
        model, linear_probe, val_loader, device, sources, use_amp=use_amp
    )

    figs = {}

    # --- Vis 1: Latent space (t-SNE / UMAP) — one plot per (source, method) ---
    for source in sources:
        for method in methods:
            if is_default:
                # Backward-compat: same short key and filename as before
                fig_key = "tsne"
                fname   = f"latent_{method}{suffix}.png"
                title   = f"Latent Space{suffix}"
            else:
                # type_source_method_epoch order
                fig_key = f"tsne_{source}_{method}"
                fname   = f"latent_{source}_{method}{suffix}.png"
                title   = f"Latent Space — {source}{suffix}"

            figs[fig_key] = plot_latent_tsne(
                embeddings_dict[source], targets,
                class_names=class_names,
                save_path=str(save_dir / fname) if save_dir else None,
                wandb_run=wandb_run,
                method=method,
                title=title,
            )

    # --- Vis 2: Confusion matrix (always one; preds from backbone features) ---
    figs["confusion"] = plot_confusion_matrix(
        preds, targets,
        class_names=class_names,
        save_path=str(save_dir / f"confusion_matrix{suffix}.png") if save_dir else None,
        wandb_run=wandb_run,
        title=f"Confusion Matrix{suffix}",
    )

    # --- Vis 3: Activation maps — one plot per requested layer ---
    for layer_path in act_layers:
        layer_label = layer_path.split(".")[-1]  # e.g. "layer4"

        if is_default:
            # Backward-compat: same short key and filename as before
            fig_key = "activation"
            fname   = f"activation_maps{suffix}.png"
            title   = f"Activation Maps{suffix}"
        else:
            # type_source_epoch order (no method for activation maps)
            fig_key = f"activation_{layer_label}"
            fname   = f"activation_{layer_label}{suffix}.png"
            title   = f"Activation Maps — {layer_label}{suffix}"

        figs[fig_key] = plot_activation_maps(
            model, val_loader, device,
            layer_name=layer_path,
            save_path=str(save_dir / fname) if save_dir else None,
            wandb_run=wandb_run,
            title=title,
            use_amp=use_amp,
        )

    # Close figures to free memory during training.
    # Skipped when both save_dir and wandb_run are None (notebook display).
    if wandb_run is not None or save_dir is not None:
        for fig in figs.values():
            plt.close(fig)

    return figs


# ---------------------------------------------------------------------------
# Checkpoint-based entry point (for notebook / ad-hoc use)
# ---------------------------------------------------------------------------

def visualize_from_checkpoint(
    ckpt_path,
    cfg_path=None,
    save_dir=None,
    wandb_run=None,
    tsne_method="tsne",           # backward-compat single-method param
    # --- Advanced options; passed directly to visualization_loop ---
    embedding_sources=None,       # None→["backbone"], "all", or list
    activation_layers=None,       # None→["layer4"], "all", or list of dot-paths
    tsne_methods=None,            # None→["tsne"], or ["tsne", "umap"]
):
    """
    Load a checkpoint and run all visualizations in one call.

    cfg_path is optional — if not provided, config.yaml is auto-discovered
    next to the checkpoint file (saved there automatically during training).

    save_dir defaults to <exp_dir>/visualizations/<checkpoint_stem>, e.g.:
        .../resnet_bcs_seed42/visualizations/latest/
        .../resnet_bcs_seed42/visualizations/epoch_0020/

    Simple usage (same as always):
        figs = visualize_from_checkpoint(ckpt_path=".../latest.pth.tar")
        figs["tsne"].show()
        figs["confusion"].show()
        figs["activation"].show()

    Advanced usage (multi-layer, multi-method, no file saving for inline display):
        figs = visualize_from_checkpoint(
            ckpt_path=".../latest.pth.tar",
            save_dir=None,                             # keep figures open for .show()
            embedding_sources=["layer2", "layer3", "layer4", "backbone", "projector"],
            activation_layers=["backbone.backbone.layer2", "backbone.backbone.layer4"],
            tsne_methods=["tsne", "umap"],
        )
        figs["tsne_backbone_tsne"].show()
        figs["tsne_layer2_umap"].show()
        figs["activation_layer2"].show()

        # Or auto-discover all layers (useful for larger backbones):
        figs = visualize_from_checkpoint(
            ckpt_path=".../latest.pth.tar",
            save_dir=None,
            embedding_sources="all",
            activation_layers="all",
        )
    """
    # Lazy imports — use model.py to avoid pulling in training-only deps (fire, wandb...)
    from examples.image_jepa.model import ImageSSL, ResNet18
    from examples.image_jepa.eval import LinearProbe
    from examples.image_jepa.dataset import get_val_transforms
    from eb_jepa.training_utils import load_config, load_checkpoint
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
    import os

    ckpt_path = Path(ckpt_path)

    # Default save_dir: <exp_dir>/visualizations/<checkpoint_stem>
    # Checkpoints are always saved as .pth.tar → strip both extensions. e.g. latest.pth.tar → latest, epoch_0020.pth.tar → epoch_0020
    if save_dir is None:
        ckpt_stem = Path(ckpt_path.stem).stem  # strips .tar then .pth
        save_dir = ckpt_path.parent / "visualizations" / ckpt_stem

    # Auto-discover config.yaml next to checkpoint if not provided
    if cfg_path is None:
        cfg_path = ckpt_path.parent / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No config.yaml found next to checkpoint at {ckpt_path.parent}.\n"
                "Either provide cfg_path explicitly or re-run training to save config.yaml."
            )

    # 1. Load config
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Build model and linear probe
    from examples.image_jepa.model import build_model, build_linear_probe
    model, features_dim = build_model(cfg)
    model = model.to(device)

    # 3. Build linear probe
    linear_probe = build_linear_probe(features_dim).to(device)

    # 4. Load checkpoint weights
    ckpt_info = load_checkpoint(ckpt_path, model, optimizer=None, device=device)
    if "linear_probe_state_dict" in ckpt_info:
        linear_probe.load_state_dict(ckpt_info["linear_probe_state_dict"])
    epoch = ckpt_info.get("epoch", 0)
    print(f"Loaded checkpoint from epoch {epoch}")

    # 5. Build val loader
    data_dir = os.environ.get("EBJEPA_DSETS", ".")
    val_dataset = CIFAR10(root=data_dir, train=False, download=True,
                          transform=get_val_transforms())
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # 6. Run visualizations
    return visualization_loop(
        model=model,
        linear_probe=linear_probe,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        wandb_run=wandb_run,
        epoch=epoch,
        tsne_method=tsne_method,
        embedding_sources=embedding_sources,
        activation_layers=activation_layers,
        tsne_methods=tsne_methods,
    )
