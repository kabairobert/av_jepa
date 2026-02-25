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
# Internal helpers: dynamic layer probe and naming
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class _LayerInfo:
    dot_path: str
    class_name: str       # module.__class__.__name__
    output_shape: tuple   # captured from dummy forward pass
    is_spatial: bool      # True when output is 4-D (N, C, H, W)


def _resolve_module(model, dot_path):
    """Resolve a dot-path string to a submodule within model."""
    module = model
    for attr in dot_path.split("."):
        module = getattr(module, attr)
    return module


def _probe_model(model, device, input_shape=(2, 3, 32, 32)):
    """
    Single dummy forward pass that records the output shape of every named
    module.  Returns an ordered dict ``{dot_path: _LayerInfo}``.

    Works for any backbone (ResNet, ViT, custom) — no hardcoding needed.
    """
    results: Dict[str, _LayerInfo] = {}
    handles: List = []

    def _make_hook(name, cls_name):
        def _h(mod, inp, out):
            if isinstance(out, torch.Tensor) and name not in results:
                results[name] = _LayerInfo(
                    dot_path=name,
                    class_name=cls_name,
                    output_shape=tuple(out.shape),
                    is_spatial=out.dim() == 4,
                )
        return _h

    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(_make_hook(name, mod.__class__.__name__)))

    dummy = torch.zeros(*input_shape, device=device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass

    for h in handles:
        h.remove()
    return results


def _naming_source_to_key(dot_path: str, class_name: str) -> str:
    """
    Filename/dict-key–safe string encoding both path and class.

    Examples:
        "backbone.backbone.layer4" + "Sequential" → "backbone_backbone_layer4_Sequential"
        "projector.2"              + "ReLU"        → "projector_2_ReLU"
    """
    return f"{dot_path.replace('.', '_')}_{class_name}"


def _naming_source_to_display(dot_path: str, class_name: str, output_shape: tuple) -> str:
    """
    Human-readable label used in plot titles.

    Examples:
        "backbone.backbone.layer4" shape (2,512,5,5) → "backbone.backbone.layer4 [Sequential | 512×5×5 → GAP'd]"
        "projector.2"              shape (2,2048)     → "projector.2 [ReLU | 2048-d]"
    """
    if len(output_shape) == 4:
        _, c, h, w = output_shape
        return f"{dot_path} [{class_name} | {c}\u00d7{h}\u00d7{w} → GAP'd]"
    d = output_shape[-1]
    return f"{dot_path} [{class_name} | {d}-d]"


def _meaningful_backbone_layers(probe: Dict[str, "_LayerInfo"]) -> List[str]:
    """
    Return block-level backbone outputs only: direct children of
    ``backbone.backbone.*`` (dot-path depth == 3).

    For a ResNet this gives: conv1, bn1, relu, maxpool, layer1, layer2,
    layer3, layer4, avgpool, fc — the semantic boundaries, not the internals
    of each BasicBlock.
    """
    return [
        p for p in probe
        if p.startswith("backbone.backbone.")
        and len(p.split(".")) == 3
    ]


def _meaningful_projector_layers(probe: Dict[str, "_LayerInfo"]) -> List[str]:
    """
    Return only the output of each complete MLP unit in the projector:
      - ReLU outputs  (end of a Linear→BN→ReLU unit)
      - the final Linear layer (no activation after it)

    Skips bare Linear and BatchNorm nodes that are mid-unit.
    Works for any depth MLP projector.
    """
    # All direct children of projector (depth == 2: "projector.<i>")
    proj_keys = [
        p for p in probe
        if p.startswith("projector.")
        and len(p.split(".")) == 2
    ]
    if not proj_keys:
        return []
    last = proj_keys[-1]
    return [
        p for p in proj_keys
        if probe[p].class_name == "ReLU" or p == last
    ]


def _resolve_layers(
    layers,
    model,
    device,
    probe: Optional[Dict[str, "_LayerInfo"]] = None,
) -> List[str]:
    """
    Normalize the *layers* argument to a concrete list of dot-paths.

    Shorthand strings
    -----------------
    ``None``             → last spatial layer (H > 1) in backbone.* (default)
    ``"all"``            → meaningful backbone layers + meaningful projector layers
    ``"backbone"``       → block-level backbone outputs only (conv1, layer1–4, avgpool…)
    ``"projector"``      → projector unit outputs only (ReLU ends + final Linear)
    ``"backbone_full"``  → every submodule under backbone.*
    ``"projector_full"`` → every submodule under projector.*
    ``"<prefix>"``       → arbitrary dot-path prefix filter (fallback)
    list                 → explicit dot-paths, returned as-is
    """
    if probe is None:
        probe = _probe_model(model, device)

    if isinstance(layers, list):
        return list(layers)

    if layers == "all":
        # Meaningful boundaries across the whole model
        result = _meaningful_backbone_layers(probe) + _meaningful_projector_layers(probe)
        return result if result else list(probe.keys())

    if layers == "backbone":
        result = _meaningful_backbone_layers(probe)
        if result:
            return result
        # Fallback for non-standard backbone nesting
        result = [p for p in probe if p.startswith("backbone.") and len(p.split(".")) == 2]
        return result if result else [p for p in probe if p.startswith("backbone.")]

    if layers == "projector":
        result = _meaningful_projector_layers(probe)
        if result:
            return result
        # Fallback: all direct projector children
        return [p for p in probe if p.startswith("projector.") and len(p.split(".")) == 2]

    if layers == "backbone_full":
        return [p for p in probe if p.startswith("backbone.")]

    if layers == "projector_full":
        return [p for p in probe if p.startswith("projector.")]

    if isinstance(layers, str):
        # Arbitrary prefix filter
        prefix = layers if layers.endswith(".") else layers + "."
        matched = [p for p in probe if p.startswith(prefix) or p == layers]
        if matched:
            return matched
        raise ValueError(
            f"layers={layers!r} matched no modules. "
            f"Available top-level namespaces: "
            + str(sorted({p.split(".")[0] for p in probe}))
        )

    # Default (layers is None): last spatial (H > 1) layer in backbone.*
    spatial_hgt1 = [
        p for p, info in probe.items()
        if info.is_spatial
        and info.output_shape[-2] > 1
        and p.startswith("backbone.")
    ]
    if spatial_hgt1:
        return [spatial_hgt1[-1]]

    # Fallback: last 4-D layer anywhere in backbone.*
    spatial_any = [p for p, info in probe.items()
                   if info.is_spatial and p.startswith("backbone.")]
    if spatial_any:
        return [spatial_any[-1]]

    # Last resort: very last module
    return [list(probe.keys())[-1]]


@torch.no_grad()
def _extract_all_embeddings(model, linear_probe, val_loader, device, dot_paths, probe, use_amp=True):
    """
    Extract embeddings from all *dot_paths* in a single forward pass per batch.

    Spatial layers (4-D output) are Global-Average-Pooled to (N, C) vectors.
    Linear-probe predictions are derived from the model's primary return value.

    Args:
        dot_paths: list of dot-path strings (from ``_resolve_layers``)
        probe:     pre-computed ``{dot_path: _LayerInfo}`` from ``_probe_model``

    Returns:
        embeddings_dict: ``{dot_path: np.ndarray (N, D)}``
        preds:           ``np.ndarray (N,)`` predicted class indices
        targets:         ``np.ndarray (N,)`` ground-truth class indices
    """
    model.eval()
    linear_probe.eval()

    buf: Dict[str, List] = {p: [] for p in dot_paths}
    all_preds, all_targets = [], []

    for data, target in val_loader:
        data   = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_buf = {p: None for p in dot_paths}  # filled by hooks before use
        handles = []
        for p in dot_paths:
            mod = _resolve_module(model, p)
            def _make_hook(path):
                def _h(m, i, o):
                    batch_buf[path] = o.detach().cpu().float()
                return _h
            handles.append(mod.register_forward_hook(_make_hook(p)))

        with autocast(device.type, enabled=use_amp):
            backbone_features, _ = model(data)

        for h in handles:
            h.remove()

        for p in dot_paths:
            t = batch_buf[p]
            assert t is not None, f"Hook for '{p}' did not fire"
            if probe[p].is_spatial:
                buf[p].append(t.mean(dim=(2, 3)))   # GAP → (N, C)
            else:
                buf[p].append(t)                     # already (N, D)

        logits = linear_probe(backbone_features.float())
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(target.cpu())

    preds   = torch.cat(all_preds,   dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    embeddings_dict = {p: torch.cat(buf[p], dim=0).numpy() for p in dot_paths}

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
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        reduced = reducer.fit_transform(embeddings)
        method_label = "UMAP"
    else:
        if method == "umap":
            print("[vis] umap-learn not installed, falling back to t-SNE.")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
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
    tsne_method="tsne",   # backward-compat single-method param
    layers=None,          # None→auto, "all", or list of dot-paths
    tsne_methods=None,    # None→[tsne_method], or ["tsne", "umap"]
):
    """
    Run all image_jepa visualizations.

    Default behaviour (``layers=None``, ``tsne_methods=None``) is identical to
    before: one t-SNE plot, one confusion matrix, one activation map.
    Returned dict keys are also unchanged: ``"tsne"``, ``"confusion"``,
    ``"activation"``.

    Advanced usage from a notebook::

        figs = visualization_loop(
            ...,
            layers=["backbone.backbone.layer2",
                    "backbone.backbone.layer4",
                    "projector.6"],
            tsne_methods=["tsne", "umap"],
        )
        # or auto-discover every named module:
        figs = visualization_loop(..., layers="all")

    When multiple layers / methods are requested the dict keys encode what was
    used: ``"tsne_backbone_backbone_layer4_Sequential_tsne"``,
    ``"activation_backbone_backbone_layer2_Sequential"``, etc.

    Every requested layer gets a t-SNE plot.  Activation maps are generated
    automatically for any layer whose output is 4-D *and* whose spatial
    dimensions are larger than 1×1 (i.e., not post-GAP).

    Args:
        model:        ImageSSL model (or any model whose forward returns
                      ``(backbone_features, projections)``)
        linear_probe: linear classifier used for confusion-matrix predictions
        val_loader:   validation DataLoader
        device:       torch device
        class_names:  list of class name strings (default: CIFAR-10)
        use_amp:      enable automatic mixed precision
        save_dir:     directory for PNG output (``None`` = no file output)
        wandb_run:    wandb run object (``None`` = no wandb logging)
        epoch:        current epoch number (appended to filenames / keys)
        tsne_method:  ``"tsne"`` or ``"umap"`` — backward-compat single param
        layers:       dot-path(s) to visualize.
                      ``None``              → auto (last backbone spatial layer)
                      ``"all"``             → meaningful backbone + projector layers
                      ``"backbone"``        → block-level backbone outputs (layer1–4, etc.)
                      ``"projector"``       → projector unit outputs (ReLU ends + final)
                      ``"backbone_full"``   → every backbone.* submodule
                      ``"projector_full"``  → every projector.* submodule
                      ``"<prefix>"``        → arbitrary dot-path prefix filter
                      list                  → explicit dot-paths
        tsne_methods: list of reduction methods — overrides ``tsne_method``

    Returns:
        dict of ``{key: matplotlib.figure.Figure}``
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    methods  = list(tsne_methods) if tsne_methods is not None else [tsne_method]
    suffix   = f"_epoch{epoch:04d}" if epoch is not None else ""
    save_dir = Path(save_dir) if save_dir is not None else None

    # Probe the model once to learn every named module's output shape/type
    probe = _probe_model(model, device)

    # Resolve which dot-paths to extract
    dot_paths = _resolve_layers(layers, model, device, probe=probe)

    # single_mode → backward-compat short keys ("tsne", "activation")
    single_mode = len(dot_paths) == 1 and len(methods) == 1

    # Single pass over val_loader: embeddings for all layers + preds + targets
    embeddings_dict, preds, targets = _extract_all_embeddings(
        model, linear_probe, val_loader, device, dot_paths, probe, use_amp=use_amp
    )

    figs = {}
    close_after = wandb_run is not None or save_dir is not None

    # --- Vis 1: Latent space (t-SNE / UMAP) — one plot per (layer, method) ---
    for dot_path in dot_paths:
        info = probe[dot_path]
        key  = _naming_source_to_key(dot_path, info.class_name)
        disp = _naming_source_to_display(dot_path, info.class_name, info.output_shape)

        for method in methods:
            if single_mode:
                fig_key = "tsne"
                fname   = f"latent_{method}{suffix}.png"
                title   = f"Latent Space{suffix}"
            else:
                fig_key = f"tsne_{key}_{method}"
                fname   = f"latent_{key}_{method}{suffix}.png"
                title   = f"Latent Space — {disp}{suffix}"

            fig = plot_latent_tsne(
                embeddings_dict[dot_path], targets,
                class_names=class_names,
                save_path=str(save_dir / fname) if save_dir else None,
                wandb_run=wandb_run,
                method=method,
                title=title,
            )
            figs[fig_key] = fig
            if close_after:
                plt.close(fig)

    # --- Vis 2: Confusion matrix (always one) ---
    fig = plot_confusion_matrix(
        preds, targets,
        class_names=class_names,
        save_path=str(save_dir / f"confusion_matrix{suffix}.png") if save_dir else None,
        wandb_run=wandb_run,
        title=f"Confusion Matrix{suffix}",
    )
    figs["confusion"] = fig
    if close_after:
        plt.close(fig)

    # --- Vis 3: Activation maps — spatial (4-D, H > 1) layers only ---
    for dot_path in dot_paths:
        info = probe[dot_path]
        # Skip flat or post-GAP (1×1) layers — activation maps are meaningless
        if not info.is_spatial:
            continue
        _, _, h, w = info.output_shape
        if h <= 1 and w <= 1:
            continue

        key  = _naming_source_to_key(dot_path, info.class_name)
        disp = _naming_source_to_display(dot_path, info.class_name, info.output_shape)

        if single_mode:
            fig_key = "activation"
            fname   = f"activation_maps{suffix}.png"
            title   = f"Activation Maps{suffix}"
        else:
            fig_key = f"activation_{key}"
            fname   = f"activation_{key}{suffix}.png"
            title   = f"Activation Maps — {disp}{suffix}"

        fig = plot_activation_maps(
            model, val_loader, device,
            layer_name=dot_path,
            save_path=str(save_dir / fname) if save_dir else None,
            wandb_run=wandb_run,
            title=title,
            use_amp=use_amp,
        )
        figs[fig_key] = fig
        if close_after:
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
    tsne_method="tsne",      # backward-compat single-method param
    # --- Advanced options; passed directly to visualization_loop ---
    layers=None,             # None→auto, "all", or list of dot-paths
    tsne_methods=None,       # None→[tsne_method], or ["tsne", "umap"]
):
    """
    Load a checkpoint and run all visualizations in one call.

    ``cfg_path`` is optional — if not provided, ``config.yaml`` is
    auto-discovered next to the checkpoint file (saved there automatically
    during training).

    ``save_dir`` defaults to ``<exp_dir>/visualizations/<checkpoint_stem>``,
    e.g.::

        .../resnet_bcs_seed42/visualizations/latest/
        .../resnet_bcs_seed42/visualizations/epoch_0020/

    Simple usage (same as always)::

        figs = visualize_from_checkpoint(ckpt_path=".../latest.pth.tar")
        figs["tsne"].show()
        figs["confusion"].show()
        figs["activation"].show()

    Advanced usage — multi-layer, multi-method, inline display::

        figs = visualize_from_checkpoint(
            ckpt_path=".../latest.pth.tar",
            save_dir=None,
            layers=[
                "backbone.backbone.layer2",
                "backbone.backbone.layer4",
                "projector.6",
            ],
            tsne_methods=["tsne", "umap"],
        )
        figs["tsne_backbone_backbone_layer4_Sequential_tsne"].show()
        figs["activation_backbone_backbone_layer2_Sequential"].show()

        # Namespace shorthands:
        figs = visualize_from_checkpoint(ckpt_path="...", layers="all")             # meaningful backbone + projector
        figs = visualize_from_checkpoint(ckpt_path="...", layers="backbone")        # block-level backbone only
        figs = visualize_from_checkpoint(ckpt_path="...", layers="projector")       # projector unit outputs only
        figs = visualize_from_checkpoint(ckpt_path="...", layers="backbone_full")   # every backbone.* submodule
        figs = visualize_from_checkpoint(ckpt_path="...", layers="projector_full")  # every projector.* submodule
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
        layers=layers,
        tsne_methods=tsne_methods,
    )
