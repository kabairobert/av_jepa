"""
Visualization utilities for image_jepa (image/CIFAR-10 specific).

Designed to work both:
  - From the training loop in main.py (called periodically during training)
  - From a Jupyter/Colab notebook (import and call directly after loading a checkpoint)

All functions accept standard PyTorch objects and optionally log to wandb.

Usage from notebook:
    from examples.image_jepa.vis import visualization_loop
    figs = visualization_loop(model, linear_probe, val_loader, device,
                               save_dir="viz_output", wandb_run=None)
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for headless/Colab environments
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
        wandb_run.log({key: wandb.Image(fig)})

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
        wandb_run.log({"vis/confusion_matrix": wandb.Image(fig)})

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
        wandb_run.log({"vis/activation_maps": wandb.Image(fig)})

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
    tsne_method="tsne",
):
    """
    Run all image_jepa visualizations.

    Performs a single val_loader pass to extract embeddings and predictions,
    then generates:
      1. t-SNE (or UMAP) latent space plot
      2. Confusion matrix
      3. Activation maps from the last ResNet layer

    Args:
        model:         ImageSSL model
        linear_probe:  LinearProbe classifier
        val_loader:    validation DataLoader
        device:        torch device
        class_names:   list of class names (defaults to CIFAR-10)
        use_amp:       enable mixed precision for the forward pass
        save_dir:      directory to save PNG figures (None = no file output)
        wandb_run:     wandb run object (None = no wandb logging)
        epoch:         current epoch number (appended to save filenames)
        tsne_method:   "tsne" or "umap"

    Returns:
        dict: {"tsne": fig, "confusion": fig, "activation": fig}
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    suffix = f"_epoch{epoch:04d}" if epoch is not None else ""
    save_dir = Path(save_dir) if save_dir is not None else None

    # --- Single pass: embeddings, predictions, targets ---
    embeddings, preds, targets = _extract_embeddings_and_preds(
        model, linear_probe, val_loader, device, use_amp=use_amp
    )

    # --- Vis 1: Latent space (t-SNE / UMAP) ---
    fig_tsne = plot_latent_tsne(
        embeddings, targets,
        class_names=class_names,
        save_path=str(save_dir / f"latent_{tsne_method}{suffix}.png") if save_dir else None,
        wandb_run=wandb_run,
        method=tsne_method,
        title=f"Latent Space{suffix}",
    )

    # --- Vis 2: Confusion matrix ---
    fig_cm = plot_confusion_matrix(
        preds, targets,
        class_names=class_names,
        save_path=str(save_dir / f"confusion_matrix{suffix}.png") if save_dir else None,
        wandb_run=wandb_run,
        title=f"Confusion Matrix{suffix}",
    )

    # --- Vis 3: Activation maps (own forward pass via hook) ---
    fig_act = plot_activation_maps(
        model, val_loader, device,
        save_path=str(save_dir / f"activation_maps{suffix}.png") if save_dir else None,
        wandb_run=wandb_run,
        title=f"Activation Maps{suffix}",
        use_amp=use_amp,
    )

    return {"tsne": fig_tsne, "confusion": fig_cm, "activation": fig_act}
