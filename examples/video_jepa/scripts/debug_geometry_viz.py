from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from eb_jepa.architectures import Projector, ResNet5, ResUNet, StateOnlyPredictor
from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VCLoss, VideoJEPA_BCS, VideoJEPA_BCS_Euler_Scaleinvariant
from eb_jepa.training_utils import load_config, resolve_projector_dims_from_cfg, setup_seed
from examples.video_jepa.vis import assemble_geometry_viz_videos, geometry_visualization_loop


class _DummyReg:
    def __init__(self):
        self.proj = nn.Identity()


class _DummyJEPA:
    def __init__(self, dstc):
        self.regularizer = _DummyReg()
        self.dstc = dstc

    def encoder(self, x):
        b, _, t, _, _ = x.shape
        return torch.randn(b, self.dstc, t, 8, 8, device=x.device)

    def unroll(self, observations, actions=None, nsteps=1, unroll_mode="parallel", compute_loss=False, return_all_steps=True):
        b, _, t, _, _ = observations.shape
        preds = [
            torch.randn(b, self.dstc, t - 2, 8, 8, device=observations.device)
            for _ in range(nsteps)
        ]
        return preds, None


def _build_jepa_from_cfg(cfg, device):
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor_model = ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)

    loss_type = cfg.loss.get("type", "vcreg")
    proj_hidden, proj_out = resolve_projector_dims_from_cfg(cfg, loss_type)
    projector = Projector(f"{cfg.model.dstc}-{proj_hidden}-{proj_out}")

    if loss_type == "bcs":
        regularizer = VideoJEPA_BCS(
            num_slices=cfg.loss.get("num_slices", 32),
            lmbd=cfg.loss.get("lmbd", 0.05),
            proj=projector,
        )
    elif loss_type == "bcs-euler-scalefree":
        regularizer = VideoJEPA_BCS_Euler_Scaleinvariant(
            num_slices=cfg.loss.get("num_slices", 32),
            lmbd=cfg.loss.get("lmbd", 0.05),
            proj=projector,
        )
    else:
        regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)

    jepa = JEPA(encoder, encoder, predictor, regularizer, SquareLossSeq(projector)).to(device)
    jepa.eval()
    return jepa


def _load_batch(cfg, device, batch_size, frames, synthetic, height=64, width=64):
    if synthetic:
        video = torch.rand(batch_size, cfg.model.dobs, frames, height, width, device=device)
        occ = torch.randint(0, 2, (batch_size, frames, 8, 8), device=device).float()
        return {"video": video, "digit_location": occ}

    dset = MovingMNISTDet(split="val")
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch.items()}


def _save_figures(figures, out_dir, epoch, include_epoch_in_filename):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, fig in figures.items():
        if include_epoch_in_filename:
            out_path = out_dir / f"{key}_epoch_{int(epoch):04d}.png"
        else:
            out_path = out_dir / f"{key}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CPU-friendly one-batch geometry visualization debug run")
    parser.add_argument("--cfg", default="examples/video_jepa/cfgs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--time", type=int, default=None, help="Alias of --frames")
    parser.add_argument("--height", type=int, default=64, help="Synthetic frame height")
    parser.add_argument("--width", type=int, default=64, help="Synthetic frame width")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Base seed; defaults to cfg.meta.seed")
    parser.add_argument(
        "--no-seed-by-epoch",
        action="store_true",
        help="Disable seed offset by epoch (default behavior offsets seed by epoch)",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use random synthetic batch instead of dataset")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional JEPA checkpoint path")
    parser.add_argument("--out-dir", type=str, default="examples/video_jepa/debug_outputs")
    parser.add_argument(
        "--assemble-videos",
        action="store_true",
        help="Assemble geometry evolution videos from out-dir/geometry_viz/epoch_*",
    )
    parser.add_argument("--time-mode", type=str, default="", help="Optional override: uniform|windowed")
    parser.add_argument(
        "--use-dummy-jepa",
        action="store_true",
        help="Use a tiny dummy JEPA (fast on CPU) to debug visualization plumbing only",
    )
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    base_seed = int(args.seed) if args.seed is not None else int(cfg.meta.seed)
    run_seed = base_seed if args.no_seed_by_epoch else base_seed + int(args.epoch)
    setup_seed(run_seed)
    print(f"seed={run_seed} (base={base_seed}, epoch_offset={not args.no_seed_by_epoch})")

    device = torch.device("cpu")
    use_dummy_jepa = bool(args.use_dummy_jepa or (args.synthetic and not args.checkpoint))
    if use_dummy_jepa:
        jepa = _DummyJEPA(dstc=cfg.model.dstc)
        print("using dummy JEPA for fast CPU debug")
    else:
        jepa = _build_jepa_from_cfg(cfg, device)

    if args.checkpoint and not use_dummy_jepa:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            jepa.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            jepa.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            jepa.load_state_dict(ckpt, strict=False)
        print(f"loaded checkpoint: {args.checkpoint}")

    frames = int(args.time) if args.time is not None else int(args.frames)

    batch = _load_batch(
        cfg=cfg,
        device=device,
        batch_size=args.batch_size,
        frames=frames,
        synthetic=args.synthetic,
        height=args.height,
        width=args.width,
    )

    geometry_cfg = cfg.logging.get("geometry_viz", {})
    if isinstance(geometry_cfg, dict):
        geometry_cfg = dict(geometry_cfg)
    else:
        geometry_cfg = OmegaConf.to_container(geometry_cfg, resolve=True)
    geometry_cfg["enabled"] = True

    long_cfg = dict(geometry_cfg.get("long_sequence", {}))
    long_cfg["enabled"] = True
    if args.time_mode:
        long_cfg["time_subsample_mode"] = args.time_mode
    geometry_cfg["long_sequence"] = long_cfg

    if use_dummy_jepa:
        plots_cfg = dict(geometry_cfg.get("plots", {}))
        plots_cfg["activation_overlays"] = False
        geometry_cfg["plots"] = plots_cfg
        print("dummy JEPA mode: disabling activation_overlays (requires real encoder modules)")

    figures, meta = geometry_visualization_loop(
        batch=batch,
        jepa=jepa,
        device=device,
        geometry_cfg=geometry_cfg,
        detection_targets=batch.get("digit_location"),
        epoch=args.epoch,
    )

    print("figure keys:", sorted(figures.keys()))
    print("long-sequence meta:", meta)

    out_dir = Path(args.out_dir) / "geometry_viz" / f"epoch_{int(args.epoch):04d}"
    _save_figures(
        figures,
        out_dir,
        epoch=args.epoch,
        include_epoch_in_filename=bool(geometry_cfg.get("include_epoch_in_filename", True)),
    )

    if args.assemble_videos:
        logs = assemble_geometry_viz_videos(args.out_dir, fps=int(geometry_cfg.get("evolution_fps", 2)))
        print("assembled video keys:", sorted(logs.keys()))


if __name__ == "__main__":
    main()
