#!/usr/bin/env python3
"""
sweep.py — launch experiment configs for one batch or all batches.

Usage:
    python sweep.py --batch B01_predictor_geometry --cfg_dir ../cfgs
    python sweep.py --all --cfg_dir ../cfgs
    python sweep.py --all --cfg_dir ../cfgs --dry-run
    python sweep.py --all --from C07_affine_l1prior_l1pred_3D2f_highnoise

Each wandb run is tagged with all batch IDs it belongs to + the config name.
Shared configs across batches are launched once with multiple batch tags.
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_batch(batch_dir: Path) -> dict:
    hyp_file = batch_dir / "hypotheses.yaml"
    if not hyp_file.exists():
        raise FileNotFoundError(f"No hypotheses.yaml in {batch_dir}")
    with open(hyp_file) as f:
        return yaml.safe_load(f)


def collect_all_batches(batches_root: Path) -> list[dict]:
    batches = []
    for batch_dir in sorted(batches_root.iterdir()):
        if batch_dir.is_dir() and (batch_dir / "hypotheses.yaml").exists():
            batches.append(load_batch(batch_dir))
    return batches


def build_config_to_batches(batches: list[dict]) -> dict[str, list[str]]:
    """Map each config name to all batch IDs that need it."""
    cfg_to_batches: dict[str, list[str]] = defaultdict(list)
    for batch in batches:
        for cfg in batch["configs"]:
            cfg_to_batches[cfg].append(batch["batch_id"])
    return dict(cfg_to_batches)


def launch_run(
    cfg_name: str,
    cfg_path: Path,
    batch_ids: list[str],
    dry_run: bool,
    epochs: int = None,
    max_batches: int = None,
    no_wandb: bool = False,
    quickrun=False
) -> None:
    tags = batch_ids + [cfg_name]
    tags_str = ",".join(tags)
    cmd = [
        "python", "../main.py",
        "--config", str(cfg_path),
        "--wandb_tags", tags_str,
    ]
    if quickrun:
        if isinstance(quickrun, bool) and quickrun is True:
            cmd += ["--quickrun"]
        else:
            cmd += [f"--quickrun={quickrun}"]
    if epochs is not None:
        cmd += ["--optim.epochs", str(epochs)]
    if max_batches is not None:
        cmd += ["--training.max_train_batches", str(max_batches)]
    if no_wandb:
        cmd += ["--logging.log_wandb", "False"]

    if dry_run:
        print(f"[DRY RUN] Would launch: {' '.join(cmd)}")
        print(f"          Tags: {tags}")
    else:
        print(f"Launching {cfg_name} with tags {tags}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"WARNING: {cfg_name} exited with code {result.returncode}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Launch experiment sweeps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=str, help="Batch ID to run (e.g. B01_predictor_geometry)")
    group.add_argument("--all", action="store_true", help="Run all batches (deduplicates shared configs)")
    parser.add_argument("--cfg_dir", type=str, default="../cfgs", help="Path to cfgs/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    parser.add_argument(
        "--from", dest="from_cfg", type=str, default=None,
        metavar="CFG_NAME",
        help="Skip all configs before CFG_NAME (inclusive start); useful to resume a stopped sweep"
    )

    # Quickrun / Overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override optim.epochs")
    parser.add_argument("--max_batches", type=int, default=None, help="Override training.max_train_batches")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--quickrun", nargs='?', const="cpu-nolog", default=False, help="Shortcut options: cpu-nolog (default if flag is set), cpu-log, gpu-nolog, gpu-log")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    batches_root = script_dir.parent / "batches"
    cfg_dir = Path(args.cfg_dir)

    if args.all:
        batches = collect_all_batches(batches_root)
        cfg_to_batches = build_config_to_batches(batches)
        all_cfgs = sorted(cfg_to_batches.items())  # deterministic order

        # Apply --from: skip everything before the named config
        if args.from_cfg is not None:
            names = [name for name, _ in all_cfgs]
            if args.from_cfg not in names:
                print(f"ERROR: --from config '{args.from_cfg}' not found in any batch.", file=sys.stderr)
                print(f"       Available configs: {names}", file=sys.stderr)
                sys.exit(1)
            start_idx = names.index(args.from_cfg)
            skipped = names[:start_idx]
            if skipped:
                print(f"Skipping {len(skipped)} already-done configs: {skipped}")
            all_cfgs = all_cfgs[start_idx:]

        print(f"Found {len(batches)} batches, {len(cfg_to_batches)} unique configs total, "
              f"launching {len(all_cfgs)}.")
        for cfg_name, batch_ids in all_cfgs:
            cfg_path = cfg_dir / f"{cfg_name}.yaml"
            if not cfg_path.exists():
                print(f"WARNING: Config file not found: {cfg_path}", file=sys.stderr)
                continue
            launch_run(
                cfg_name, cfg_path, batch_ids, dry_run=args.dry_run,
                epochs=args.epochs, max_batches=args.max_batches, no_wandb=args.no_wandb,
                quickrun=args.quickrun
            )
    else:
        batch_dir = batches_root / args.batch
        if not batch_dir.exists():
            print(f"ERROR: Batch directory not found: {batch_dir}", file=sys.stderr)
            sys.exit(1)
        batch = load_batch(batch_dir)
        batch_id = batch["batch_id"]
        cfgs = batch["configs"]

        # Apply --from within a single batch too
        if args.from_cfg is not None:
            if args.from_cfg not in cfgs:
                print(f"ERROR: --from config '{args.from_cfg}' not found in batch {batch_id}.", file=sys.exit(1))
            start_idx = cfgs.index(args.from_cfg)
            cfgs = cfgs[start_idx:]

        print(f"Launching batch {batch_id} ({len(cfgs)} configs)")
        for cfg_name in cfgs:
            cfg_path = cfg_dir / f"{cfg_name}.yaml"
            if not cfg_path.exists():
                print(f"WARNING: Config file not found: {cfg_path}", file=sys.stderr)
                continue
            launch_run(
                cfg_name, cfg_path, [batch_id], dry_run=args.dry_run,
                epochs=args.epochs, max_batches=args.max_batches, no_wandb=args.no_wandb,
                quickrun=args.quickrun
            )


if __name__ == "__main__":
    main()
