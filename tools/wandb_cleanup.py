#!/usr/bin/env python3
"""
W&B run cleanup utility with safe dry-run defaults.

Delete W&B runs and optionally their artifact files directly from the W&B API.
Designed with safety defaults: all operations dry-run by default and require --yes confirmation.

Input: W&B run path (entity/project/run_id) via --run flag
Output: Summary of deleted/skipped runs and files

Usage examples:
  # Dry-run (default): show what would be deleted, no changes
  python wandb_cleanup.py --run robertkabai-um/eb_jepa/o17io1li

  # Actually delete the run and its artifact files (irreversible, requires --yes)
  python wandb_cleanup.py --run robertkabai-um/eb_jepa/o17io1li --no-dry-run --delete-files --yes

Requirements: wandb Python package, W&B authentication (wandb login or WANDB_API_KEY env var).
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe WandB run cleanup (delete run and optionally its files)")
    p.add_argument(
        "--run",
        required=True,
        help="Run path: entity/project/run_id (e.g. user/proj/abcd1234)",
    )
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Show what would be deleted (default)")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Perform deletion")
    p.add_argument("--delete-files", action="store_true", help="Also delete files uploaded to the run (optional)")
    p.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    return p.parse_args()


def list_files(run) -> List:
    try:
        return list(run.files())
    except Exception:
        # Some runs may have many files or restricted access; return empty on failure
        return []


def main() -> None:
    args = parse_args()
    api = wandb.Api()

    try:
        run = api.run(args.run)
    except Exception as e:
        print(f"ERROR: could not fetch run '{args.run}': {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Found run: id={run.id}, name={getattr(run, 'name', None)}, state={getattr(run, 'state', None)}")

    files = list_files(run)
    print(f"Files uploaded: {len(files)}")
    if files:
        for f in files[:20]:
            print(" -", f.name)
        if len(files) > 20:
            print(" - ...")

    if args.dry_run:
        print("\nDry run — nothing will be deleted. Rerun with --no-dry-run to delete.")
        return

    if not args.yes:
        resp = input(f"Proceed to delete run '{args.run}'? This is irreversible (yes/[no]): ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted by user.")
            return

    if args.delete_files and files:
        for f in files:
            try:
                print(f"Deleting file: {f.name}")
                f.delete()
            except Exception as e:
                print(f"Failed to delete file {f.name}: {e}", file=sys.stderr)

    try:
        run.delete()
        print("Run deleted.")
    except Exception as e:
        print(f"Failed to delete run: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
