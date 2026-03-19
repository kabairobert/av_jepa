#!/usr/bin/env python
"""
Inspect W&B run metadata and history.

Usage: python wandb_inspect_run.py --run <run_id_or_path> [--entity ENTITY] [--project PROJECT] [--scan-prefix PREFIX] [--json]

Input modes:
  - Short run ID: --run q1wxqpk2 (requires --entity and --project to be set or auto-discovered)
  - Full path: --run entity/project/run_id or --run entity/project/short_id
  
Output: Human-readable summary by default, JSON with --json flag.
"""

import sys
import json
import argparse
from collections import defaultdict

try:
    import wandb
except ImportError as e:
    print("Failed to import wandb:", e, file=sys.stderr)
    sys.exit(3)


def resolve_run(run_spec, entity=None, project=None):
    """
    Resolve a run from a spec string or short ID.
    
    Args:
        run_spec: Full path (entity/project/run_id), or short ID if entity/project provided
        entity: W&B entity (optional if full path provided)
        project: W&B project (optional if full path provided)
    
    Returns:
        (path_str, run_obj) or raises exception
    """
    api = wandb.Api()
    
    # If run_spec contains slashes, treat as full path
    if "/" in run_spec:
        try:
            run = api.run(run_spec)
            return (run_spec, run)
        except Exception as e:
            raise ValueError(f"Failed to load run from path '{run_spec}': {e}")
    
    # Otherwise, treat as short ID and require entity/project
    if not entity or not project:
        raise ValueError(
            f"Short run ID '{run_spec}' requires --entity and --project to be specified. "
            f"Alternatively, provide full path: entity/project/run_id"
        )
    
    full_path = f"{entity}/{project}/{run_spec}"
    try:
        run = api.run(full_path)
        return (full_path, run)
    except Exception as e:
        raise ValueError(f"Failed to load run from '{full_path}': {e}")


def inspect_run(run_spec, entity=None, project=None, scan_prefix="geometry_viz/", json_output=False):
    """
    Inspect a W&B run's metadata and history.
    
    Scans for keys matching scan_prefix in run history by default.
    """
    try:
        path, run = resolve_run(run_spec, entity, project)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    output = {}
    
    # Basic run info
    output["run_path"] = path
    output["run_name"] = getattr(run, "name", None)
    output["run_id"] = run.id
    output["run_short_id"] = getattr(run, "short_id", None)
    
    # Summary info
    summary_keys = list(run.summary.keys())[:50]  # First 50 for brevity
    output["summary_keys_sample"] = summary_keys
    output["summary_keys_count"] = len(run.summary.keys())
    
    # Scan history for specified prefix
    counts = defaultdict(int)
    entries = []
    try:
        for rec in run.scan_history():
            step = rec.get("_step")
            for k, v in rec.items():
                if isinstance(k, str) and k.startswith(scan_prefix) and v is not None:
                    counts[k] += 1
                    entries.append((step, k))
    except Exception as e:
        output["history_scan_error"] = str(e)
        entries = []
    
    output[f"{scan_prefix}counts"] = dict(counts)
    output[f"{scan_prefix}sample_entries"] = entries[:50]
    steps_with_media = sorted(set(s for s, _k in entries if s is not None))
    output[f"{scan_prefix}steps_with_data"] = steps_with_media
    
    if json_output:
        print(json.dumps(output, indent=2, default=str))
    else:
        # Human-readable output
        print(f"Found run: {path}")
        print(f"Run name: {output['run_name']}")
        print(f"Run id: {output['run_id']} (short: {output['run_short_id']})")
        print(f"Summary keys ({output['summary_keys_count']} total, showing first {len(summary_keys)}):")
        print(json.dumps(summary_keys, indent=2))
        print(f"\n{scan_prefix} media counts:")
        print(json.dumps(output[f"{scan_prefix}counts"], indent=2))
        print(f"\nSample {scan_prefix} entries (step, key):")
        for step, key in output[f"{scan_prefix}sample_entries"]:
            print(f"  {step}, {key}")
        print(f"\nSteps with {scan_prefix} data: {output[f'{scan_prefix}steps_with_data']}")
        print("\nDone")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Inspect W&B run metadata and history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full path (entity/project/run_id)
  python wandb_inspect_run.py --run robertkabai-um/eb_jepa/q1wxqpk2
  
  # Short ID with entity and project
  python wandb_inspect_run.py --run q1wxqpk2 --entity robertkabai-um --project eb_jepa
  
  # Custom prefix and JSON output
  python wandb_inspect_run.py --run q1wxqpk2 --entity robertkabai-um --project eb_jepa \\
    --scan-prefix my_prefix/ --json
        """
    )
    
    parser.add_argument(
        "--run", 
        required=True, 
        help="W&B run ID (short or full path: entity/project/run_id)"
    )
    parser.add_argument(
        "--entity", 
        default=None, 
        help="W&B entity (required if --run is a short ID)"
    )
    parser.add_argument(
        "--project", 
        default=None, 
        help="W&B project (required if --run is a short ID)"
    )
    parser.add_argument(
        "--scan-prefix", 
        default="geometry_viz/", 
        help="Key prefix to scan for in run history (default: geometry_viz/)"
    )
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output as JSON instead of human-readable format"
    )
    
    args = parser.parse_args()
    
    return inspect_run(
        args.run,
        entity=args.entity,
        project=args.project,
        scan_prefix=args.scan_prefix,
        json_output=args.json
    )


if __name__ == "__main__":
    sys.exit(main())
