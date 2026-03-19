#!/usr/bin/env python3
"""
Local diagnostics index resolver.

Query machine-readable diagnostics events stored in run folders during training.
Requires a local run directory with diagnostics/index.json produced by eb_jepa training.

Input: Path to a run folder containing diagnostics/ subfolder
Output: JSON-formatted event and metric queries from the diagnostics index

Usage examples:
  # List all events
  python resolve_diagnostics.py /path/to/run/folder
  
  # Inspect a specific event
  python resolve_diagnostics.py /path/to/run/folder --event event_001
  
  # Find events containing a specific metric
  python resolve_diagnostics.py /path/to/run/folder --metric loss/loss_jepa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve diagnostics entries from a run folder.")
    parser.add_argument("run_dir", help="Path to the run directory containing diagnostics/")
    parser.add_argument("--event", help="Specific event id to inspect")
    parser.add_argument("--metric", help="Metric key to search for")
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List all event IDs in the index"
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all unique metrics across all events"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact output (one item per line) for list operations"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    index_path = run_dir / "diagnostics" / "index.json"
    if not index_path.exists():
        raise SystemExit(f"Diagnostics index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    events = index.get("events", {})

    # List all events
    if args.list_events:
        event_ids = sorted(events.keys())
        if args.compact:
            for eid in event_ids:
                print(eid)
        else:
            print(json.dumps(event_ids, indent=2))
        return

    # List all unique metrics
    if args.list_metrics:
        all_metrics = set()
        for event in events.values():
            all_metrics.update(event.get("metrics", []))
        metric_list = sorted(all_metrics)
        if args.compact:
            for metric in metric_list:
                print(metric)
        else:
            print(json.dumps(metric_list, indent=2))
        return

    # Query by specific event
    if args.event:
        event = events.get(args.event)
        if event is None:
            raise SystemExit(f"Event not found: {args.event}")
        print(json.dumps(event, indent=2, sort_keys=True))
        return

    # Query by metric key
    if args.metric:
        matches = []
        for event_id, event in events.items():
            if args.metric in event.get("metrics", []):
                matches.append({"event_id": event_id, **event})
        if not matches:
            raise SystemExit(f"Metric not found in diagnostics index: {args.metric}")
        print(json.dumps(matches, indent=2, sort_keys=True))
        return

    # Default: print full index
    print(json.dumps({"events": events}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
