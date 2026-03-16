#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve diagnostics entries from a run folder.")
    parser.add_argument("run_dir", help="Path to the run directory containing diagnostics/")
    parser.add_argument("--event", help="Specific event id to inspect")
    parser.add_argument("--metric", help="Metric key to search for")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    index_path = run_dir / "diagnostics" / "index.json"
    if not index_path.exists():
        raise SystemExit(f"Diagnostics index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    events = index.get("events", {})

    if args.event:
        event = events.get(args.event)
        if event is None:
            raise SystemExit(f"Event not found: {args.event}")
        print(json.dumps(event, indent=2, sort_keys=True))
        return

    if args.metric:
        matches = []
        for event_id, event in events.items():
            if args.metric in event.get("metrics", []):
                matches.append({"event_id": event_id, **event})
        if not matches:
            raise SystemExit(f"Metric not found in diagnostics index: {args.metric}")
        print(json.dumps(matches, indent=2, sort_keys=True))
        return

    print(json.dumps({"events": events}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
