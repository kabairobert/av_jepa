#!/usr/bin/env python3
"""
run_hypotheses.py — pull wandb metrics, evaluate decision rules, write RESULTS.md.

Usage:
    python run_hypotheses.py --batch B01_predictor_geometry --wandb_project YOUR_PROJECT
    python run_hypotheses.py --all --wandb_project YOUR_PROJECT

Requires: wandb, pyyaml
Install:  pip install wandb pyyaml
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import yaml

try:
    import wandb
except ImportError:
    raise ImportError("wandb not installed. Run: pip install wandb")


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def fetch_run_metrics(wandb_project: str, cfg_name: str, batch_id: str) -> dict | None:
    """
    Find the wandb run tagged with both batch_id and cfg_name.
    Returns final metric values as a flat dict, or None if not found.
    """
    api = wandb.Api()
    runs = api.runs(
        wandb_project,
        filters={"tags": {"$all": [batch_id, cfg_name]}}
    )
    runs = list(runs)
    if not runs:
        return None
    if len(runs) > 1:
        print(f"  WARNING: Multiple runs found for {cfg_name} in {batch_id}, using most recent.")
    run = sorted(runs, key=lambda r: r.created_at, reverse=True)[0]
    # summary holds final values
    return dict(run.summary)


def collect_metrics(batch: dict, wandb_project: str) -> dict[str, dict]:
    """
    Returns {cfg_name: {metric_name: value}} for all configs in batch.
    Missing runs stored as None.
    """
    batch_id = batch["batch_id"]
    metrics = {}
    for cfg in batch["configs"]:
        print(f"  Fetching {cfg}...")
        m = fetch_run_metrics(wandb_project, cfg, batch_id)
        metrics[cfg] = m
        if m is None:
            print(f"    NOT FOUND (run missing or not tagged correctly)")
    return metrics


# ---------------------------------------------------------------------------
# Decision rule evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(condition: str, a_val: float, b_val: float, threshold: float) -> bool:
    """
    Evaluate a condition string with {a}, {b}, {threshold} placeholders.
    Supports: >, <, >=, <=, abs(...) < threshold
    """
    expr = condition.replace("{a}", str(a_val)).replace("{b}", str(b_val)).replace("{threshold}", str(threshold))
    # handle abs(...) pattern
    expr = re.sub(r"abs\(([^)]+)\)", lambda m: str(abs(eval(m.group(1)))), expr)
    try:
        return bool(eval(expr))
    except Exception as e:
        print(f"    WARNING: Could not evaluate condition '{expr}': {e}")
        return False


def evaluate_hypothesis(hyp: dict, metrics: dict[str, dict]) -> dict:
    """
    Evaluate all decision rules for a hypothesis.
    Returns a result dict with verdicts, metric values, and narrative.
    """
    cfg_a = hyp["configs_compared"]["a"]
    cfg_b = hyp["configs_compared"]["b"]
    m_a = metrics.get(cfg_a) or {}
    m_b = metrics.get(cfg_b) or {}

    rule_results = []
    for rule in hyp["decision_rules"]:
        metric = rule["metric"]
        threshold = rule.get("threshold")
        a_val = m_a.get(metric)
        b_val = m_b.get(metric)

        if threshold is None:
            rule_results.append({
                "metric": metric, "verdict": "⚠️ THRESHOLD_NOT_SET",
                "a_val": a_val, "b_val": b_val, "delta": None
            })
            continue

        if a_val is None or b_val is None:
            rule_results.append({
                "metric": metric, "verdict": "❓ MISSING_DATA",
                "a_val": a_val, "b_val": b_val, "delta": None
            })
            continue

        passed = evaluate_condition(rule["condition"], a_val, b_val, threshold)
        verdict = rule["verdict_true"] if passed else rule["verdict_false"]
        delta = b_val - a_val
        rule_results.append({
            "metric": metric,
            "verdict": f"✅ {verdict}" if "SUPPORTED" in verdict or "OUTPERFORMS" in verdict or "PRESERVED" in verdict
                       else (f"❌ {verdict}" if "REFUTED" in verdict or "COMPRESSED" in verdict
                             else f"⚠️ {verdict}"),
            "a_val": round(a_val, 4),
            "b_val": round(b_val, 4),
            "delta": round(delta, 4)
        })

    # generate narrative
    narrative = generate_narrative(hyp, cfg_a, cfg_b, rule_results)

    return {
        "id": hyp["id"],
        "claim": hyp["claim"],
        "cfg_a": cfg_a,
        "cfg_b": cfg_b,
        "rule_results": rule_results,
        "check_plots": hyp.get("check_plots", []),
        "narrative": narrative,
    }


def generate_narrative(hyp: dict, cfg_a: str, cfg_b: str, rule_results: list) -> str:
    """
    Auto-generate a 1-2 sentence plain-English narrative from rule results.
    """
    lines = []
    for r in rule_results:
        metric = r["metric"]
        verdict = r["verdict"]
        a_val = r["a_val"]
        b_val = r["b_val"]
        delta = r["delta"]

        if delta is None:
            lines.append(f"{metric}: data missing or threshold not set — cannot evaluate.")
            continue

        direction = "higher" if delta > 0 else "lower"
        magnitude = abs(delta)
        short_a = cfg_a.split("_")[0]  # e.g. C01
        short_b = cfg_b.split("_")[0]  # e.g. C03

        lines.append(
            f"{metric}: {short_b} scored {b_val:.4f} vs {short_a}'s {a_val:.4f} "
            f"(Δ={delta:+.4f}, {direction} by {magnitude:.4f}). {verdict}."
        )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------

def write_results_md(batch: dict, hyp_results: list, output_path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {batch['batch_id']} Results\n",
        f"> Auto-generated by `run_hypotheses.py`. Last updated: {now}\n",
        f"**Focus:** {batch['description']}\n",
        "---\n",
    ]

    for r in hyp_results:
        lines.append(f"## {r['id']}: {r['claim']}\n")

        # metric table
        cfg_a_short = r['cfg_a'].split('_')[0]
        cfg_b_short = r['cfg_b'].split('_')[0]
        lines.append(f"| Metric | {r['cfg_a']} ({cfg_a_short}) | {r['cfg_b']} ({cfg_b_short}) | Δ | Verdict |")
        lines.append("|---|---|---|---|---|")
        for rr in r["rule_results"]:
            a = rr['a_val'] if rr['a_val'] is not None else 'N/A'
            b = rr['b_val'] if rr['b_val'] is not None else 'N/A'
            d = f"{rr['delta']:+.4f}" if rr['delta'] is not None else 'N/A'
            lines.append(f"| {rr['metric']} | {a} | {b} | {d} | {rr['verdict']} |")
        lines.append("")

        # narrative
        lines.append(f"**Summary:** {r['narrative']}\n")

        # plot flags
        if r["check_plots"]:
            lines.append("⚠️ **Manual plot check required:**")
            for p in r["check_plots"]:
                lines.append(f"- [ ] {p}")
        lines.append("\n---\n")

    output_path.write_text("\n".join(lines))
    print(f"  Written: {output_path}")


def write_master_results_md(all_results: list[tuple[dict, list]], output_path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Master Results\n",
        f"> Auto-generated by `run_hypotheses.py --all`. Last updated: {now}\n",
        "| ID | Batch | Claim | Verdict | Key Metric | Δ |",
        "|---|---|---|---|---|---|",
    ]

    for batch, hyp_results in all_results:
        for r in hyp_results:
            # pick first rule result as key metric
            rr = r["rule_results"][0] if r["rule_results"] else {}
            verdict = rr.get("verdict", "—")
            metric = rr.get("metric", "—")
            delta = f"{rr['delta']:+.4f}" if rr.get("delta") is not None else "—"
            claim_short = r["claim"][:60] + "..." if len(r["claim"]) > 60 else r["claim"]
            lines.append(f"| {r['id']} | {batch['batch_id']} | {claim_short} | {verdict} | {metric} | {delta} |")

    output_path.write_text("\n".join(lines))
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate hypotheses from wandb runs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=str, help="Batch ID to evaluate")
    group.add_argument("--all", action="store_true", help="Evaluate all batches")
    parser.add_argument("--wandb_project", type=str, required=True,
                        help="wandb project name (e.g. 'myorg/ssl_dual_alignment')")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    batches_root = script_dir.parent / "batches"

    if args.all:
        batch_dirs = sorted(d for d in batches_root.iterdir()
                            if d.is_dir() and (d / "hypotheses.yaml").exists())
        all_results = []
        for batch_dir in batch_dirs:
            with open(batch_dir / "hypotheses.yaml") as f:
                batch = yaml.safe_load(f)
            print(f"\nEvaluating {batch['batch_id']}...")
            metrics = collect_metrics(batch, args.wandb_project)
            hyp_results = [evaluate_hypothesis(h, metrics) for h in batch["hypotheses"]]
            write_results_md(batch, hyp_results, batch_dir / "RESULTS.md")
            all_results.append((batch, hyp_results))

        master_path = script_dir.parent / "MASTER_RESULTS.md"
        write_master_results_md(all_results, master_path)

    else:
        batch_dir = batches_root / args.batch
        if not batch_dir.exists():
            print(f"ERROR: Batch not found: {batch_dir}")
            raise SystemExit(1)
        with open(batch_dir / "hypotheses.yaml") as f:
            batch = yaml.safe_load(f)
        print(f"Evaluating {batch['batch_id']}...")
        metrics = collect_metrics(batch, args.wandb_project)
        hyp_results = [evaluate_hypothesis(h, metrics) for h in batch["hypotheses"]]
        write_results_md(batch, hyp_results, batch_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
