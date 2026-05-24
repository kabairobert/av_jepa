#!/usr/bin/env python3
"""
run_hypotheses.py — pull wandb metrics, evaluate decision rules, write RESULTS.md.

Workflow:
    1. Group runs by configuration name.
    2. Identify 'Train' run (base metrics) and latest 'Eval' run (geometry/plots).
    3. If --eval, run eval.py on local checkpoints for missing evals.
    4. Merge metrics (Train.summary + Eval.summary) for report evaluation.

Usage:
    python run_hypotheses.py --batch B01_predictor_geometry --wandb_project YOUR_PROJECT
    python run_hypotheses.py --batch B06_prior_vs_predictor_noise --eval --wandb_project YOUR_PROJECT
"""

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml

try:
    import wandb
except ImportError:
    raise ImportError("wandb not installed. Run: pip install wandb")


# ---------------------------------------------------------------------------
# Path mapping & Local Eval
# ---------------------------------------------------------------------------

def resolve_local_run_dir(wb_path_str: str) -> Path | None:
    """Map W&B absolute path (e.g. from Colab) to local project root."""
    if not wb_path_str:
        return None
    p = Path(wb_path_str)
    # Pattern: .../checkpoints/sslda/dev_YYYY-MM-DD_HH-MM/exp_folder/latest.pth.tar
    parts = p.parts
    try:
        idx = parts.index("sslda")
        sweep_folder = parts[idx + 1]
        exp_folder = parts[idx + 2]
        
        # This script is at multimodal_experiments/ssl_dual_alignment/hyp_testing/scripts/
        # Root is 5 levels up: multimodal_experiments (1), ssl_dual_alignment (2), 
        # hyp_testing (3), scripts (4), run_hypotheses.py (5)
        # Actually 4 levels up: scripts -> hyp_testing -> ssl_dual_alignment -> multimodal_experiments -> eb_jepa_private
        script_dir = Path(__file__).parent
        root = script_dir.parent.parent.parent.parent.parent
        local_root = root / "checkpoints" / "sslda" / sweep_folder / exp_folder
        
        if local_root.exists():
            return local_root
    except (ValueError, IndexError):
        pass
    return None


def run_local_eval(run_dir: Path):
    """Execute eval.py subprocess on a specific run folder."""
    print(f"  >>> Running local eval on: {run_dir}")
    cmd = [
        "python", "-m", "multimodal_experiments.ssl_dual_alignment.eval",
        "--folder", str(run_dir),
        "--log_wandb", "true",
        "--log_interactive_3d", "true"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: eval.py failed for {run_dir.name}: {e}")


# ---------------------------------------------------------------------------
# Metrics registry
# ---------------------------------------------------------------------------

def load_registry(scripts_dir: Path) -> dict[str, str]:
    """Load alias -> wandb_key mapping from metrics_registry.yaml."""
    registry_path = scripts_dir.parent / "metrics_registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"metrics_registry.yaml not found at {registry_path}")
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data.get("metrics", {})


def resolve_metric(alias: str, registry: dict[str, str]) -> str:
    """Resolve alias to wandb key. If alias not in registry, return as-is."""
    return registry.get(alias, alias)


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def collect_metrics(batch: dict, wandb_project: str, do_eval: bool, rerun_eval: bool) -> dict[str, dict]:
    """Fetch and merge metrics for all configs in batch."""
    batch_id = batch["batch_id"]
    api = wandb.Api()
    results = {}

    for cfg_name in batch["configs"]:
        print(f"  Processing {cfg_name}...")
        
        # 1. Fetch all runs for this config + batch
        runs = list(api.runs(wandb_project, filters={"tags": {"$all": [batch_id, cfg_name]}}))
        if not runs:
            print(f"    NOT FOUND: No runs tagged with [{batch_id}, {cfg_name}]")
            continue

        # 2. Group into Train (no 'eval' tag) and Eval (has 'eval' tag)
        train_run = None
        eval_runs = []
        for r in runs:
            if "eval" in r.tags:
                eval_runs.append(r)
            else:
                # If multiple trains, take newest
                if train_run is None or r.created_at > train_run.created_at:
                    train_run = r

        if train_run is None:
            print(f"    WARNING: No main training run found for {cfg_name} (only eval runs?)")
            continue

        # 3. Determine if we need to run a new eval
        latest_eval = sorted(eval_runs, key=lambda r: r.created_at)[-1] if eval_runs else None
        
        needs_eval = rerun_eval or (do_eval and latest_eval is None)
        if needs_eval:
            # Get checkpoint path from Train run
            ckpt_path = train_run.summary.get("eval/checkpoint_path")
            if ckpt_path:
                local_dir = resolve_local_run_dir(ckpt_path)
                if local_dir:
                    run_local_eval(local_dir)
                    # Re-fetch runs to get the new eval output
                    runs = list(api.runs(wandb_project, filters={"tags": {"$all": [batch_id, cfg_name]}}))
                    eval_runs = [r for r in runs if "eval" in r.tags]
                    latest_eval = sorted(eval_runs, key=lambda r: r.created_at)[-1] if eval_runs else None
                else:
                    print(f"    WARNING: Local checkpoint dir not found for path in W&B: {ckpt_path}")
            else:
                print(f"    WARNING: 'eval/checkpoint_path' metric missing in Train run summary.")

        # 4. Merge summaries
        merged_summary = dict(train_run.summary)
        if latest_eval:
            # Eval summary takes precedence for overlapping keys (e.g. geometry metrics)
            merged_summary.update(dict(latest_eval.summary))
            print(f"    Merged metrics from eval run: {latest_eval.name}")
        
        results[cfg_name] = merged_summary

    return results


# ---------------------------------------------------------------------------
# Decision rule evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(condition: str, val_dict: dict[str, float], threshold: float) -> bool:
    expr = condition
    for name, val in val_dict.items():
        expr = expr.replace(f"{{{name}}}", str(val))
    expr = expr.replace("{threshold}", str(threshold))
    expr = re.sub(r"abs\(([^)]+)\)", lambda m: str(abs(eval(m.group(1)))), expr)
    try:
        # Use simple eval; we trust our hypotheses.yaml
        return bool(eval(expr))
    except Exception as e:
        print(f"    WARNING: Could not evaluate condition '{expr}': {e}")
        return False


def evaluate_hypothesis(hyp: dict, metrics: dict[str, dict], registry: dict[str, str]) -> dict:
    configs = hyp["configs_compared"]
    m_configs = {name: metrics.get(cfg_name, {}) for name, cfg_name in configs.items()}

    rule_results = []
    handled_metrics = set()
    
    # 1. Evaluate decision rules
    for rule in hyp["decision_rules"]:
        alias = rule["metric"]
        handled_metrics.add(alias)
        wandb_key = resolve_metric(alias, registry)
        threshold = rule.get("threshold")
        
        val_dict = {name: m_configs[name].get(wandb_key) for name in configs}

        # Check if any values needed for condition are missing
        tokens = re.findall(r"\{([^}]+)\}", rule["condition"])
        missing = any(val_dict.get(tok) is None for tok in tokens if tok != "threshold")

        if threshold is None:
            rule_results.append({
                "metric": alias,
                "verdict": "⚠️ THRESHOLD_NOT_SET",
                "a_val": val_dict.get("a"),
                "b_val": val_dict.get("b"),
                **{f"{name}_val": val_dict.get(name) for name in configs},
                "delta": None
            })
            continue

        if missing:
            rule_results.append({
                "metric": alias,
                "verdict": "❓ MISSING_DATA",
                "a_val": val_dict.get("a"),
                "b_val": val_dict.get("b"),
                **{f"{name}_val": val_dict.get(name) for name in configs},
                "delta": None
            })
            continue

        passed = evaluate_condition(rule["condition"], val_dict, threshold)
        verdict = rule["verdict_true"] if passed else rule["verdict_false"]
        
        a_val = val_dict.get("a")
        b_val = val_dict.get("b")
        delta = b_val - a_val if (a_val is not None and b_val is not None) else None
        
        rule_results.append({
            "metric": alias,
            "verdict": f"✅ {verdict}" if any(x in verdict for x in ["SUPPORTED", "OUTPERFORMS", "PRESERVED", "IMPROVED", "CLEAN", "BETTER", "DRIVES", "DEPENDENT"])
                       else (f"❌ {verdict}" if any(x in verdict for x in ["REFUTED", "COMPRESSED", "DIRTY", "INSUFFICIENT"])
                             else f"⚠️ {verdict}"),
            "a_val": round(a_val, 4) if a_val is not None else None,
            "b_val": round(b_val, 4) if b_val is not None else None,
            **{f"{name}_val": round(val, 4) if val is not None else None for name, val in val_dict.items()},
            "delta": round(delta, 4) if delta is not None else None
        })

    # 2. Add remaining primary metrics
    primary_metrics = hyp.get("primary_metrics", [])
    for alias in primary_metrics:
        if alias in handled_metrics:
            continue
        wandb_key = resolve_metric(alias, registry)
        val_dict = {name: m_configs[name].get(wandb_key) for name in configs}
        
        a_val = val_dict.get("a")
        b_val = val_dict.get("b")
        delta = b_val - a_val if (a_val is not None and b_val is not None) else None
        
        rule_results.append({
            "metric": alias,
            "verdict": "\u2014",
            "a_val": round(a_val, 4) if a_val is not None else None,
            "b_val": round(b_val, 4) if b_val is not None else None,
            **{f"{name}_val": round(val, 4) if val is not None else None for name, val in val_dict.items()},
            "delta": round(delta, 4) if delta is not None else None
        })

    cfg_a = configs["a"]
    cfg_b = configs["b"]
    narrative = generate_narrative(hyp, cfg_a, cfg_b, rule_results)
    return {
        "id": hyp["id"],
        "claim": hyp["claim"],
        "cfg_a": cfg_a,
        "cfg_b": cfg_b,
        "configs_compared": configs,
        "rule_results": rule_results,
        "check_plots": hyp.get("check_plots", []),
        "narrative": narrative,
    }


def generate_narrative(hyp: dict, cfg_a: str, cfg_b: str, rule_results: list) -> str:
    lines = []
    for r in rule_results:
        if r["verdict"] == "\u2014":
            continue
        metric = r["metric"]
        verdict = r["verdict"]
        a_val = r["a_val"]
        b_val = r["b_val"]
        delta = r["delta"]
        if delta is None:
            lines.append(f"{metric}: data missing or threshold not set.")
            continue
        direction = "higher" if delta > 0 else "lower"
        magnitude = abs(delta)
        short_a = cfg_a.split("_")[0]
        short_b = cfg_b.split("_")[0]
        lines.append(
            f"{metric}: {short_b} scored {b_val:.4f} vs {short_a}'s {a_val:.4f} "
            f"(\u0394={delta:+.4f}, {direction} by {magnitude:.4f}). {verdict}."
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
        
        cfg_keys = sorted(r["configs_compared"].keys())
        headers = ["Metric"]
        for key in cfg_keys:
            cfg_name = r["configs_compared"][key]
            short_name = cfg_name.split('_')[0]
            headers.append(f"{cfg_name} ({short_name})")
        headers.append("Δ")
        headers.append("Verdict")
        
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        for rr in r["rule_results"]:
            row = [rr['metric']]
            for key in cfg_keys:
                val = rr.get(f"{key}_val")
                row.append(str(val) if val is not None else 'N/A')
            d = f"{rr['delta']:+.4f}" if rr['delta'] is not None else 'N/A'
            row.append(d)
            row.append(rr['verdict'])
            lines.append("| " + " | ".join(row) + " |")
            
        lines.append("")
        lines.append(f"**Summary:** {r['narrative']}\n")
        if r["check_plots"]:
            lines.append("\u26a0\ufe0f **Manual plot check required:**")
            for p in r["check_plots"]:
                lines.append(f"- [ ] {p}")
        lines.append("\n---\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {output_path}")


def write_master_results_md(all_results: list[tuple[dict, list]], output_path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Master Results\n",
        f"> Auto-generated by `run_hypotheses.py --all`. Last updated: {now}\n",
        "| ID | Batch | Claim | Verdict | Key Metric | \u0394 |",
        "|---|---|---|---|---|---|",
    ]
    for batch, hyp_results in all_results:
        for r in hyp_results:
            rr = r["rule_results"][0] if r["rule_results"] else {}
            verdict = rr.get("verdict", "\u2014")
            metric = rr.get("metric", "\u2014")
            delta = f"{rr['delta']:+.4f}" if rr.get("delta") is not None else "\u2014"
            claim_short = r["claim"][:60] + "..." if len(r["claim"]) > 60 else r["claim"]
            lines.append(f"| {r['id']} | {batch['batch_id']} | {claim_short} | {verdict} | {metric} | {delta} |")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate hypotheses from wandb runs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=str, help="Batch ID to evaluate")
    group.add_argument("--all", action="store_true", help="Evaluate all batches")
    
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--eval", action="store_true", help="Run local eval.py for missing eval runs")
    parser.add_argument("--eval-rerun", action="store_true", help="Force local eval.py for ALL runs")
    
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    batches_root = script_dir.parent / "batches"
    registry = load_registry(script_dir)

    if args.all:
        batch_dirs = sorted(d for d in batches_root.iterdir()
                            if d.is_dir() and (d / "hypotheses.yaml").exists())
        all_results = []
        for batch_dir in batch_dirs:
            with open(batch_dir / "hypotheses.yaml") as f:
                batch = yaml.safe_load(f)
            print(f"\nEvaluating {batch['batch_id']}...")
            metrics = collect_metrics(batch, args.wandb_project, args.eval, args.eval_rerun)
            hyp_results = [evaluate_hypothesis(h, metrics, registry) for h in batch["hypotheses"]]
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
        metrics = collect_metrics(batch, args.wandb_project, args.eval, args.eval_rerun)
        hyp_results = [evaluate_hypothesis(h, metrics, registry) for h in batch["hypotheses"]]
        write_results_md(batch, hyp_results, batch_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
