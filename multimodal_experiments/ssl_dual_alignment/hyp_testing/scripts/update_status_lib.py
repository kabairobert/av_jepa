"""
update_status_lib.py — shared logic for batch STATUS.md generation.

Each batch's update_status.py should call generate_status_md() with
batch-specific parameters instead of reimplementing the W&B query,
HTML scan, and table-writing logic.
"""

import os
from pathlib import Path
from typing import Callable, Optional

import wandb
import yaml


def generate_status_md(
    batch_tag: str,
    batch_id_str: str,
    cfg_files: list[Path],
    status_file: Path,
    checkpoint_root: Path,
    cfg_to_params_fn: Callable[[dict], str],
    entity: str = "robertkabai-um",
    project: str = "eb_jepa",
    cfg_tag_filter: Optional[Callable[[str], bool]] = None,
) -> None:
    """Generate STATUS.md for a batch by querying W&B and scanning local HTML files.

    Args:
        batch_tag:       W&B tag used to filter runs (e.g. "B11_data_scaling").
        batch_id_str:    Human-readable batch identifier for the status header.
        cfg_files:       Sorted list of config YAML paths for this batch.
        status_file:     Path to the STATUS.md file to write.
        checkpoint_root: Root of local checkpoint directory to scan for HTML files.
        cfg_to_params_fn: Function (cfg_dict -> str) that formats the parameter
                          summary column for a config. Batch-specific.
        entity:          W&B entity name.
        project:         W&B project name.
        cfg_tag_filter:  Optional predicate (tag -> bool) to identify the config-
                         name tag among a run's tags. Defaults to checking that the
                         tag starts with the batch_tag prefix and contains the
                         config stem.
    """
    api = wandb.Api()

    # --- 1. Query W&B runs for this batch ---
    print(f"Querying W&B for {batch_id_str} status...")
    runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": [batch_tag]}})

    # --- 2. Scan local filesystem for Plotly HTML files ---
    print("Scanning for local Plotly HTML files...")
    run_id_to_latest_html: dict[str, Path] = {}
    if checkpoint_root.exists():
        for html_file in checkpoint_root.glob("**/interactive_3d_4way_html_*.html"):
            run_id = None
            for part in html_file.parts:
                if part.startswith("run-") and "-" in part:
                    run_id = part.split("-")[-1]
                    break
            if run_id:
                existing = run_id_to_latest_html.get(run_id)
                if existing is None or html_file.stat().st_mtime > existing.stat().st_mtime:
                    run_id_to_latest_html[run_id] = html_file

    # --- 3. Map cfg_name -> first matching run ---
    # Default filter: tag that starts with batch_tag + "_" and matches a cfg file stem.
    cfg_stems = {f.stem for f in cfg_files}

    def _default_tag_filter(tag: str) -> bool:
        return tag in cfg_stems

    tag_filter = cfg_tag_filter if cfg_tag_filter is not None else _default_tag_filter

    cfg_to_run: dict[str, dict] = {}
    for run in runs:
        cfg_tag_match = next((t for t in run.tags if tag_filter(t)), None)
        if cfg_tag_match and cfg_tag_match not in cfg_to_run:
            cfg_to_run[cfg_tag_match] = {
                "id": run.id,
                "state": run.state,
                "url": run.url,
                "html": run_id_to_latest_html.get(run.id),
            }

    # --- 4. Write STATUS.md ---
    status_dir = status_file.parent
    with open(status_file, "w") as f:
        f.write(f"# {batch_id_str} Sweep Status\n\n")
        f.write("| Config | Status | Parameters | WandB Link | Rel Link | Abs Link |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")

        for cfg_file in cfg_files:
            cfg_name = cfg_file.stem
            with open(cfg_file) as y:
                cfg_dict = yaml.safe_load(y)
            params = cfg_to_params_fn(cfg_dict)

            run_info = cfg_to_run.get(cfg_name)
            if run_info:
                wandb_link = f"[W&B]({run_info['url']})"
                abs_html = run_info["html"]
                if abs_html:
                    rel_html = os.path.relpath(abs_html, status_dir)
                    html_rel_link = f"[HTML]({rel_html})"
                    html_abs_link = f"[HTML]({abs_html})"
                else:
                    html_rel_link = "-"
                    html_abs_link = "-"
                f.write(
                    f"| {cfg_name} | {run_info['state'].upper()} | {params} "
                    f"| {wandb_link} | {html_rel_link} | {html_abs_link} |\n"
                )
            else:
                f.write(f"| {cfg_name} | TODO | {params} | - | - | - |\n")

    print(f"Status file written: {status_file}")
