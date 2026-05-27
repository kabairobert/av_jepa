import wandb
import os
import yaml
from pathlib import Path

api = wandb.Api()
entity = "robertkabai-um"
project = "eb_jepa"

# Resolve paths dynamically relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent.parent
checkpoint_root = ROOT_DIR / "checkpoints" / "sslda"
cfg_dir = SCRIPT_DIR.parent.parent.parent / "cfgs"
all_cfg_files = sorted(list(cfg_dir.glob("B11_*.yaml")))

print(f"Querying WandB for B11 status...")
runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": ["B11_data_scaling"]}})

# Pre-map all local plotly files to their run IDs
print("Scanning for local Plotly HTML files...")
run_id_to_latest_html = {}
if checkpoint_root.exists():
    for html_file in checkpoint_root.glob("**/interactive_3d_4way_html_*.html"):
        parts = html_file.parts
        run_id = None
        for p in parts:
            if p.startswith("run-") and "-" in p:
                run_id = p.split("-")[-1]
                break
        
        if run_id:
            if run_id not in run_id_to_latest_html:
                run_id_to_latest_html[run_id] = html_file
            else:
                if html_file.stat().st_mtime > run_id_to_latest_html[run_id].stat().st_mtime:
                    run_id_to_latest_html[run_id] = html_file

# Map config name to run info
cfg_to_run = {}
for run in runs:
    cfg_tag = next((tag for tag in run.tags if tag.startswith("B11_")), None)
    if cfg_tag:
        if cfg_tag not in cfg_to_run:
            cfg_to_run[cfg_tag] = {
                "id": run.id,
                "state": run.state,
                "url": run.url,
                "html": run_id_to_latest_html.get(run.id)
            }

status_file = SCRIPT_DIR / "STATUS.md"
status_dir = status_file.parent

with open(status_file, "w") as f:
    f.write("# B11 Sweep Status\n\n")
    f.write("| Config | Status | Num Samples | WandB Link | Rel Link | Abs Link |\n")
    f.write("| --- | --- | --- | --- | --- | --- |\n")
    
    for cfg_file in all_cfg_files:
        cfg_name = cfg_file.stem
        with open(cfg_file, "r") as y:
            c = yaml.safe_load(y)
            num_samples = c['data']['num_samples']

        run_info = cfg_to_run.get(cfg_name)
        if run_info:
            wandb_link = f"[W&B]({run_info['url']})"
            abs_html = run_info['html']
            if abs_html:
                rel_html = os.path.relpath(abs_html, status_dir)
                html_rel_link = f"[HTML]({rel_html})"
                html_abs_link = f"[HTML]({abs_html})"
            else:
                html_rel_link = "-"
                html_abs_link = "-"
            
            f.write(f"| {cfg_name} | {run_info['state'].upper()} | {num_samples} | {wandb_link} | {html_rel_link} | {html_abs_link} |\n")
        else:
            f.write(f"| {cfg_name} | TODO | {num_samples} | - | - | - |\n")

print(f"Status file created at {status_file}")
