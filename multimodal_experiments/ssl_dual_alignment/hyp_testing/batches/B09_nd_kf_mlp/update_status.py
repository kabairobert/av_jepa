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
all_cfg_files = sorted(list(cfg_dir.glob("B09_*.yaml")))

print(f"Querying WandB for B09 status...")
runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": ["B09_nd_kf_mlp"]}})

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
    cfg_tag = next((tag for tag in run.tags if tag.startswith("B09_")), None)
    if cfg_tag:
        cfg_to_run[cfg_tag] = {
            "id": run.id,
            "state": run.state,
            "url": run.url,
            "html": run_id_to_latest_html.get(run.id)
        }

status_file = SCRIPT_DIR / "STATUS.md"
status_dir = status_file.parent

with open(status_file, "w") as f:
    f.write("# B09 Sweep Status\n\n")
    f.write("| Config | Status | Dataset Config | Noise Params | Prior | Predictor | WandB Link | Rel Link | Abs Link |\n")
    f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    
    for cfg_file in all_cfg_files:
        cfg_name = cfg_file.stem
        with open(cfg_file, "r") as y:
            c = yaml.safe_load(y)
            noise = f"Asy:{c['data']['asymmetric_noise_rate_a']}/Ext:{c['data']['external_noise_ratio']}"
            prior = f"{c['loss']['prior_type']}" if c['loss']['lambda_prior'] > 0 else "None"
            pred = f"{c['loss']['pred_loss']}" if c['loss']['lambda_pred'] > 0 else "None"
            dims = f"D:{c['data']['d_out']}/k:{c['data']['k_shared']}/m:{c['data']['m_unique']}"

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
            
            f.write(f"| {cfg_name} | {run_info['state'].upper()} | {dims} | {noise} | {prior} | {pred} | {wandb_link} | {html_rel_link} | {html_abs_link} |\n")
        else:
            f.write(f"| {cfg_name} | TODO | {dims} | {noise} | {prior} | {pred} | - | - | - |\n")

print(f"Status file created at {status_file}")
