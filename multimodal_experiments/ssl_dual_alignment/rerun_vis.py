"""
Reruns 3D interactive visualization generation for training checkpoints.

How it works:
1. Scans checkpoint directories (`checkpoints/sslda/`) matching a batch prefix.
2. Loads config.yaml, rebuilds dataset, and builds model from each run.
3. Loads latest model checkpoint, processes data, and projects inputs/outputs.
4. Generates HTML interactive 3D visualizations (input/output space).
5. Saves HTML files locally under `hyp_testing/batches/<batch_prefix>_capacity_scaling_fixed/rerun_vis_<timestamp>/`.
"""
import os
import sys
import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from eb_jepa.training_utils import load_config, setup_device, load_checkpoint
from multimodal_experiments.ssl_dual_alignment.dataset import PointType, build_dataset_from_config
from multimodal_experiments.ssl_dual_alignment.model_builder import build_model_and_predictors
from multimodal_experiments.ssl_dual_alignment.vis import build_interactive_input3d_html, build_interactive_output3d_html, to_numpy

def discover_checkpoints(run_dir):
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("epoch_*.pth.tar"))
    latest = run_dir / "latest.pth.tar"
    if latest.exists():
        if not ckpts:
            ckpts.append(latest)
        else:
            newest_epoch = max(ckpts, key=lambda p: p.stat().st_mtime)
            if latest.stat().st_mtime > newest_epoch.stat().st_mtime:
                ckpts.append(latest)
    return ckpts

def process_batch(batch_prefix):
    script_dir = Path(__file__).resolve().parent
    checkpoint_root = script_dir.parents[1] / "checkpoints" / "sslda"
    
    # Find all run directories for this batch
    run_dirs = []
    for p in checkpoint_root.rglob("config.yaml"):
        if p.parent.name.startswith(batch_prefix):
            run_dirs.append(p.parent)
    
    if not run_dirs:
        print(f"No runs found for {batch_prefix}")
        return
        
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = script_dir / "hyp_testing" / "batches" / f"{batch_prefix}_capacity_scaling_fixed" / f"rerun_vis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = setup_device("cuda" if torch.cuda.is_available() else "cpu")
    
    for run_dir in tqdm(run_dirs, desc=f"Processing {batch_prefix}"):
        cfg_path = run_dir / "config.yaml"
        cfg = load_config(str(cfg_path))
        
        # Build dataset
        data_cfg = cfg.data
        eval_num_samples = int(data_cfg.get('eval_num_samples', 4096))
        eval_data_cfg_overrides = OmegaConf.create({'num_samples': eval_num_samples})
        eval_cfg = OmegaConf.merge(cfg, OmegaConf.create({'data': eval_data_cfg_overrides}))
        eval_set = build_dataset_from_config(eval_cfg, seed=cfg.meta.seed)
        
        is_3d = (eval_set.data_a.shape[1] >= 3)
        if not is_3d:
            continue
            
        # Build model
        built = build_model_and_predictors(cfg, device)
        full_model = built["full_model"]
        dual_model = built["dual_model"]
        predictor_a2b = built["predictor_a2b"]
        
        # Load latest checkpoint
        ckpts = discover_checkpoints(run_dir)
        if not ckpts:
            print(f"No checkpoints found in {run_dir}")
            continue
            
        ckpt = ckpts[-1]
        load_checkpoint(ckpt, full_model, optimizer=None, device=device)
        dual_model.eval()
        
        vis_cfg = cfg.get("visualization", {})
        point_size_min = vis_cfg.get("point_size_min", 4.0)
        point_size_max = vis_cfg.get("point_size_max", 20.0)
        point_size_default = vis_cfg.get("point_size_default", 5.0)
        
        max_interactive_points = 2000
        
        data_a = to_numpy(eval_set.data_a)
        data_b = to_numpy(eval_set.data_b)
        param_values = to_numpy(eval_set.param_values)
        idxs = None
        if data_a.shape[0] > max_interactive_points:
            idxs = np.random.choice(data_a.shape[0], size=max_interactive_points, replace=False)
            data_a = data_a[idxs]
            data_b = data_b[idxs]
            param_values = param_values[idxs]
            
        with torch.no_grad():
            out_a, _ = dual_model.model_a(torch.tensor(data_a, device=device, dtype=torch.float32))
            out_b, _ = dual_model.model_b(torch.tensor(data_b, device=device, dtype=torch.float32))
        out_a = out_a.detach().cpu().numpy()
        out_b = out_b.detach().cpu().numpy()
        
        pt_a = getattr(eval_set, "point_type_a", None)
        pt_b = getattr(eval_set, "point_type_b", None)
        if pt_a is not None:
            pt_a = to_numpy(pt_a)
            pt_b = to_numpy(pt_b)
            if idxs is not None:
                pt_a = pt_a[idxs]
                pt_b = pt_b[idxs]
                
        html_input = build_interactive_input3d_html(
            data_a, data_b, np.asarray(param_values),
            point_type_a=pt_a,
            point_type_b=pt_b,
            min_height_px=420,
            point_size_min=point_size_min,
            point_size_max=point_size_max,
            point_size_default=3.0 * (point_size_default / 5.0),
        )
        html_output = build_interactive_output3d_html(
            out_a, out_b, np.asarray(param_values),
            point_type_a=pt_a,
            point_type_b=pt_b,
            min_height_px=420,
            predictor_a2b=predictor_a2b,
            point_size_min=point_size_min,
            point_size_max=point_size_max,
            point_size_default=3.0 * (point_size_default / 5.0),
        )
        
        if html_input is not None:
            out_path_input = out_dir / f"{run_dir.name}_interactive_input3d.html"
            with open(out_path_input, "w") as f:
                f.write(html_input)
        if html_output is not None:
            out_path_output = out_dir / f"{run_dir.name}_interactive_output3d.html"
            with open(out_path_output, "w") as f:
                f.write(html_output)
                
    print(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    process_batch("B16b")
    process_batch("B16c")
