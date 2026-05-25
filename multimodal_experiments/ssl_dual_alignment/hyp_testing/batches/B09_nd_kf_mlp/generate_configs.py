import os
from pathlib import Path
import yaml

# Resolve config directory dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR.parent.parent.parent / "cfgs"
os.makedirs(CFG_DIR, exist_ok=True)

dataset_configs = {
    "D0": {"d_out": 3, "k_shared": 2, "m_unique": 1, "num_dims": 3, "hidden_units": 128},
    "D1": {"d_out": 10, "k_shared": 2, "m_unique": 8, "num_dims": 10, "hidden_units": 256},
    "D2": {"d_out": 10, "k_shared": 5, "m_unique": 5, "num_dims": 10, "hidden_units": 256},
    "D3": {"d_out": 20, "k_shared": 5, "m_unique": 15, "num_dims": 20, "hidden_units": 512},
}

noise_levels = {
    1: {"asymmetric_noise_rate_a": 0.05, "asymmetric_noise_rate_b": 0.05, "external_noise_ratio": 0.00},
    2: {"asymmetric_noise_rate_a": 0.15, "asymmetric_noise_rate_b": 0.15, "external_noise_ratio": 0.00},
    3: {"asymmetric_noise_rate_a": 0.00, "asymmetric_noise_rate_b": 0.00, "external_noise_ratio": 0.30},
}

prior_levels = {
    0: {"lambda_prior": 0.0, "lambda_sparse": 0.0, "prior_type": "l1"},
    1: {"lambda_prior": 0.5, "lambda_sparse": 0.1, "prior_type": "l1"},
    2: {"lambda_prior": 0.5, "lambda_sparse": 0.1, "prior_type": "l2"},
}

pred_levels = {
    0: {"lambda_pred": 0.0, "pred_loss": "l1"},
    1: {"lambda_pred": 1.0, "pred_loss": "l1"},
    2: {"lambda_pred": 1.0, "pred_loss": "l2"},
}

def get_base_cfg(d_code, d_params):
    return {
        "meta": {
            "seed": 12345,
            "device": "auto",
            "checkpoint_dir": "checkpoints"
        },
        "data": {
            "type": "nd-kf-mlp",
            "num_samples": 4096,
            "batch_size": 128,
            "num_workers": 0,
            "asymmetric_noise_magnitude": 0.1,
            "noise_bbox_expansion": 0.25,
            "d_out": d_params["d_out"],
            "k_shared": d_params["k_shared"],
            "m_unique": d_params["m_unique"],
            "manifold_noise_a": 0.02,
            "manifold_noise_b": 0.02
        },
        "model": {
            "stage_count": 6,
            "num_dims": d_params["num_dims"],
            "hidden_units": d_params["hidden_units"],
            "predictor_type": "affine"
        },
        "loss": {
            "type": "ebm",
            "lambda_jac": 1.0,
            "congruence_mode": "none",
            "congruence_tau": 0.5
        },
        "optim": {
            "epochs": 150,
            "lr": 1.0e-3
        },
        "logging": {
            "log_wandb": True,
            "save_every": 150,
            "tqdm_silent": False,
            "wandb_group": "B09_nd_kf_mlp"
        }
    }

def main():
    count = 0
    for d_code, d_params in dataset_configs.items():
        for n_idx, n_params in noise_levels.items():
            for p1_idx, p1_params in prior_levels.items():
                for p2_idx, p2_params in pred_levels.items():
                    # Skip Prior: None, Predictor: None
                    if p1_idx == 0 and p2_idx == 0:
                        continue
                        
                    cfg = get_base_cfg(d_code, d_params)
                    cfg["data"].update(n_params)
                    cfg["loss"].update(p1_params)
                    cfg["loss"].update(p2_params)
                    
                    name = f"B09_{d_code}_N{n_idx}P{p1_idx}{p2_idx}.yaml"
                    filepath = CFG_DIR / name
                    
                    with open(filepath, 'w') as f:
                        yaml.dump(cfg, f, sort_keys=False)
                    count += 1
                        
    print(f"Generated {count} B09 configuration files in {CFG_DIR}")

if __name__ == "__main__":
    main()
