import os
import yaml
from pathlib import Path

def generate_configs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    cfg_dir = base_dir / "cfgs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid parameters
    scales = {
        "1x":  {"num_samples": 4096,   "epochs": 1280},
        "2x":  {"num_samples": 8192,   "epochs": 640},
        "4x":  {"num_samples": 16384,  "epochs": 320},
        "8x":  {"num_samples": 32768,  "epochs": 160},
        "16x": {"num_samples": 65536,  "epochs": 80},
        "32x": {"num_samples": 131072, "epochs": 40},
    }
    
    embed_types = {
        "R": "3d-3f-2c-rot",
        "M": "3d-3f-2c-mlp"
    }
    
    dims = ["10", "20"]
    
    for scale_name, scale_params in scales.items():
        for e_key, e_val in embed_types.items():
            for d in dims:
                config_name = f"B11_{scale_name}_{e_key}{d}_N1P21"
                
                cfg = {
                    "meta": {
                        "seed": 12345,
                        "device": "auto",
                        "checkpoint_dir": "checkpoints"
                    },
                    "data": {
                        "type": e_val,
                        "embed_dim": int(d),
                        "num_samples": scale_params["num_samples"],
                        "batch_size": 512,
                        "num_workers": 0,
                        "asymmetric_noise_magnitude": 0.1,
                        "noise_bbox_expansion": 0.25,
                        "u3a_scale": 0.12,
                        "u3b_scale": 0.12,
                        "turns": 1.0,
                        "wave_amplitude": 1.0,
                        "manifold_noise_a": 0.02,
                        "manifold_noise_b": 0.02,
                        "asymmetric_noise_rate_a": 0.0,
                        "asymmetric_noise_rate_b": 0.0,
                        "external_noise_ratio": 0.1
                    },
                    "model": {
                        "stage_count": 6,
                        "num_dims": int(d),
                        "hidden_units": 128,
                        "predictor_type": "affine"
                    },
                    "loss": {
                        "type": "ebm",
                        "lambda_jac": 1.0,
                        "congruence_mode": "none",
                        "congruence_tau": 0.5,
                        "lambda_prior": 0.5,
                        "lambda_sparse": 0.1,
                        "prior_type": "l2",
                        "lambda_pred": 1.0,
                        "pred_loss": "l1"
                    },
                    "optim": {
                        "epochs": scale_params["epochs"],
                        "lr": 0.001
                    },
                    "logging": {
                        "log_wandb": True,
                        "save_every": scale_params["epochs"],  # Save at the end
                        "tqdm_silent": False,
                        "wandb_group": "B11_data_scaling"
                    }
                }
                
                with open(cfg_dir / f"{config_name}.yaml", "w") as f:
                    yaml.dump(cfg, f, sort_keys=False)
                    
                print(f"Generated {config_name}.yaml")

if __name__ == "__main__":
    generate_configs()
