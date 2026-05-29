import os
import yaml
from pathlib import Path

def generate_configs():
    # Target directory is the cfgs folder under ssl_dual_alignment
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    cfg_dir = base_dir / "cfgs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    
    # Scaling parameters: (Scale Name, Num Samples, Batch Size)
    scales = [
        ("1x", 4096, 128),
        ("2x", 8192, 256),
        ("4x", 16384, 512),
        ("8x", 32768, 1024),
        ("16x", 65536, 2048),
        ("32x", 131072, 4096),
        ("64x", 262144, 4096),
        ("128x", 524288, 4096),
        ("256x", 1048576, 4096)
    ]
    
    for scale_name, num_samples, batch_size in scales:
        config_name = f"B12_{scale_name}_M20_N1P21"
        
        cfg = {
            "meta": {
                "seed": 12345,
                "device": "auto",
                "checkpoint_dir": "checkpoints"
            },
            "data": {
                "type": "3d-3f-2c-mlp",
                "embed_dim": 20,
                "num_samples": num_samples,
                "batch_size": batch_size,
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
                "num_dims": 20,
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
                "epochs": 150,
                "lr": 0.001
            },
            "logging": {
                "log_wandb": True,
                "save_every": 150,
                "tqdm_silent": False,
                "wandb_group": "B12_data_scaling"
            }
        }
        
        cfg_path = cfg_dir / f"{config_name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
            
        print(f"Generated {cfg_path.name}")

if __name__ == "__main__":
    generate_configs()
