"""Generate B15 sweep configs: 5 dimensions upscaling (32, 64, 128, 256, 512).
Logarithmic flow depth scaling with increased width for 512D:
  - 32D  -> S=8,   hidden_units=128
  - 64D  -> S=10,  hidden_units=128
  - 128D -> S=12,  hidden_units=128
  - 256D -> S=14,  hidden_units=128
  - 512D -> S=16,  hidden_units=256

Fixed: dataset=nd-kf-mlp, shared_factor_dist=normal, mlp_depth=2, k_shared=3, m_unique=0.
"""
import yaml
from pathlib import Path


def generate_configs():
    # cfgs/ lives 4 levels up from this file
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    cfg_dir = base_dir / "cfgs" / "B15_capacity_scaling"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # (dim, layers, hidden_units)
    scale_params = [
        (32, 8, 128),
        (64, 10, 128),
        (128, 12, 128),
        (256, 14, 128),
        (512, 16, 256),
    ]

    generated = []
    for num_dims, stage_count, hidden_units in scale_params:
        config_name = f"B15_{num_dims}D_S{stage_count}"

        cfg = {
            "meta": {
                "seed": 12345,
                "device": "auto",
                "checkpoint_dir": "checkpoints",
            },
            "data": {
                "type": "nd-kf-mlp",
                "shared_factor_dist": "normal",
                "k_shared": 3,
                "m_unique": 0,
                "d_out": num_dims,
                "mlp_depth": 2,
                "num_samples": 1048576,
                "batch_size": 4096,
                "num_workers": 0,
                "manifold_noise_a": 1.0,
                "manifold_noise_b": 1.0,
                "asymmetric_noise_magnitude": 0.1,
                "noise_bbox_expansion": 0.25,
                "asym_corrupt_rate_a": 0.0,
                "asym_corrupt_rate_b": 0.0,
                "external_noise_ratio": 0.1,
            },
            "model": {
                "stage_count": stage_count,
                "num_dims": num_dims,
                "hidden_units": hidden_units,
                "predictor_type": "affine",
                "coupling_type": "affine",
                "coupling_clamp": 2.0,
                "affine_subnet_layers": 2,
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
                "pred_loss": "l1",
            },
            "optim": {
                "epochs": 100,
                "lr": 0.001,
            },
            "logging": {
                "log_wandb": True,
                "save_every": 100,
                "tqdm_silent": False,
                "wandb_group": "B15_capacity_scaling",
            },
        }

        cfg_path = cfg_dir / f"{config_name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"Generated {cfg_path.name}")
        generated.append(config_name)

    print(f"\nTotal: {len(generated)} configs")
    for name in generated:
        print(f"  {name}")


if __name__ == "__main__":
    generate_configs()
