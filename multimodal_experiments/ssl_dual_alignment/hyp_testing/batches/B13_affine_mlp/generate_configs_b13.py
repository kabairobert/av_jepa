"""Generate B13 sweep configs: 3 data scales × 2 flow depths = 6 configs.

Fixed: coupling_type=affine, hidden_units=128, lambda_jac=1.0, lambda_prior=0.5,
       prior_type=l2, mlp_depth=2 (2-layer MLP distortion).
Sweep: dataset_multiplier ∈ {1x, 16x, 256x}, stage_count ∈ {6, 12}.
"""
import yaml
from pathlib import Path


def generate_configs():
    # cfgs/ lives 4 levels up from this file
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    cfg_dir = base_dir / "cfgs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # (scale_name, num_samples, batch_size)
    scales = [
        ("1x",   4_096,     128),
        ("16x",  65_536,  2_048),
        ("256x", 1_048_576, 4_096),
    ]

    stage_counts = [6, 12]

    generated = []
    for scale_name, num_samples, batch_size in scales:
        for stage_count in stage_counts:
            config_name = f"B13_{scale_name}_S{stage_count}"

            cfg = {
                "meta": {
                    "seed": 12345,
                    "device": "auto",
                    "checkpoint_dir": "checkpoints",
                },
                "data": {
                    "type": "3d-3f-2c-mlp",
                    "embed_dim": 20,
                    "mlp_depth": 2,
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
                    "external_noise_ratio": 0.1,
                },
                "model": {
                    "stage_count": stage_count,
                    "num_dims": 20,
                    "hidden_units": 128,
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
                    "epochs": 150,
                    "lr": 0.001,
                },
                "logging": {
                    "log_wandb": True,
                    "save_every": 150,
                    "tqdm_silent": False,
                    "wandb_group": "B13_affine_mlp",
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
