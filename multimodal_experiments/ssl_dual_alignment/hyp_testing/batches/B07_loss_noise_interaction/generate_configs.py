import os
import yaml

CFG_DIR = "/gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/cfgs"
os.makedirs(CFG_DIR, exist_ok=True)

noise_levels = {
    1: {"asymmetric_noise_rate_a": 0.15, "asymmetric_noise_rate_b": 0.15, "external_noise_ratio": 0.00, "manifold_noise_a": 0.02, "manifold_noise_b": 0.02},
    2: {"asymmetric_noise_rate_a": 0.25, "asymmetric_noise_rate_b": 0.25, "external_noise_ratio": 0.00, "manifold_noise_a": 0.02, "manifold_noise_b": 0.02},
    3: {"asymmetric_noise_rate_a": 0.375, "asymmetric_noise_rate_b": 0.375, "external_noise_ratio": 0.00, "manifold_noise_a": 0.02, "manifold_noise_b": 0.02},
    4: {"asymmetric_noise_rate_a": 0.00, "asymmetric_noise_rate_b": 0.00, "external_noise_ratio": 0.50, "manifold_noise_a": 0.02, "manifold_noise_b": 0.02},
    5: {"asymmetric_noise_rate_a": 0.09, "asymmetric_noise_rate_b": 0.09, "external_noise_ratio": 0.75, "manifold_noise_a": 0.02, "manifold_noise_b": 0.02},
    6: {"asymmetric_noise_rate_a": 0.25, "asymmetric_noise_rate_b": 0.25, "external_noise_ratio": 0.00, "manifold_noise_a": 0.04, "manifold_noise_b": 0.04},
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

def get_base_cfg():
    return {
        "meta": {
            "seed": 12345,
            "device": "auto",
            "checkpoint_dir": "checkpoints"
        },
        "data": {
            "type": "3d-2f-common",
            "num_samples": 4096,
            "batch_size": 128,
            "num_workers": 0,
            "asymmetric_noise_magnitude": 0.1,
            "noise_bbox_expansion": 0.25
        },
        "model": {
            "stage_count": 6,
            "num_dims": 3,
            "hidden_units": 128,
            "predictor_type": "affine"
        },
        "loss": {
            "type": "ebm",
            "lambda_jac": 1.0,
            "congruence_mode": "none",
            "congruence_tau": 0.5
        },
        "optim": {
            "epochs": 200,
            "lr": 1.0e-3
        },
        "logging": {
            "log_wandb": True,
            "save_every": 50,
            "tqdm_silent": False
        }
    }

def main():
    count = 0
    for n_idx, n_params in noise_levels.items():
        for p1_idx, p1_params in prior_levels.items():
            for p2_idx, p2_params in pred_levels.items():
                cfg = get_base_cfg()
                cfg["data"].update(n_params)
                cfg["loss"].update(p1_params)
                cfg["loss"].update(p2_params)
                
                name = f"B07_NPP{n_idx}{p1_idx}{p2_idx}.yaml"
                filepath = os.path.join(CFG_DIR, name)
                
                with open(filepath, 'w') as f:
                    yaml.dump(cfg, f, sort_keys=False)
                count += 1
                    
    print(f"Generated {count} configuration files.")

if __name__ == "__main__":
    main()