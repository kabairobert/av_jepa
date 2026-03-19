from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eb_jepa.training_utils import load_config, get_exp_name


def _missing_training_keys(cfg):
    required = ("data", "model", "loss", "optim")
    return [key for key in required if key not in cfg]


def print_for(cfg_path):
    cfg = load_config(cfg_path, quiet=True)
    name = get_exp_name("video_jepa", cfg)
    print(f"{cfg_path.name} -> {name}")


if __name__ == "__main__":
    targets = [
        ("video_jepa", Path("examples/video_jepa/cfgs")),
        ("image_jepa", Path("examples/image_jepa/cfgs")),
    ]
    for name, cfg_dir in targets:
        if not cfg_dir.exists():
            print(f"configs directory not found: {cfg_dir}")
            continue
        print(f"\n=== {name} configs ===")
        for p in sorted(cfg_dir.glob("*.yaml")):
            try:
                cfg = load_config(p, quiet=True)
            except Exception as exc:
                print(f"{p} -> SKIP (failed to load config: {exc})")
                continue

            # Some YAMLs in cfg folders are eval/logging overrides only.
            # get_exp_name expects a full training config shape.
            missing_keys = _missing_training_keys(cfg)
            if missing_keys:
                print(f"{p} -> SKIP (non-training config, missing: {', '.join(missing_keys)})")
                continue

            try:
                run_name = get_exp_name(name, cfg)
                print(f"{p} -> {run_name}")
            except Exception as exc:
                print(f"{p} -> SKIP (run-name error: {exc})")
