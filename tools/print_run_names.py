from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eb_jepa.training_utils import load_config, get_exp_name


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
            cfg = load_config(p, quiet=True)
            run_name = get_exp_name(name, cfg)
            print(f"{p} -> {run_name}")
