import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from examples.video_jepa.main import run, load_config

if __name__ == '__main__':
    cfg = load_config('examples/video_jepa/cfgs/default.yaml', None)
    # Apply safe overrides for a CPU smoke test
    try:
        cfg.meta.device = 'cpu'
    except Exception:
        pass
    try:
        cfg.logging.log_wandb = False
    except Exception:
        pass
    try:
        cfg.optim.epochs = 1
    except Exception:
        pass
    try:
        cfg.data.batch_size = 1
    except Exception:
        pass
    try:
        if 'training' not in cfg:
            cfg.training = {}
        cfg.training['use_amp'] = False
    except Exception:
        pass

    # Run
    run(cfg=cfg)
