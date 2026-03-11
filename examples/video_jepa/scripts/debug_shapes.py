"""
Debug script: run encoder + projector on a single validation sample (CPU)
Saves full traceback for encoder/projector failures.
Run with: .venv\Scripts\python.exe examples/video_jepa/scripts/debug_shapes.py
"""

import sys
import traceback
import torch

sys.path.append('.')

from eb_jepa.training_utils import load_config
from eb_jepa.architectures import ResNet5, Projector
from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from torch.utils.data import DataLoader


def main(cfg_path='examples/video_jepa/cfgs/default.yaml'):
    cfg = load_config(cfg_path)
    device = torch.device('cpu')
    print('Using device:', device)

    # Build models
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc).to(device)
    proj_hidden = cfg.model.dstc * cfg.loss.get('proj_hidden_mult', 4)
    proj_out = cfg.model.dstc * cfg.loss.get('proj_out_mult', 4)
    projector = Projector(f"{cfg.model.dstc}-{proj_hidden}-{proj_out}").to(device)

    # Dataset
    ds = MovingMNISTDet(split='val')
    print('Validation dataset length:', len(ds))

    # Try single-sample access (no DataLoader iterator side-effects)
    try:
        sample = ds[0]
        batch = {k: v.to(device) for k, v in sample.items()}
        x = batch.get('video')
        print('raw x:', list(x.shape), x.dtype, x.device)
    except Exception as e:
        print('Failed loading sample from dataset:')
        traceback.print_exc()
        return

    # Run encoder
    try:
        with torch.no_grad():
            enc_out = encoder(x)
        print('encoder output shape:', list(enc_out.shape))
    except Exception:
        print('Encoder forward failed:')
        traceback.print_exc()
        enc_out = None

    # Prepare projector input and run projector.shape_str
    if enc_out is None:
        print('Skipping projector because encoder failed.')
        return

    proj_in = enc_out
    if proj_in.dim() > 2:
        proj_in = proj_in.view(proj_in.size(0), -1)
    try:
        with torch.no_grad():
            s = projector.shape_str(proj_in)
        print('projector.shape_str ->', s)
    except Exception:
        print('Projector.shape_str failed:')
        traceback.print_exc()
        try:
            with torch.no_grad():
                out = projector(proj_in)
            print('projector forward out shape:', list(out.shape))
        except Exception:
            print('Projector forward failed as well:')
            traceback.print_exc()


if __name__ == '__main__':
    main()
