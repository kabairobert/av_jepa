"""
Run the `_capture_shapes` helper from examples.video_jepa.main on a single val sample.
"""
import sys
sys.path.append('.')
from examples.video_jepa import main as m
from eb_jepa.training_utils import load_config
from eb_jepa.architectures import ResNet5, Projector
from eb_jepa.datasets.moving_mnist import MovingMNISTDet

cfg = load_config('examples/video_jepa/cfgs/default.yaml')
import torch
device = torch.device('cpu')
encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc).to(device)
proj_hidden = cfg.model.dstc * cfg.loss.get('proj_hidden_mult',4)
proj_out = cfg.model.dstc * cfg.loss.get('proj_out_mult',4)
projector = Projector(f"{cfg.model.dstc}-{proj_hidden}-{proj_out}").to(device)

ds = MovingMNISTDet(split='val')
print('Dataset len:', len(ds))
sample = ds[0]
x = sample['video'].to(device)
print('sample video shape', list(x.shape))
shapes = m._capture_shapes(x, encoder, projector, device)
print('captured shapes:', shapes)
