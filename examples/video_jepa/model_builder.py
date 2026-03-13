import torch
import torch.nn as nn

from eb_jepa.architectures import DetHead, Projector, ResNet5, ResUNet, StateOnlyPredictor
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import SquareLossSeq, VCLoss, VideoJEPA_BCS, VideoJEPA_BCS_Euler_Scaleinvariant


def build_video_jepa_and_probes(cfg, device):
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor_model = ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)

    loss_type = cfg.loss.get("type", "vcreg")
    bcs_types = ("bcs", "bcs-euler-scalefree")
    if loss_type in bcs_types:
        default_h_mult, default_o_mult = 4, 1
    else:
        default_h_mult, default_o_mult = 4, 4

    h_mult = cfg.loss.get("proj_hidden_mult", default_h_mult)
    o_mult = cfg.loss.get("proj_out_mult", default_o_mult)
    proj_hidden = cfg.model.dstc * h_mult
    proj_out = cfg.model.dstc * o_mult
    projector = Projector(f"{cfg.model.dstc}-{proj_hidden}-{proj_out}")

    if loss_type == "bcs":
        regularizer = VideoJEPA_BCS(
            num_slices=cfg.loss.get("num_slices"),
            lmbd=cfg.loss.get("lmbd"),
            proj=projector,
        )
    elif loss_type == "bcs-euler-scalefree":
        regularizer = VideoJEPA_BCS_Euler_Scaleinvariant(
            num_slices=cfg.loss.get("num_slices"),
            lmbd=cfg.loss.get("lmbd"),
            proj=projector,
        )
    else:
        regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)

    jepa = JEPA(encoder, encoder, predictor, regularizer, SquareLossSeq(projector)).to(device)

    decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs)
    dethead = DetHead(cfg.model.dstc, cfg.model.hpre, cfg.model.dobs)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
    detection_head = JEPAProbe(jepa, dethead, nn.BCEWithLogitsLoss()).to(device)

    return {
        "jepa": jepa,
        "pixel_decoder": pixel_decoder,
        "detection_head": detection_head,
        "encoder": encoder,
        "predictor": predictor,
        "projector": projector,
        "regularizer": regularizer,
    }
