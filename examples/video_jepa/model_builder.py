import torch
import torch.nn as nn

from eb_jepa.architectures import (
    DetHead,
    LinearNet,
    MLPNet,
    Projector,
    ResNet5,
    ResUNet,
    StateOnlyPredictor,
)
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import (
    BCS_LOSS_TYPES,
    SquareLossSeq,
    VCLoss,
    VideoJEPA_BCS,
    VideoJEPA_BCS_Euler_Scaleinvariant,
)
from eb_jepa.training_utils import resolve_projector_dims_from_cfg


def _build_predictor_model(cfg, predictor_type, in_d, out_d):
    if predictor_type == "resunet":
        return ResUNet(in_d, cfg.model.hpre, out_d)
    if predictor_type == "mlpnet":
        hidden = cfg.model.get("mlp_hidden_dim", cfg.model.hpre)
        return MLPNet(in_d, hidden, out_d)
    if predictor_type == "linearnet":
        return LinearNet(in_d, out_d)
    raise ValueError(
        f"Unknown model.predictor_type='{predictor_type}'. "
        "Supported: resunet, mlpnet, linearnet."
    )


def build_video_jepa_and_probes(cfg, device):
    # Stage 1: Build encoder.
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)

    # Stage 2: Resolve projector dimensions with centralized policy.
    loss_type = cfg.loss.get("type", "vcreg")
    proj_hidden, proj_out = resolve_projector_dims_from_cfg(cfg, loss_type)
    projector = Projector(f"{cfg.model.dstc}-{proj_hidden}-{proj_out}")

    # Stage 3: Build predictor family and select predictor space.
    predictor_type = cfg.model.get("predictor_type", "resunet")
    predictor_space = cfg.model.get("predictor_space", "encoder")
    if predictor_space not in ("encoder", "projector"):
        raise ValueError(
            f"Unknown model.predictor_space='{predictor_space}'. Supported: encoder, projector."
        )
    if predictor_space == "projector":
        if loss_type not in BCS_LOSS_TYPES:
            raise ValueError(
                "model.predictor_space='projector' currently supports only BCS-family losses "
                f"{BCS_LOSS_TYPES}; got loss.type='{loss_type}'."
            )
        if loss_type != "bcs-euler-scalefree":
            raise ValueError(
                "model.predictor_space='projector' is currently supported only when "
                "loss.type='bcs-euler-scalefree'."
            )

    if predictor_space == "projector":
        # StateOnlyPredictor concatenates (prev_state, next_state) on channels,
        # so predictor input width is 2x the single-state channel width.
        predictor_in_d = 2 * proj_out
        predictor_out_d = proj_out
    else:
        # Same 2x reason as above, now in encoder feature space.
        predictor_in_d = 2 * cfg.model.dstc
        predictor_out_d = cfg.model.dstc

    predictor_model = _build_predictor_model(
        cfg, predictor_type=predictor_type, in_d=predictor_in_d, out_d=predictor_out_d
    )
    predictor = StateOnlyPredictor(predictor_model, context_length=2)

    # Stage 4: Build regularizer.
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

    # Stage 5: Assemble JEPA + probes.
    predcost_proj = projector if predictor_space == "encoder" else nn.Identity()
    jepa = JEPA(
        encoder,
        encoder,
        predictor,
        regularizer,
        SquareLossSeq(predcost_proj),
        predictor_space=predictor_space,
        predictor_proj=projector,
    ).to(device)

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
