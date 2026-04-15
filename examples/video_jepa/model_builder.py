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
from eb_jepa.jepa import JEPA, JEPAProbe, MultiSourceJEPAProbe
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


def _resolve_model_routing(cfg, loss_type, proj_out):
    predictor_type = cfg.model.get("predictor_type", "resunet")
    predictor_space = cfg.model.get("predictor_space", "encoder")

    has_predictor_space = False
    has_probe_source = False
    try:
        has_predictor_space = "predictor_space" in cfg.model
    except Exception:
        has_predictor_space = False
    try:
        has_probe_source = "probe_source" in cfg.model
    except Exception:
        has_probe_source = False

    if has_probe_source:
        probe_source = cfg.model.get("probe_source")
    elif has_predictor_space:
        probe_source = predictor_space
    else:
        probe_source = "encoder"

    if predictor_space not in ("encoder", "projector"):
        raise ValueError(
            f"Unknown model.predictor_space='{predictor_space}'. Supported: encoder, projector."
        )
    if probe_source not in ("encoder", "projector", "both"):
        raise ValueError(
            f"Unknown model.probe_source='{probe_source}'. Supported: encoder, projector, both."
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
        predictor_in_d = 2 * proj_out
        predictor_out_d = proj_out
    else:
        predictor_in_d = 2 * cfg.model.dstc
        predictor_out_d = cfg.model.dstc

    return {
        "predictor_type": predictor_type,
        "predictor_space": predictor_space,
        "probe_source": probe_source,
        "predictor_in_d": predictor_in_d,
        "predictor_out_d": predictor_out_d,
        "probe_dims": {
            "encoder": cfg.model.dstc,
            "projector": proj_out,
        },
    }


def _build_probe_set(cfg, jepa, device, source, probe_dim):
    decoder = ImageDecoder(probe_dim, cfg.model.dobs)
    dethead = DetHead(probe_dim, cfg.model.hpre, cfg.model.dobs)
    pixel_decoder = JEPAProbe(
        jepa, decoder, nn.MSELoss(), feature_source=source
    ).to(device)
    detection_head = JEPAProbe(
        jepa, dethead, nn.BCEWithLogitsLoss(), feature_source=source
    ).to(device)
    return pixel_decoder, detection_head


def build_video_jepa_and_probes(cfg, device):
    # Stage 1: Build encoder.
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)

    # Stage 2: Resolve projector dimensions with centralized policy.
    loss_type = cfg.loss.get("type", "vcreg")
    proj_hidden, proj_out = resolve_projector_dims_from_cfg(cfg, loss_type)
    projector = Projector(f"{cfg.model.dstc}-{proj_hidden}-{proj_out}")

    # Stage 3: Resolve routing and build predictor family.
    routing = _resolve_model_routing(cfg, loss_type=loss_type, proj_out=proj_out)
    predictor_type = routing["predictor_type"]
    predictor_space = routing["predictor_space"]
    predictor_in_d = routing["predictor_in_d"]
    predictor_out_d = routing["predictor_out_d"]

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

    probe_source = routing["probe_source"]
    probe_dims = routing["probe_dims"]
    active_probe_source = predictor_space
    if probe_source == "both":
        enc_pixel, enc_det = _build_probe_set(
            cfg, jepa, device, source="encoder", probe_dim=probe_dims["encoder"]
        )
        proj_pixel, proj_det = _build_probe_set(
            cfg, jepa, device, source="projector", probe_dim=probe_dims["projector"]
        )
        pixel_decoder = MultiSourceJEPAProbe(
            {"encoder": enc_pixel, "projector": proj_pixel},
            active_source=active_probe_source,
        ).to(device)
        detection_head = MultiSourceJEPAProbe(
            {"encoder": enc_det, "projector": proj_det},
            active_source=active_probe_source,
        ).to(device)
    else:
        pixel_decoder, detection_head = _build_probe_set(
            cfg,
            jepa,
            device,
            source=probe_source,
            probe_dim=probe_dims[probe_source],
        )
        active_probe_source = probe_source

    return {
        "jepa": jepa,
        "pixel_decoder": pixel_decoder,
        "detection_head": detection_head,
        "encoder": encoder,
        "predictor": predictor,
        "projector": projector,
        "regularizer": regularizer,
        "predictor_space": predictor_space,
        "probe_source": probe_source,
        "active_probe_source": active_probe_source,
    }
