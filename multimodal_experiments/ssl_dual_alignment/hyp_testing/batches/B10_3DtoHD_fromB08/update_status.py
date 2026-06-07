"""B10 status update — thin wrapper over shared update_status_lib."""
import sys
from pathlib import Path

# Allow importing from hyp_testing/scripts without installing as a package
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from update_status_lib import generate_status_md  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent.parent
cfg_dir = SCRIPT_DIR.parent.parent.parent / "cfgs" / "B10_3DtoHD_fromB08"


def _b10_params(cfg: dict) -> str:
    noise = f"Asy:{cfg['data']['asymmetric_noise_rate_a']}/Ext:{cfg['data']['external_noise_ratio']}"
    prior = f"Pri:{cfg['loss']['prior_type']}" if cfg['loss']['lambda_prior'] > 0 else "Pri:None"
    pred = f"Pre:{cfg['loss']['pred_loss']}" if cfg['loss']['lambda_pred'] > 0 else "Pre:None"
    return f"{noise}, {prior}, {pred}"


generate_status_md(
    batch_tag="B10_3DtoHD_fromB08",
    batch_id_str="B10",
    cfg_files=sorted(cfg_dir.glob("B10_[RM]*.yaml")),
    status_file=SCRIPT_DIR / "STATUS.md",
    checkpoint_root=ROOT_DIR / "checkpoints" / "sslda",
    cfg_to_params_fn=_b10_params,
)
