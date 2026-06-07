"""B11 status update — thin wrapper over shared update_status_lib."""
import sys
from pathlib import Path

# Allow importing from hyp_testing/scripts without installing as a package
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from update_status_lib import generate_status_md  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent.parent.parent
cfg_dir = SCRIPT_DIR.parent.parent.parent / "cfgs" / "B11_data_scaling"


def _b11_params(cfg: dict) -> str:
    return str(cfg['data']['num_samples'])


generate_status_md(
    batch_tag="B11_data_scaling",
    batch_id_str="B11",
    cfg_files=sorted(cfg_dir.glob("B11_*.yaml")),
    status_file=SCRIPT_DIR / "STATUS.md",
    checkpoint_root=ROOT_DIR / "checkpoints" / "sslda",
    cfg_to_params_fn=_b11_params,
)
