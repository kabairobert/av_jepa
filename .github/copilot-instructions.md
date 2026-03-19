# EB-JEPA: Copilot Agent Instructions

PyTorch Energy-Based JEPA for video/action embodied learning.

## Core Files

- Training: [examples/video_jepa/main.py](examples/video_jepa/main.py)
- Eval: [examples/video_jepa/eval.py](examples/video_jepa/eval.py)
- Models: [eb_jepa/jepa.py](eb_jepa/jepa.py), [eb_jepa/architectures.py](eb_jepa/architectures.py)
- Losses: [eb_jepa/losses.py](eb_jepa/losses.py)

## Tools

Details: [.github/instructions/tools.instructions.md](.github/instructions/tools.instructions.md)

- `python tools/wandb_inspect_run.py --run <id_or_path>`
- `python tools/wandb_cleanup.py --run <entity/project/run_id>`
- `python tools/resolve_diagnostics.py <run_dir>`
- `uv run python tools/print_run_names.py`

## Epoch Rule

- Internal: 0-based `epoch_idx`
- User-facing: 1-based `epoch_display` / `epoch_completed`
- `epoch_10` means 10 completed epochs
- Eval must read both new (`epoch_idx`) and legacy (`epoch`) checkpoints

## Checkpoint Fields

- `epoch_idx`: 0-based resume index
- `epoch_completed`: 1-based completed count
- `epoch`: legacy fallback
- `step`: global optimizer step
- `steps_per_epoch`: step/epoch alignment

## Diagnostics

- Path: `<run_dir>/diagnostics/index.json`
- Schema: [examples/video_jepa/diagnostics.py](examples/video_jepa/diagnostics.py)

Updated: March 19, 2026
