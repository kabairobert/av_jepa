---
name: tools-instructions
description: "Use when: working with tools/* scripts, debugging W&B utilities, querying diagnostics, or extending CLI functionality"
applyTo:
  - "tools/**"
---

# Tools Folder Instructions

## Tool Reference

| Script | Purpose | Key Flags |
|--------|---------|-----------|
| `wandb_inspect_run.py` | W&B run inspect | `--run`, `--entity`, `--project`, `--scan-prefix`, `--json` |
| `wandb_cleanup.py` | W&B cleanup | `--run`, `--no-dry-run`, `--delete-files`, `--yes` |
| `resolve_diagnostics.py` | Local diagnostics query | `--list-events`, `--list-metrics`, `--metric`, `--event`, `--compact` |
| `print_run_names.py` | Config name generation | none |
| `inspect_wandb_run.py` | Deprecated wrapper | delegates to `wandb_inspect_run.py` |

## Commands

```bash
# W&B run (full path preferred)
python tools/wandb_inspect_run.py --run entity/project/run_id

# W&B run (short id)
python tools/wandb_inspect_run.py --run q1wxqpk2 --entity entity --project project

# Local diagnostics
python tools/resolve_diagnostics.py /path/to/run --list-events --compact
python tools/resolve_diagnostics.py /path/to/run --metric loss/loss_jepa
python tools/resolve_diagnostics.py /path/to/run --event event_001

# Cleanup (dry-run default)
python tools/wandb_cleanup.py --run entity/project/run_id
```

## Rules

- New W&B tools must use `wandb_` prefix
- `--run` should accept short id + entity/project and full path
- Keep diagnostics queries read-only
- Prefer explicit flag names (`--list-*`, `--find-*`, `--filter-*`)
- Keep `inspect_wandb_run.py` only as compatibility wrapper

Updated: March 19, 2026
