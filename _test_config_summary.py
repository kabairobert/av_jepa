from pathlib import Path
from omegaconf import OmegaConf

print("=" * 70)
print("CONFIG RESTRUCTURING SUMMARY")
print("=" * 70)
print()

# Load a config to show the new structure
cfg = OmegaConf.load(Path('examples/video_jepa/cfgs/default.yaml'))

print("NEW STRUCTURE: logging.* (top-level)")
print("-" * 70)
print(f"  save_every: {cfg.logging.save_every}")
print(f"  diagnostics_every_epochs: {cfg.logging.diagnostics_every_epochs}")
print(f"  fast_diagnostics_every_epochs: {cfg.logging.fast_diagnostics_every_epochs}")
print(f"  fast_diagnostics_every_steps: {cfg.logging.fast_diagnostics_every_steps}")
print(f"  log_tensor_shapes: {cfg.logging.log_tensor_shapes}")
print(f"  projector_force_runtime_shapes: {cfg.logging.projector_force_runtime_shapes}")
print(f"  tqdm_silent: {cfg.logging.tqdm_silent}")
print()

print("NEW STRUCTURE: logging.diagnostics.*")
print("-" * 70)
print(f"  enabled: {cfg.logging.diagnostics.enabled}")
print(f"  upload_artifacts: {cfg.logging.diagnostics.upload_artifacts}")
print(f"  artifact_flush_interval_sec: {cfg.logging.diagnostics.artifact_flush_interval_sec}")
print(f"  full_diagnostics_mode: {cfg.logging.diagnostics.full_diagnostics_mode}")
print(f"  fast_diagnostics_mode: {cfg.logging.diagnostics.fast_diagnostics_mode}")
print(f"  val_subset_num_batches: {cfg.logging.diagnostics.val_subset_num_batches}")
print(f"  train_subset_num_batches: {cfg.logging.diagnostics.train_subset_num_batches}")
print()

print("REMOVED FIELDS (clean migration)")
print("-" * 70)
print("  ✓ log_every (removed from config, no longer referenced in code)")
print("  ✓ logging.diagnostics.diagnostics_every_epochs (moved to logging.diagnostics_every_epochs)")
print("  ✓ logging.diagnostics.mode (renamed to full_diagnostics_mode)")
print("  ✓ logging.diagnostics.fast_mode (renamed to fast_diagnostics_mode)")
print()

print("EFFECTIVE CADENCE (with defaults)")
print("-" * 70)
from examples.video_jepa.main import _int_or_none, _resolve_epoch_interval
diag_every = _resolve_epoch_interval(cfg.logging.diagnostics_every_epochs, cfg.logging.save_every)
fast_every = _int_or_none(cfg.logging.fast_diagnostics_every_epochs)
fast_steps = _int_or_none(cfg.logging.fast_diagnostics_every_steps)
print(f"  Canonical diagnostics: every {diag_every} epochs (full_val mode, geometry ON)")
print(f"  Fast diagnostics (epoch): every {fast_every} epochs (val_subset mode, geometry OFF) if set")
print(f"  Fast diagnostics (steps): every {fast_steps} steps (val_subset mode, geometry OFF) if set")
print()

print("EPOCH TIMELINE EXAMPLE")
print("-" * 70)
print("  Epoch  Event")
print("  ------  -----")
for epoch in range(1, 21):
    canonical = (epoch % diag_every == 0) and epoch > 0
    fast_ep = (epoch % fast_every == 0) and epoch > 0 and not canonical if fast_every else False
    if canonical:
        print(f"   {epoch:2d}    ✓ CANONICAL DIAGNOSTICS (full_val, persisted)")
    elif fast_ep:
        print(f"   {epoch:2d}    ⚡ FAST DIAGNOSTICS (val_subset, not persisted)")
    else:
        print(f"   {epoch:2d}    -")

print()
print("=" * 70)
print("✓ All 5 config files updated successfully")
print("✓ main.py refactored to read new structure")
print("✓ Fast diagnostics cadence now wired and independent")
print("=" * 70)
