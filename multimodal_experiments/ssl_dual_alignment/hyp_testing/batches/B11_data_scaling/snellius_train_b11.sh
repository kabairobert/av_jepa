#!/bin/bash
#SBATCH --job-name=b11_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=02:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Note: set -e intentionally omitted. It does not propagate into xargs subshells,
# so individual run failures would not terminate the sweep anyway. Run failures are
# instead captured explicitly in failed_runs.log below.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

FAILED_LOG="multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B11_data_scaling/failed_runs.log"
> "$FAILED_LOG"  # truncate at start

echo "Starting B11 Data Scaling Sweep (24 configs)..."
echo "Running parallel configurations across 9 CPUs on a MIG partition."

# Find all B11 configs and pipe them to xargs.
# xargs keeps exactly 8 background processes running.
# || true: individual failures are isolated and logged; they do NOT kill the sweep.
ls multimodal_experiments/ssl_dual_alignment/cfgs/B11_*.yaml | \
    xargs -n 1 -P 8 -I {} bash -c '
        cfg="{}"
        name=$(basename "$cfg" .yaml)
        echo "Launching $name"
        uv run python -m multimodal_experiments.ssl_dual_alignment.main \
            --config "$cfg" \
            --wandb_tags "B11_data_scaling,$name" \
        || echo "FAILED: $name" >> '"$FAILED_LOG"'
    '

echo "B11 Sweep complete."
if [ -s "$FAILED_LOG" ]; then
    echo "Some runs failed:"
    cat "$FAILED_LOG"
else
    echo "All runs completed successfully."
fi
