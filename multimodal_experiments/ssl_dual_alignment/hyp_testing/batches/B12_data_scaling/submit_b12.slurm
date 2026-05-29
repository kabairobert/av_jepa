#!/bin/bash
#SBATCH --job-name=b12_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Note: set -e intentionally omitted. xargs handles processes in subshells,
# so individual run failures are isolated and captured explicitly in failed_runs.log below.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

FAILED_LOG="multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B12_data_scaling/failed_runs.log"
> "$FAILED_LOG"  # truncate at start

echo "Starting B12 Data Scaling Sweep (9 configs: 1x to 256x)..."
echo "Running parallel configurations across 9 CPUs on a MIG partition."

# Find all B12 configs and pipe them to xargs.
# xargs keeps exactly 9 background processes running concurrently.
# || true: individual failures are isolated and logged; they do NOT kill the sweep.
ls multimodal_experiments/ssl_dual_alignment/cfgs/B12_*.yaml | \
    xargs -n 1 -P 9 -I {} bash -c '
        cfg="{}"
        name=$(basename "$cfg" .yaml)
        echo "Launching $name"
        uv run python -m multimodal_experiments.ssl_dual_alignment.main \
            --config "$cfg" \
            --wandb_tags "B12_data_scaling,$name" \
        || echo "FAILED: $name" >> '"$FAILED_LOG"'
    '

echo "B12 Sweep complete."
if [ -s "$FAILED_LOG" ]; then
    echo "Some runs failed:"
    cat "$FAILED_LOG"
else
    echo "All runs completed successfully."
fi
