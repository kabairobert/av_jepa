#!/bin/bash
#SBATCH --job-name=b13_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 6 configs run in parallel (one per CPU).
# set -e intentionally omitted: xargs subshell failures are logged, not fatal.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

FAILED_LOG="multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B13_affine_mlp/failed_runs.log"
> "$FAILED_LOG"  # truncate at start

echo "Starting B13 Affine MLP Sweep (6 configs: 3 scales × 2 depths)..."

ls multimodal_experiments/ssl_dual_alignment/cfgs/B13_affine_mlp/B13_*.yaml | \
    xargs -n 1 -P 6 -I {} bash -c '
        cfg="{}"
        name=$(basename "$cfg" .yaml)
        echo "Launching $name"
        uv run python -m multimodal_experiments.ssl_dual_alignment.main \
            --config "$cfg" \
            --wandb_tags "B13_affine_mlp,$name" \
        || echo "FAILED: $name" >> '"$FAILED_LOG"'
    '

echo "B13 Sweep complete."
if [ -s "$FAILED_LOG" ]; then
    echo "Some runs failed:"
    cat "$FAILED_LOG"
else
    echo "All runs completed successfully."
fi
