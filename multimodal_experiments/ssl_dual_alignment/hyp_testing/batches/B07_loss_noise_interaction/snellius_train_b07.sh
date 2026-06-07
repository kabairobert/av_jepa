#!/bin/bash
#SBATCH --job-name=b07_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=02:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

echo "Starting B07 Parallel Sweep..."
echo "Running 8 parallel configurations across 9 CPUs on a MIG partition."

# Find all B07 configs and pipe them to xargs.
# xargs keeps exactly 8 background processes running.
ls multimodal_experiments/ssl_dual_alignment/cfgs/B07_loss_noise_interaction/B07_NPP*.yaml | \
    xargs -n 1 -P 8 -I {} bash -c '
        cfg="{}"
        name=$(basename "$cfg" .yaml)
        echo "Launching $name"
        uv run python -m multimodal_experiments.ssl_dual_alignment.main \
            --config "$cfg" \
            --wandb_tags "B07_loss_noise_interaction,$name"
    '

echo "B07 Sweep complete."
