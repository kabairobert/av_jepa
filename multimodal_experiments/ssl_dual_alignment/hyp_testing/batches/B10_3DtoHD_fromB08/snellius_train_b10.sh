#!/bin/bash
#SBATCH --job-name=b10_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

echo "Starting B10 Volumetric Sweep (72 configs)..."
echo "Running parallel configurations across 9 CPUs on a MIG partition."

# Find all B10 configs and pipe them to xargs.
# xargs keeps exactly 8 background processes running.
ls multimodal_experiments/ssl_dual_alignment/cfgs/B10_[RM]*.yaml | \
    xargs -n 1 -P 8 -I {} bash -c '
        cfg="{}"
        name=$(basename "$cfg" .yaml)
        echo "Launching $name"
        uv run python -m multimodal_experiments.ssl_dual_alignment.main \
            --config "$cfg" \
            --wandb_tags "B10_3DtoHD_fromB08,$name"
    '

echo "B10 Sweep complete."
