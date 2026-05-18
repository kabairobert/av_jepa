#!/bin/bash
#SBATCH --job-name=b07_todo
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

echo "Starting B07 TODO Sweep (13 configs)..."

# List of TODO configs
TODO_CFGS=(
    "B07_NPP301"
    "B07_NPP401"
    "B07_NPP500"
    "B07_NPP501"
    "B07_NPP502"
    "B07_NPP510"
    "B07_NPP511"
    "B07_NPP512"
    "B07_NPP520"
    "B07_NPP521"
    "B07_NPP522"
    "B07_NPP601"
    "B07_NPP621"
)

# Run in parallel (P=8)
printf "%s\n" "${TODO_CFGS[@]}" | xargs -I {} -P 8 bash -c '
    cfg_name="{}"
    cfg="multimodal_experiments/ssl_dual_alignment/cfgs/${cfg_name}.yaml"
    echo "Launching $cfg_name"
    uv run python -m multimodal_experiments.ssl_dual_alignment.main \
        --config "$cfg" \
        --wandb_tags "B07_loss_noise_interaction,$cfg_name"
'

echo "B07 TODO Sweep complete."
