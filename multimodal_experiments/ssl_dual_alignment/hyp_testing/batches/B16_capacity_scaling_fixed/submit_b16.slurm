#!/bin/bash
#SBATCH --job-name=b16_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=3:30:00
#SBATCH --array=0-5
#SBATCH --output=/gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B16_capacity_scaling_fixed/logs/%A_%a.out
#SBATCH --error=/gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B16_capacity_scaling_fixed/logs/b16_%A_%a.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

mkdir -p multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B16_capacity_scaling_fixed/logs

# List of B16 configurations in order of SLURM_ARRAY_TASK_ID
CONFIGS=(
  "B16_16D_S6"
  "B16_32D_S8"
  "B16_64D_S10"
  "B16_128D_S12"
  "B16_256D_S14"
  "B16_512D_S16"
)

CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CFG_PATH="multimodal_experiments/ssl_dual_alignment/cfgs/B16_capacity_scaling_fixed/${CONFIG_NAME}.yaml"

echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "Config Name: $CONFIG_NAME"
echo "Config Path: $CFG_PATH"

echo "Launching $CONFIG_NAME..."
uv run python -m multimodal_experiments.ssl_dual_alignment.main \
    --config "$CFG_PATH" \
    --wandb_tags "B16_capacity_scaling_fixed,$CONFIG_NAME"

echo "Run finished for $CONFIG_NAME."
