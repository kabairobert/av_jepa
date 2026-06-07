#!/bin/bash
#SBATCH --job-name=b15_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=/gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B15_capacity_scaling/logs/b15_%A_%a.out
#SBATCH --error=/gpfs/home3/rkabai/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B15_capacity_scaling/logs/b15_%A_%a.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

mkdir -p multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B15_capacity_scaling/logs

# List of B15 configurations in order of SLURM_ARRAY_TASK_ID
CONFIGS=(
  "B15_32D_S8"
  "B15_64D_S10"
  "B15_128D_S12"
  "B15_256D_S14"
  "B15_512D_S16"
)

CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CFG_PATH="multimodal_experiments/ssl_dual_alignment/cfgs/B15_capacity_scaling/${CONFIG_NAME}.yaml"

echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "Config Name: $CONFIG_NAME"
echo "Config Path: $CFG_PATH"

echo "Launching $CONFIG_NAME..."
uv run python -m multimodal_experiments.ssl_dual_alignment.main \
    --config "$CFG_PATH" \
    --wandb_tags "B15_capacity_scaling,$CONFIG_NAME"

echo "Run finished for $CONFIG_NAME."
