#!/bin/bash
#SBATCH --job-name=dalign_eval
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

# List of run folders to evaluate. Edit to fit your runs.
RUN_FOLDERS=(
    "/gpfs/home3/rkabai/github/eb_jepa_private/checkpoints/dual_disentangle/dev_2026-05-07_10-08/dalign_3d-2f-common_pred_diagonal_loss_ebm_l1_True_sparse_0.1_seed12345/"
    # Add more run folders here, one per line
)

echo "Starting eval for ${#RUN_FOLDERS[@]} runs"

idx=1
for RUN_FOLDER in "${RUN_FOLDERS[@]}"; do
    echo "[${idx}/${#RUN_FOLDERS[@]}] Evaluating: ${RUN_FOLDER}"

    uv run python -m multimodal_experiments.ssl_dual_alignment.eval \
        --folder "${RUN_FOLDER}" \
        # --max_batches 100 \
        --log_wandb true \
        --log_interactive_3d true \
        --interactive_min_height 420

    echo "Completed: ${RUN_FOLDER}"
    idx=$((idx + 1))
done

echo "All evals complete." 
