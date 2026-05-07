#!/bin/bash
#SBATCH --job-name=dual_exp
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

# export EBJEPA_DSETS=$TMPDIR/datasets

# mkdir -p $TMPDIR/datasets
# cp -r ~/JEPA/datasets/* $TMPDIR/datasets/

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

CFG_MATRIX=(
    "multimodal_experiments/ssl_dual_alignment/cfgs/paired_factors_2D.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred_mlp_l2_wnosparse_2D.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred_diag_l1_wsparse_2D.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred_diag_l1_wsparse_3D1f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred_diag_l1_wsparse_3D2f.yaml"
)
total=${#CFG_MATRIX[@]}

idx=1
for cfg in "${CFG_MATRIX[@]}"; do
    echo "Starting training ${idx}/${total}: ${cfg}"
    uv run python -m multimodal_experiments.ssl_dual_alignment.main --fname "${cfg}" --logging.notes "SSL dual alignment w/noise, 5 model vers"
    idx=$((idx + 1))
done

echo "Training complete."
