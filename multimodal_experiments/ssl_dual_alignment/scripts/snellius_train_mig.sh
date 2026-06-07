#!/bin/bash
#SBATCH --job-name=dual_exp
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

# export EBJEPA_DSETS=$TMPDIR/datasets

# mkdir -p $TMPDIR/datasets
# cp -r ~/JEPA/datasets/* $TMPDIR/datasets/

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

CFG_MATRIX=(
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_diag_l1_wdense_3D2f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_diag_l1_wsparse_3D2f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_diag_l2_wdense_3D2f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_diag_l2_wsparse_3D2f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_mlp_l1_wdense_3D2f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_mlp_l1_wsparse_3D2f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_mlp_l2_wdense_3D2f.yaml"
    "multimodal_experiments/ssl_dual_alignment/cfgs/ebm_pred/ebm_pred_mlp_l2_wsparse_3D2f.yaml"
)
total=${#CFG_MATRIX[@]}

idx=1
for cfg in "${CFG_MATRIX[@]}"; do
    echo "Starting training ${idx}/${total}: ${cfg}"
    uv run python -m multimodal_experiments.ssl_dual_alignment.main --fname "${cfg}" --logging.notes "SSL_align_noise|data:3D2f-rot|pred:mlp/diag|l1:true/false|wsprase:0/0.1"
    idx=$((idx + 1))
done

echo "Training complete."
