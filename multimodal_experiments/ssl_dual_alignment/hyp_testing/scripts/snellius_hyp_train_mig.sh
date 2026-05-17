#!/bin/bash
#SBATCH --job-name=dual_exp
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

# export EBJEPA_DSETS=$TMPDIR/datasets

# mkdir -p $TMPDIR/datasets
# cp -r ~/JEPA/datasets/* $TMPDIR/datasets/

cd ~/github/eb_jepa_private/multimodal_experiments/ssl_dual_alignment/hyp_testing

export PATH="$HOME/.cargo/bin:$PATH"

echo "Starting training B06_prior_vs_predictor_noise"
uv run sweep.py --batch B06_prior_vs_predictor_noise 

echo "Training complete."
