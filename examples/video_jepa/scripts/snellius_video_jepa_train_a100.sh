#!/bin/bash
#SBATCH --job-name=ejv
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e  # Exit immediately on any error

# Make CUDA allocator expandable to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load required modules
# module load 2023
# module load Python/3.11.3-GCCcore-12.3.0
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Source secrets (API keys etc.)
source ~/.secrets

# Set dataset environment variable to TMPDIR (fast local storage)
export EBJEPA_DSETS=$TMPDIR/datasets

# Copy dataset to TMPDIR (fast local node storage)
echo "Copying datasets to TMPDIR..."
mkdir -p $TMPDIR/datasets
cp -r ~/JEPA/datasets/* $TMPDIR/datasets/
echo "Dataset copy complete."

# Navigate to repo
cd ~/github/eb_jepa_private

# Add uv to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Example smoke-test override if you want a sub-epoch run:
# uv run python -m examples.video_jepa.main --fname "examples/video_jepa/cfgs/sigreg_linear_encoder.yaml" --training.max_train_batches=3 --optim.epochs=10 --logging.log_wandb=false --logging.diagnostics.enabled=false

# 6-way predictor comparison matrix (architecture x location):
#   ResUNet/MLP/Linear x Encoder/Projector
CFG_MATRIX=(
    # "examples/video_jepa/cfgs/sigreg.yaml"
    "examples/video_jepa/cfgs/sigreg_resunet_projector.yaml"
    # "examples/video_jepa/cfgs/sigreg_mlp_encoder.yaml"
    # "examples/video_jepa/cfgs/sigreg_mlp_projector.yaml"
    # "examples/video_jepa/cfgs/sigreg_linear_encoder.yaml"
    # "examples/video_jepa/cfgs/sigreg_linear_projector.yaml"
)
total=${#CFG_MATRIX[@]}

idx=1
for cfg in "${CFG_MATRIX[@]}"; do
    echo "Starting training ${idx}/${total}: ${cfg}"
    uv run python -m examples.video_jepa.main --fname "${cfg}" --training.max_train_batches=2 --optim.epochs=10 --logging.log_wandb=False
    idx=$((idx + 1))
done

echo "Training complete."
