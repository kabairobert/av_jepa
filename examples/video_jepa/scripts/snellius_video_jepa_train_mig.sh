#!/bin/bash
#SBATCH --job-name=ejv
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e  # Exit immediately on any error

# Load required modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

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

# Run training
echo "Starting training..."
uv run python -m examples.video_jepa.main \
    --fname examples/video_jepa/cfgs/sigreg-mig.yaml

echo "Training complete."
