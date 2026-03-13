#!/bin/bash
#SBATCH --job-name=ejv_eval
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

export EBJEPA_DSETS=$TMPDIR/datasets

mkdir -p $TMPDIR/datasets
cp -r ~/JEPA/datasets/* $TMPDIR/datasets/

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

RUN_FOLDER=~/JEPA/checkpoints/video_jepa/REPLACE_ME

echo "Starting standalone eval on run folder: $RUN_FOLDER"
uv run python -m examples.video_jepa.eval \
    --folder "$RUN_FOLDER"

echo "Eval complete."
