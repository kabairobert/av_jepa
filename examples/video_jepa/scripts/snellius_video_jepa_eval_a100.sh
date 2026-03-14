#!/bin/bash
#SBATCH --job-name=ejv_eval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
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

RUN_FOLDER="/gpfs/home3/rkabai/github/eb_jepa_private/checkpoints/video_jepa/dev_2026-03-09_09-33/resnet_bs84_bcs-euler-scalefree_proj64x64_ns32_lmbd10_lr1e-03_seed2025/"

echo "Starting standalone eval on run folder: $RUN_FOLDER"
uv run python -m examples.video_jepa.eval \
    --folder "$RUN_FOLDER" \
    --eval_cfg "/gpfs/home3/rkabai/github/eb_jepa_private/examples/video_jepa/cfgs/eval_overrides_template.yaml"

echo "Eval complete."
