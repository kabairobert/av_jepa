#!/bin/bash
#SBATCH --job-name=ejv_eval
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=01:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Make CUDA allocator expandable to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

export EBJEPA_DSETS=$TMPDIR/datasets

mkdir -p $TMPDIR/datasets
cp -r ~/JEPA/datasets/* $TMPDIR/datasets/

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

RUN_FOLDER="/gpfs/home3/rkabai/github/eb_jepa_private/checkpoints/video_jepa/dev_2026-03-02_00-11/resnet_bs32_lr0.001_std10.0_cov100.0_seed2025/"

echo "Starting standalone eval on run folder: $RUN_FOLDER"
uv run python -m examples.video_jepa.eval \
    --folder "$RUN_FOLDER" \
    --eval_cfg "/gpfs/home3/rkabai/github/eb_jepa_private/examples/video_jepa/cfgs/eval_overrides_template.yaml"
    
echo "Eval 1 complete."


RUN_FOLDER="/gpfs/home3/rkabai/github/eb_jepa_private/checkpoints/video_jepa/dev_2026-02-28_23-26/resnet_bs24_lr0.001_std10.0_cov100.0_seed2025/"

echo "Starting standalone eval on run folder: $RUN_FOLDER"
uv run python -m examples.video_jepa.eval \
    --folder "$RUN_FOLDER" \
    --eval_cfg "/gpfs/home3/rkabai/github/eb_jepa_private/examples/video_jepa/cfgs/eval_overrides_template.yaml"
    
echo "Eval 2 complete."