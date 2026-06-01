#!/bin/bash
#SBATCH --job-name=b14_sweep
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B14_scaling_up_z_dim/logs/b14_%j.out
#SBATCH --error=multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B14_scaling_up_z_dim/logs/b14_%j.err

# 5 configs run in parallel (one per CPU).
# set -e intentionally omitted: xargs subshell failures are logged, not fatal.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

FAILED_LOG="multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B14_scaling_up_z_dim/logs/failed_runs.log"
mkdir -p multimodal_experiments/ssl_dual_alignment/hyp_testing/batches/B14_scaling_up_z_dim/logs
> "$FAILED_LOG"  # truncate at start

echo "Starting B14 Scaling Sweep (5 configs)..."

run_job() {
    cfg="$1"
    name=$(basename "$cfg" .yaml)
    echo "Launching $name"
    uv run python -m multimodal_experiments.ssl_dual_alignment.main \
        --config "$cfg" \
        --wandb_tags "B14_scaling_up_z_dim,$name" \
    || echo "FAILED: $name" >> "$FAILED_LOG"
}

echo "Running Group 1 (High + Low VRAM): 512D, 64D, 32D"
run_job "multimodal_experiments/ssl_dual_alignment/cfgs/B14_512D_S12.yaml" &
#run_job "multimodal_experiments/ssl_dual_alignment/cfgs/B14_64D_S6.yaml" &
#run_job "multimodal_experiments/ssl_dual_alignment/cfgs/B14_32D_S4.yaml" &
wait

#echo "Running Group 2 (Medium VRAM): 256D, 128D"
#run_job "multimodal_experiments/ssl_dual_alignment/cfgs/B14_256D_S10.yaml" &
#run_job "multimodal_experiments/ssl_dual_alignment/cfgs/B14_128D_S8.yaml" &
#wait

echo "B14 Sweep complete."
if [ -s "$FAILED_LOG" ]; then
    echo "Some runs failed:"
    cat "$FAILED_LOG"
else
    echo "All runs completed successfully."
fi
