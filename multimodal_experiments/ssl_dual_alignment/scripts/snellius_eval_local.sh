source ~/.secrets

cd ~/github/eb_jepa_private

export PATH="$HOME/.cargo/bin:$PATH"

# List of run folders to evaluate. Edit to fit your runs.
RUN_FOLDERS=(
#    "/home/rkabai/github/eb_jepa_private/checkpoints/dual_disentangle/dev_2026-05-07_09-47/dalign_3d-av-1f-common_pred_diagonal_loss_ebm_l1_True_sparse_0.1_seed12345"
    "/home/rkabai/github/eb_jepa_private/checkpoints/dual_disentangle/dev_2026-05-07_11-11/dalign_3d-2f-common_pred_diagonal_loss_ebm_l1_True_sparse_0.1_seed12345/"
)

echo "Starting eval for ${#RUN_FOLDERS[@]} runs"

idx=1
for RUN_FOLDER in "${RUN_FOLDERS[@]}"; do
    echo "[${idx}/${#RUN_FOLDERS[@]}] Evaluating: ${RUN_FOLDER}"

    uv run python -m multimodal_experiments.ssl_dual_alignment.eval \
        --folder "${RUN_FOLDER}" \
        --log_wandb true \
        --log_interactive_3d true \
        --interactive_min_height 420
    echo "Completed: ${RUN_FOLDER}"
    idx=$((idx + 1))
done
# Other options:
        # --max_batches 100 \

echo "All evals complete." 
