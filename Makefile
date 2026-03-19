.ONESHELL:
.PHONY: help
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

run_image_jepa: ## Run the image JEPA example
	uv run python examples/image_jepa/main.py

run_video_jepa: ## Run the video JEPA example
	uv run python examples/video_jepa/main.py

run_ac_video_jepa: ## Run the action-conditioned video JEPA example
	uv run python examples/ac_video_jepa/main.py

# ===== Tools: W&B & Diagnostics =====

generate_run_names: ## Generate experiment names from config files
	uv run python tools/print_run_names.py

inspect_wandb: ## Inspect a W&B run (specify with RUN_ID, ENTITY, PROJECT env vars or --run flag)
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Usage: make inspect_wandb RUN_ID=q1wxqpk2 ENTITY=entity PROJECT=project"; \
		echo "Or: make inspect_wandb -- --run entity/project/run_id"; \
		uv run python tools/wandb_inspect_run.py --help; \
	else \
		uv run python tools/wandb_inspect_run.py --run $(RUN_ID) --entity $(ENTITY) --project $(PROJECT); \
	fi

cleanup_wandb_dryrun: ## Dry-run W&B cleanup (show what would be deleted)
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Usage: make cleanup_wandb_dryrun RUN_ID=entity/project/run_id"; \
		uv run python tools/wandb_cleanup.py --help; \
	else \
		uv run python tools/wandb_cleanup.py --run $(RUN_ID); \
	fi

resolve_diagnostics: ## Query diagnostics from a local run folder
	@if [ -z "$(RUN_DIR)" ]; then \
		echo "Usage: make resolve_diagnostics RUN_DIR=/path/to/run [LIST_EVENTS=1] [METRIC=key]"; \
		uv run python tools/resolve_diagnostics.py --help; \
	else \
		if [ "$(LIST_EVENTS)" = "1" ]; then \
			uv run python tools/resolve_diagnostics.py $(RUN_DIR) --list-events --compact; \
		elif [ -n "$(METRIC)" ]; then \
			uv run python tools/resolve_diagnostics.py $(RUN_DIR) --metric $(METRIC); \
		else \
			uv run python tools/resolve_diagnostics.py $(RUN_DIR); \
		fi \
	fi
