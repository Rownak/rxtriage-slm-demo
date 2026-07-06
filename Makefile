.PHONY: setup data train serve eval demo test

setup:
	uv sync

test:
	uv run pytest

data:
	@echo "TODO (Phase 1-2): reference data fetch + synthetic corpus generation"

train:
	@echo "TODO (Phase 3): QLoRA fine-tuning"

serve:
	@echo "TODO (Phase 4): vLLM serving + router"

eval:
	@echo "TODO (Phase 5): evaluation harness"

demo:
	@echo "TODO (Phase 5): end-to-end raw message -> triaged exception demo"
