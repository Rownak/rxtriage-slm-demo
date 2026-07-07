.PHONY: setup data train serve eval demo test

setup:
	uv sync

test:
	uv run pytest

data:
	uv run python -m datagen.assemble
	uv run python -m datagen.format_chat

train:
	@echo "TODO (Phase 3): QLoRA fine-tuning"

serve:
	@echo "TODO (Phase 4): vLLM serving + router"

eval:
	@echo "TODO (Phase 5): evaluation harness"

demo:
	@echo "TODO (Phase 5): end-to-end raw message -> triaged exception demo"
