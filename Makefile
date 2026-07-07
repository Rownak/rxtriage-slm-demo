.PHONY: setup data train serve eval baselines demo test

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

# Phase 3.1: zero/few-shot Qwen2.5-3B (Ollama) + gpt-4o-mini baselines.
# Requires `ollama pull qwen2.5:3b-instruct` and OPENAI_API_KEY in .env.
eval baselines:
	uv run python -m eval.run_baselines

demo:
	@echo "TODO (Phase 5): end-to-end raw message -> triaged exception demo"
