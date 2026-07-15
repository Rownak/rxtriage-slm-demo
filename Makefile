.PHONY: setup data train train-unsloth train-eval merge quantize-awq smoke-awq serve eval baselines demo test

setup:
	uv sync

test:
	uv run pytest

data:
	uv run python -m datagen.assemble
	uv run python -m datagen.format_chat

# Phase 3.2: QLoRA fine-tune Qwen2.5-3B (4-bit, LoRA r=16). Config in
# training/qwen3b_qlora.yaml. Runs the venv Python directly (NOT `uv run`): the
# CUDA torch wheel is installed manually, and `uv run --group train` would
# re-resolve and replace it with a CPU-only build. See training/README notes.
train:
	.venv/Scripts/python -m training.train

# Same QLoRA run via Unsloth's FastLanguageModel (faster/lower-VRAM). Needs
# `uv pip install unsloth --torch-backend=auto` on top of the train group.
train-unsloth:
	.venv/Scripts/python -m training.train_unsloth

# Acceptance check: fine-tuned adapter vs. base_zeroshot baseline on val.
train-eval:
	.venv/Scripts/python -m training.eval_val

# Phase 3.4: merge the (Unsloth) LoRA adapter into fp16 safetensors.
merge:
	.venv/Scripts/python -m training.merge

# AWQ (W4A16) export via llmcompressor, in the isolated .venv-quant env — see
# pyproject.toml's `quant` group comment for setup (separate venv because
# llmcompressor's resolver wants a different torch/transformers pin than
# the training stack).
quantize-awq:
	.venv-quant/Scripts/python -m training.quantize_awq

# Acceptance check: 20-sample schema-validity smoke test on the AWQ model.
smoke-awq:
	.venv-quant/Scripts/python -m training.smoke_awq

serve:
	@echo "TODO (Phase 4): vLLM serving + router"

# Phase 3.1: zero/few-shot Qwen2.5-3B (Ollama) + gpt-4o-mini baselines.
# Requires `ollama pull qwen2.5:3b-instruct` and OPENAI_API_KEY in .env.
eval baselines:
	uv run python -m eval.run_baselines

demo:
	@echo "TODO (Phase 5): end-to-end raw message -> triaged exception demo"
