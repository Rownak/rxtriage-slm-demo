"""Phase 3.4 — merge the LoRA adapter into the base model as bf16 safetensors.

Loads the ORIGINAL Qwen2.5-3B-Instruct (bf16 — not the Unsloth 4-bit mirror
the adapter was trained against, and not fp16) and merges the LoRA weights on
top via plain peft. LoRA A/B matrices are base-quantization-agnostic — they
only need the base layer shapes to match — and merging into a full-precision
base avoids the rounding/OOM issues documented for merging back into a
4-bit-loaded base. This is the standard QLoRA -> merge pattern (both PEFT's
and Unsloth's own docs recommend it over a 4-bit merge).

bf16, not fp16: Qwen2.5 is trained/released in bf16, and its activations at
deeper layers can exceed fp16's much smaller dynamic range (~65504 max vs
bf16's ~3.4e38). An fp16 merge caused NaN/inf activations during AWQ
calibration (training/quantize_awq.py) at layer 33 — a known llmcompressor
failure mode for Qwen models loaded in fp16 (confirmed root cause in
vllm-project/llm-compressor#2213). Training itself already used bf16 compute
(qwen3b_qlora.yaml's bnb_4bit_compute_dtype), so this also matches training
precision.

Output feeds training/quantize_awq.py (Phase 3.4's AWQ export step).

Usage:
    uv run python -m training.merge                            # default: unsloth adapter
    uv run python -m training.merge --adapter <dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

# datasets/pyarrow must import before torch (DLL load-order guard — see
# training/eval_val.py and training/train_unsloth.py for the same note).
import datasets  # noqa: F401

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = "training/runs/qwen3b_qlora/unsloth/adapter"
DEFAULT_OUT = "training/runs/qwen3b_qlora/unsloth/merged_bf16"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter dir")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output dir for merged bf16 model")
    args = parser.parse_args()

    print(f"Loading base model (bf16): {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging LoRA weights into base...")
    model = model.merge_and_unload()

    n_params = sum(p.numel() for p in model.parameters())
    dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"Merged model: {n_params / 1e9:.2f}B params, dtypes={dtypes}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved merged bf16 model to {out_dir}")


if __name__ == "__main__":
    main()
