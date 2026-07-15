"""Phase 3.4 — AWQ (W4A16) quantization of the merged bf16 model, via llmcompressor.

Runs in the isolated `.venv-quant` environment (NOT the main training venv —
llmcompressor's resolver wants a different torch/transformers pin than the
training stack; see pyproject.toml's `quant` group comment). Calibrates on our
own corpus (data/corpus/train_chat.jsonl) rather than a generic chat dataset,
since that matches the actual input distribution (EPCIS XML / X12 / CSV
messages) this model will see in production.

Loads bf16, not fp16: fp16 caused NaN/inf activations during calibration at
layer 33 (Qwen2.5's activation range exceeds fp16's ~65504 max at that depth)
— confirmed root cause for this exact failure in vllm-project/llm-compressor#2213.
See training/merge.py for the matching bf16 merge.

Usage:
    .venv-quant/Scripts/python -m training.quantize_awq
    .venv-quant/Scripts/python -m training.quantize_awq --merged <dir> --out <dir> --n-calib 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets  # noqa: F401  (import-order guard — keep first, see training/merge.py)

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MERGED = "training/runs/qwen3b_qlora/unsloth/merged_bf16"
DEFAULT_OUT = "training/runs/qwen3b_qlora/unsloth/awq"
TRAIN_CHAT = "data/corpus/train_chat.jsonl"
MAX_SEQ_LEN = 512


def load_calibration_dataset(tokenizer, n: int) -> Dataset:
    """First n samples of our own train corpus, chat-templated + tokenized."""
    records = []
    with open(TRAIN_CHAT, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))

    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in records]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(example):
        return tokenizer(
            example["text"], max_length=MAX_SEQ_LEN, truncation=True, add_special_tokens=False
        )

    return ds.map(tokenize, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", default=DEFAULT_MERGED, help="merged bf16 model dir")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output dir for AWQ model")
    parser.add_argument("--n-calib", type=int, default=256, help="calibration sample count")
    args = parser.parse_args()

    print(f"Loading merged model: {args.merged}")
    tokenizer = AutoTokenizer.from_pretrained(args.merged)
    model = AutoModelForCausalLM.from_pretrained(args.merged, dtype="bfloat16", device_map="cuda:0")

    print(f"Building calibration set ({args.n_calib} samples from {TRAIN_CHAT})")
    calib_ds = load_calibration_dataset(tokenizer, args.n_calib)

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
    ]

    print("Running AWQ oneshot quantization...")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=args.n_calib,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), save_compressed=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved AWQ-quantized model to {out_dir}")


if __name__ == "__main__":
    main()
