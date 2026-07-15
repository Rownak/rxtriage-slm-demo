"""Phase 3.4 acceptance check — does the AWQ-quantized model produce schema-valid
output on 20 smoke samples?

Loads the AWQ (compressed-tensors W4A16) checkpoint directly via plain
transformers — no vLLM needed. transformers auto-detects `quant_method:
"compressed-tensors"` in the checkpoint's config.json (via the installed
`compressed-tensors` package) and swaps quantized Linear layers for
`CompressedLinear`, which decompresses int4 weights and runs the forward pass
in PyTorch. vLLM is the optimized-serving target (Phase 4); this is a
correctness check, not a throughput benchmark.

Must run in `.venv-quant` (needs the `compressed-tensors` package installed
there; the main training venv doesn't have it).

Usage:
    .venv-quant/Scripts/python -m training.smoke_awq               # default 20 samples
    .venv-quant/Scripts/python -m training.smoke_awq --n 20 --model <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# datasets/pyarrow must import before torch (DLL load-order guard — see
# training/eval_val.py and training/train_unsloth.py for the same note).
import datasets  # noqa: F401

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# eval/ lives at the repo root, one level up from training/ — add it to the
# path so this script works when invoked as `.venv-quant/.../python -m training.smoke_awq`
# from the repo root (same working-dir assumption as eval_val.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import aggregate, parse_output, score_sample  # noqa: E402

DEFAULT_MODEL = "training/runs/qwen3b_qlora/unsloth/awq"
VAL_CHAT = "data/corpus/val_chat.jsonl"
VAL_RAW = "data/corpus/val.jsonl"


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 1024) -> str:
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    inputs = tokenizer.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AWQ model dir")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    chat = [json.loads(x) for x in Path(VAL_CHAT).read_text(encoding="utf-8").splitlines()]
    raw = [json.loads(x) for x in Path(VAL_RAW).read_text(encoding="utf-8").splitlines()]
    chat, raw = chat[: args.n], raw[: args.n]

    print(f"Loading AWQ model from {args.model}")
    model, tokenizer = load_model(args.model)

    per_sample = []
    for i, (c, r) in enumerate(zip(chat, raw), 1):
        text = generate(model, tokenizer, c["messages"])
        pred = parse_output(text)
        per_sample.append(score_sample(pred, r["target"], r["raw_message"]))
        print(f"  {i}/{len(chat)}  schema_valid={per_sample[-1]['schema_valid']}")

    agg = aggregate(per_sample)
    print("\n=== AWQ-quantized model — 20-sample smoke test ===")
    print(f"  schema-valid : {agg['schema_valid_rate']:.1%}")
    print(f"  macro-F1     : {agg['exception_macro_f1']:.3f}")
    print(f"  header EM    : {agg['header_exact_match_rate']:.1%}")
    print(f"  hallucination: {agg['hallucination_rate']:.1%}")

    out = Path(args.model) / "smoke_eval.json"
    out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
