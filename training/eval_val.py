"""Phase 3.2 acceptance check — does the QLoRA adapter beat the base model on val?

Loads Qwen2.5-3B-Instruct in 4-bit + the trained LoRA adapter, generates on a
slice of val_chat.jsonl, and scores with the SAME metric core the Phase 3.1
baselines used (eval/metrics.py). Prints schema-validity + macro-F1 so we can
confirm the acceptance bar: val schema-validity > baseline after 1 epoch.

The honest bar is base_zeroshot (same model, no adapter) from
eval/results/baselines.json: 13.1% schema-valid / 0.126 macro-F1.

Usage:
    uv run --group train python -m training.eval_val               # default 100 samples
    uv run --group train python -m training.eval_val --n 200 --adapter <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# datasets/pyarrow must import before torch on native Windows (DLL load-order
# crash otherwise). transformers pulls datasets in; import it first to be safe.
import datasets  # noqa: F401  (import-order guard — keep first)

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval.metrics import aggregate, parse_output, score_sample

DEFAULT_CONFIG = "training/qwen3b_qlora.yaml"
VAL_CHAT = "data/corpus/val_chat.jsonl"
# Raw val (has the oracle `target` + raw_message) — paired to _chat by line order.
VAL_RAW = "data/corpus/val.jsonl"


def load_model(cfg: dict, adapter_dir: str):
    q = cfg["quant"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 1024) -> str:
    # Drop the gold assistant turn; prompt the model to produce it.
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    # return_dict gives us the attention_mask too — pad_token == eos_token here,
    # so without an explicit mask the model can't tell padding from a real EOS.
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
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--adapter", default=None, help="adapter dir (default: <output_dir>/adapter)")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    # Default to the HF run's adapter; pass --adapter <dir> for the Unsloth one.
    adapter_dir = args.adapter or str(Path(cfg["output_dir"]) / "hf" / "adapter")

    chat = [json.loads(x) for x in Path(VAL_CHAT).read_text(encoding="utf-8").splitlines()]
    raw = [json.loads(x) for x in Path(VAL_RAW).read_text(encoding="utf-8").splitlines()]
    chat, raw = chat[: args.n], raw[: args.n]

    model, tokenizer = load_model(cfg, adapter_dir)

    per_sample = []
    for i, (c, r) in enumerate(zip(chat, raw), 1):
        text = generate(model, tokenizer, c["messages"])
        pred = parse_output(text)
        per_sample.append(score_sample(pred, r["target"], r["raw_message"]))
        if i % 20 == 0 or i == len(chat):
            print(f"  {i}/{len(chat)}")

    agg = aggregate(per_sample)
    print("\n=== Fine-tuned (val) vs base_zeroshot baseline ===")
    print(f"  schema-valid : {agg['schema_valid_rate']:.1%}  (baseline 13.1%)")
    print(f"  macro-F1     : {agg['exception_macro_f1']:.3f}  (baseline 0.126)")
    print(f"  header EM    : {agg['header_exact_match_rate']:.1%}  (baseline 44.7%)")
    print(f"  hallucination: {agg['hallucination_rate']:.1%}")

    # Write next to the adapter that was evaluated (e.g. .../hf/ or .../unsloth/),
    # not the shared output_dir — otherwise back-to-back HF and Unsloth evals
    # would overwrite each other's val_eval.json.
    out = Path(adapter_dir).parent / "val_eval.json"
    out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
