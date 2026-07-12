"""Phase 3.2 — QLoRA fine-tuning of Qwen2.5-3B-Instruct on a single GPU, via Unsloth.

Unsloth was originally skipped (see training/train.py) because it requires
Triton, which has no official Windows wheels. `triton-windows` closes that gap,
so this is the Unsloth path: FastLanguageModel loads the 4-bit base + attaches
LoRA in one patched call, giving faster/lower-memory training than the plain
transformers+peft+trl stack in training/train.py. Both scripts read the same
YAML config (training/qwen3b_qlora.yaml) so hyperparameters stay identical —
only the model-loading and PEFT-attach mechanics differ.

Usage:
    uv run python -m training.train_unsloth                 # full 1-epoch run
    uv run python -m training.train_unsloth --max-steps 5    # smoke test
    uv run python -m training.train_unsloth --config other.yaml
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

# IMPORTANT: import datasets (pyarrow-backed) BEFORE torch/unsloth. On this
# native-Windows box, importing torch first and datasets second segfaults the
# process (a DLL load order conflict between torch's bundled libs and
# pyarrow). datasets-first is safe. See the same guard in training/train.py.
import datasets  # noqa: F401  (import-order guard — keep first)
from datasets import load_dataset

import torch
import yaml
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from training.run_meta import collect_run_meta, write_run_meta

DEFAULT_CONFIG = "training/qwen3b_qlora.yaml"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CSVLoggerCallback(TrainerCallback):
    """Append every Trainer log row to metrics.csv — same format as training/train.py
    so the two runs are directly comparable."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.fields = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.fields)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0, 3)}
        for k in ("loss", "eval_loss", "learning_rate", "grad_norm"):
            if k in logs:
                row[k] = logs[k]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])


def build_model_and_tokenizer(cfg: dict):
    # FastLanguageModel folds 4-bit quant config + model load into one call —
    # Unsloth always uses NF4 double-quant, matching training/train.py's
    # bitsandbytes settings, so cfg["quant"] isn't threaded through here.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_len"],
        dtype=None,  # auto-detect (bfloat16 on Ampere+)
        load_in_4bit=cfg["quant"]["load_in_4bit"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def make_formatting_func(tokenizer):
    # Unsloth's SFTTrainer wrapper requires an explicit formatting_func for
    # chat-format data — unlike plain TRL (training/train.py), it does not
    # auto-apply the tokenizer's chat template to a `messages` column. It's
    # called two ways: once with a single example (a probe that just needs a
    # list back), then via dataset.map(batched=True) with a dict-of-lists
    # (one list entry per example in the batch) — handle both shapes.
    def formatting_func(batch):
        conversations = batch["messages"]
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]  # single-example probe call
        return [tokenizer.apply_chat_template(m, tokenize=False) for m in conversations]
    return formatting_func


def make_sft_config(cfg: dict, max_steps: int | None) -> SFTConfig:
    t = cfg["train"]
    # Same trade-off as training/train.py: full-sequence loss, no completion-only
    # masking (Simplicity Rule) — see that file's comment for the full rationale.
    return SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=t["epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=t["per_device_batch"],
        per_device_eval_batch_size=t["per_device_batch"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        max_seq_length=cfg["max_seq_len"],
        packing=False,
        report_to="none",
        seed=cfg["seed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Backend-specific subdir so HF and Unsloth runs don't overwrite each other.
    out_dir = Path(cfg["output_dir"]) / "unsloth"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()} | "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    dataset = load_dataset(
        "json",
        data_files={"train": cfg["train_file"], "validation": cfg["val_file"]},
    )

    model, tokenizer = build_model_and_tokenizer(cfg)

    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's own checkpointing (lower VRAM than HF's)
        random_state=cfg["seed"],
    )

    sft_config = make_sft_config(cfg, args.max_steps)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        formatting_func=make_formatting_func(tokenizer),
        callbacks=[CSVLoggerCallback(out_dir / "metrics.csv")],
    )

    t0 = time.time()
    train_output = trainer.train()
    wall_seconds = time.time() - t0

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved best-val LoRA adapter to {adapter_dir}")
    print(f"Metrics log: {out_dir / 'metrics.csv'}")

    # Record timing/data/schedule for the HF-vs-Unsloth comparison.
    meta = collect_run_meta(
        backend="unsloth",
        cfg=cfg,
        n_train=len(dataset["train"]),
        n_val=len(dataset["validation"]),
        train_metrics=train_output.metrics,
        max_steps=args.max_steps,
        wall_seconds=wall_seconds,
    )
    write_run_meta(out_dir, meta)


if __name__ == "__main__":
    main()
