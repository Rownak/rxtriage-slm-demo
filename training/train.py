"""Phase 3.2 — QLoRA fine-tuning of Qwen2.5-3B-Instruct on a single GPU.

Plain Hugging Face stack (transformers + peft + trl + bitsandbytes) rather than
Unsloth: Unsloth requires Triton, which has no official Windows wheels, and this
box is native Windows. The deliverable is identical — 4-bit base, LoRA r=16,
completion-only loss on the JSON output, best-val checkpoint, CSV logging.

All hyperparameters live in training/qwen3b_qlora.yaml (CLAUDE.md Conventions).

Usage:
    uv run --group train python -m training.train                 # full 1-epoch run
    uv run --group train python -m training.train --max-steps 5   # smoke test
    uv run --group train python -m training.train --config other.yaml
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

# IMPORTANT: import datasets (pyarrow-backed) BEFORE torch. On this native-Windows
# box, importing torch first and datasets second segfaults the process (a DLL load
# order conflict between torch's bundled libs and pyarrow). datasets-first is safe.
import datasets  # noqa: F401  (import-order guard — keep first)
from datasets import load_dataset

import torch
import yaml
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from training.run_meta import collect_run_meta, write_run_meta

DEFAULT_CONFIG = "training/qwen3b_qlora.yaml"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CSVLoggerCallback(TrainerCallback):
    """Append every Trainer log row to metrics.csv — simplest 'CSV or wandb' option.

    Trainer emits separate dicts for train logs (loss, lr) and eval logs
    (eval_loss); we write whatever keys are present so both land in one file.
    """

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


# Qwen2.5's stock chat template lumps assistant, user, and system content into
# one emit line and has no {% generation %} tags, so TRL's assistant_only_loss
# can't build the assistant mask. This patch pulls the plain-assistant case (our
# data has no tool calls) into its own branch and wraps its content in
# {% generation %}…{% endgeneration %}, which is exactly what TRL keys on to mask
# loss to the assistant turn. Everything else (system/user emission, the tool-call
# branch, add_generation_prompt) is left byte-for-byte unchanged.
_STOCK_ASSISTANT_CLAUSE = (
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) '
    'or (message.role == "assistant" and not message.tool_calls) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
)
_PATCHED_ASSISTANT_CLAUSE = (
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
    '    {%- elif message.role == "assistant" and not message.tool_calls %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}{{- message.content }}{% endgeneration %}"
    "{{- '<|im_end|>' + '\\n' }}"
)


def patch_template_for_assistant_mask(tokenizer):
    """Add {% generation %} tags to Qwen's chat template so assistant_only_loss works."""
    tpl = tokenizer.chat_template
    if "{% generation %}" in tpl:
        return  # already patched / template already supports masking
    if _STOCK_ASSISTANT_CLAUSE not in tpl:
        raise RuntimeError(
            "Qwen chat template shape changed — can't safely inject generation tags. "
            "Re-inspect the template and update patch_template_for_assistant_mask()."
        )
    tokenizer.chat_template = tpl.replace(_STOCK_ASSISTANT_CLAUSE, _PATCHED_ASSISTANT_CLAUSE)


def build_model_and_tokenizer(cfg: dict):
    q = cfg["quant"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    patch_template_for_assistant_mask(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"]
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer


def make_sft_config(cfg: dict, max_steps: int | None) -> SFTConfig:
    t = cfg["train"]
    # SFTConfig subclasses TrainingArguments. TRL applies the tokenizer chat
    # template to the `messages` column automatically. assistant_only_loss masks
    # the system+user prompt so loss is computed only on the assistant's JSON
    # output — the model is graded on what it must generate, not on copying the
    # prompt back.
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t["optim"],
        max_length=cfg["max_seq_len"],
        packing=False,
        assistant_only_loss=True,
        report_to="none",
        seed=cfg["seed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Backend-specific subdir so the HF and Unsloth runs don't overwrite each
    # other's adapter/metrics/run_meta (we compare the two).
    out_dir = Path(cfg["output_dir"]) / "hf"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()} | "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    dataset = load_dataset(
        "json",
        data_files={"train": cfg["train_file"], "validation": cfg["val_file"]},
    )

    model, tokenizer = build_model_and_tokenizer(cfg)

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = make_sft_config(cfg, args.max_steps)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
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
        backend="hf",
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
