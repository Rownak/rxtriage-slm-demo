"""Shared training-run metadata capture — for the HF-vs-Unsloth timing comparison.

Both training/train.py and training/train_unsloth.py call `write_run_meta()` after
training to record duration, data size, epochs/steps, and environment into
run_meta.json in the run's output dir. Keeping this in one place guarantees the two
backends report identical fields, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def collect_run_meta(
    *,
    backend: str,
    cfg: dict,
    n_train: int,
    n_val: int,
    train_metrics: dict,
    max_steps: int | None,
    wall_seconds: float,
) -> dict:
    """Assemble the run-metadata dict. `train_metrics` is trainer.train()'s output."""
    t = cfg["train"]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    # runtime the Trainer measured (excludes model load/tokenize); wall_seconds is
    # the whole train() call including any eval passes.
    train_runtime = train_metrics.get("train_runtime")
    global_step = train_metrics.get("step") or train_metrics.get("global_step")
    return {
        "backend": backend,  # "hf" | "unsloth" — the whole point of this file
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": cfg["model_name"],
        "gpu": gpu,
        "torch_version": torch.__version__,
        # --- data ---
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        # --- schedule ---
        "epochs": t["epochs"],
        "max_steps_arg": max_steps,          # None for a full run; set for smoke tests
        "global_steps": global_step,
        "effective_batch": t["per_device_batch"] * t["grad_accum"],
        "per_device_batch": t["per_device_batch"],
        "grad_accum": t["grad_accum"],
        "max_seq_len": cfg["max_seq_len"],
        # --- timing (the comparison payload) ---
        "wall_seconds": round(wall_seconds, 1),
        "train_runtime_seconds": round(train_runtime, 1) if train_runtime else None,
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "seconds_per_step": (
            round(train_runtime / global_step, 3) if train_runtime and global_step else None
        ),
        # --- final loss signal ---
        "train_loss": train_metrics.get("train_loss"),
        "mean_token_accuracy": train_metrics.get("mean_token_accuracy"),
        "platform": platform.platform(),
    }


def write_run_meta(out_dir: Path, meta: dict) -> None:
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Run metadata: {path}")
    print(
        f"  backend={meta['backend']} | {meta['n_train_samples']} train samples | "
        f"{meta['epochs']} epoch(s) | {meta['global_steps']} steps | "
        f"wall={meta['wall_seconds']}s | {meta['seconds_per_step']}s/step"
    )
