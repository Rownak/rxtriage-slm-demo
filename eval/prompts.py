"""Prompt construction for baseline evaluation.

Reuses the EXACT system prompt and user-turn format the training records use
(datagen/format_chat.py) so the baseline sees precisely what a fine-tuned model
would — no prompt drift between eval and training. `SYSTEM_PROMPT` is imported,
not copied, to keep a single source of truth.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Single source of truth for the system prompt + user-turn shape.
from datagen.format_chat import SYSTEM_PROMPT  # noqa: F401  (re-exported)

TRAIN_PATH = Path("data/corpus/train.jsonl")


def _user_content(raw_message: str, fmt: str) -> str:
    """Mirror datagen.format_chat.build_chat's user turn exactly."""
    return f"Format: {fmt}\n\n{raw_message}"


_fewshot_cache: dict[tuple[int, int], list[dict]] = {}


def _load_fewshot_pool(n: int, seed: int) -> list[dict]:
    """Draw n few-shot examples from the TRAIN split only (memoized per n/seed).

    Never from a test/val split — few-shot demos are effectively conditioning
    data, and the working agreements forbid touching held-out splits. Seeded so
    the demos are identical across models and runs; cached so train.jsonl isn't
    re-read once per evaluated sample.
    """
    if n <= 0:
        return []
    key = (n, seed)
    if key not in _fewshot_cache:
        with open(TRAIN_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        rng = random.Random(seed)
        picks = rng.sample(lines, n)
        _fewshot_cache[key] = [json.loads(line) for line in picks]
    return _fewshot_cache[key]


def build_messages(
    raw_message: str,
    fmt: str,
    few_shot: int = 0,
    fewshot_seed: int = 7,
) -> list[dict]:
    """Build the OpenAI-style messages list for one sample.

    few_shot > 0 prepends that many (user=raw, assistant=canonical JSON) example
    turns drawn from the train split, between the system prompt and the query.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in _load_fewshot_pool(few_shot, fewshot_seed):
        messages.append({"role": "user", "content": _user_content(ex["raw_message"], ex["format"])})
        messages.append(
            {"role": "assistant", "content": json.dumps(ex["target"], separators=(",", ":"))}
        )
    messages.append({"role": "user", "content": _user_content(raw_message, fmt)})
    return messages
