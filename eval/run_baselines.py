"""Phase 3.1 baselines — evaluate zero/few-shot Qwen2.5-3B and gpt-4o-mini.

Runs each configured model over the frozen test_iid subset (eval/subset.py),
scores every output with the shared metric core (eval/metrics.py), and writes
eval/results/baselines.json (committed artifact) + a compact markdown table.

The three baseline configs:
  base_zeroshot  — Qwen2.5-3B-Instruct via Ollama, no examples
  base_fewshot   — same model, N few-shot demos from the train split
  frontier       — gpt-4o-mini via the OpenAI API, zero-shot

These are the floor the fine-tuned model must beat in Phase 3.2+.

Usage:
    uv run python -m eval.run_baselines                 # all 3, full subset
    uv run python -m eval.run_baselines --n 3           # smoke test
    uv run python -m eval.run_baselines --models frontier
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from eval import models as model_mod
from eval.metrics import aggregate, parse_output, score_sample
from eval.prompts import build_messages
from eval.subset import SUBSET, ensure_subset

RESULTS_DIR = Path("eval/results")
RESULTS_JSON = RESULTS_DIR / "baselines.json"
RESULTS_MD = RESULTS_DIR / "baselines.md"

OLLAMA_MODEL = "qwen2.5:3b-instruct"
FRONTIER_MODEL = "gpt-4o-mini"

# Note in the report: Ollama serves a Q4_K_M quant, not the bf16 weights the
# trainer starts from — this is a serving-quant floor, called out honestly.
OLLAMA_QUANT_NOTE = "Ollama default quant (Q4_K_M); not the bf16 base the trainer starts from."


def _make_model(name: str, fewshot: int):
    """Return (caller, few_shot_count) for a baseline config name."""
    if name == "base_zeroshot":
        return model_mod.ollama_model(OLLAMA_MODEL), 0
    if name == "base_fewshot":
        return model_mod.ollama_model(OLLAMA_MODEL), fewshot
    if name == "frontier":
        return model_mod.openai_model(FRONTIER_MODEL), 0
    raise ValueError(f"unknown model config: {name}")


def evaluate(name: str, samples: list[dict], fewshot: int) -> dict:
    """Run one baseline config over all samples and aggregate its metrics."""
    caller, n_shot = _make_model(name, fewshot)
    per_sample = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        messages = build_messages(s["raw_message"], s["format"], few_shot=n_shot)
        output = caller.generate(messages)
        pred = parse_output(output)
        per_sample.append(score_sample(pred, s["target"], s["raw_message"]))
        if i % 20 == 0 or i == len(samples):
            print(f"  [{name}] {i}/{len(samples)}")
    elapsed = time.time() - t0

    agg = aggregate(per_sample)
    agg["seconds"] = round(elapsed, 1)
    agg["sec_per_sample"] = round(elapsed / len(samples), 3) if samples else 0.0
    agg["few_shot"] = n_shot
    print(
        f"  [{name}] done in {elapsed:.0f}s — "
        f"schema_valid={agg['schema_valid_rate']:.2%} "
        f"macroF1={agg['exception_macro_f1']:.3f} "
        f"halluc={agg['hallucination_rate']:.2%}"
    )
    return agg


def _md_table(results: dict) -> str:
    """Compact headline table for the report."""
    header = (
        "| model | n | schema-valid | exc macro-F1 | exc micro-F1 | "
        "header exact-match | hallucination | sec/sample |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, m in results.items():
        rows.append(
            f"| {name} | {m['n']} | {m['schema_valid_rate']:.1%} | "
            f"{m['exception_macro_f1']:.3f} | {m['exception_micro_f1']:.3f} | "
            f"{m['header_exact_match_rate']:.1%} | {m['hallucination_rate']:.1%} | "
            f"{m['sec_per_sample']} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="base_zeroshot,base_fewshot,frontier",
        help="comma-separated baseline configs to run",
    )
    parser.add_argument("--n", type=int, default=200, help="subset target size")
    parser.add_argument("--fewshot", type=int, default=2, help="few-shot example count")
    parser.add_argument(
        "--limit", type=int, default=None, help="cap samples actually evaluated (smoke tests)"
    )
    args = parser.parse_args()

    samples = ensure_subset(args.n)
    if args.limit is not None:
        samples = samples[: args.limit]
    subset_hash = hashlib.sha256(SUBSET.read_bytes()).hexdigest()
    print(f"Evaluating {len(samples)} samples (subset sha256={subset_hash[:12]}…)")

    configs = [m.strip() for m in args.models.split(",") if m.strip()]
    results: dict[str, dict] = {}
    for name in configs:
        print(f"\n== {name} ==")
        results[name] = evaluate(name, samples, args.fewshot)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "phase": "3.1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(samples),
            "subset_file": str(SUBSET),
            "subset_sha256": subset_hash,
            "fewshot": args.fewshot,
            "models": {
                "base": f"{OLLAMA_MODEL} (Ollama)",
                "frontier": f"{FRONTIER_MODEL} (OpenAI API)",
            },
            "notes": OLLAMA_QUANT_NOTE,
        },
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(
        f"# Phase 3.1 baselines\n\n"
        f"n={len(samples)} · subset sha256 `{subset_hash[:12]}…` · "
        f"few-shot={args.fewshot} · {OLLAMA_QUANT_NOTE}\n\n" + _md_table(results),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS_JSON} and {RESULTS_MD}")


if __name__ == "__main__":
    main()
