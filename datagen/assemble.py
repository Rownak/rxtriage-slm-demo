"""Assemble the labeled corpus: 10k (raw message, format) -> (canonical JSON) samples.

Pipeline per sample: draw an engine-clean bundle (datagen/generate.py) ->
optionally inject one defect (datagen/inject.py) -> re-verify the label with
the rule engine (rules/engine.py, the ground-truth oracle) -> render to one
raw format (datagen/render.py). Class balance: ~30% NONE, the rest spread
evenly across the 7 defect types; formats split ~equally EPCIS/X12/CSV.

Defect-visibility routing (see claude/executions/phase_2.md "Open issues"):
  - Aggregation bundles render to EPCIS only — the X12/CSV renderers emit
    products[0] only, so a multi-product target would train hallucination.
  - BROKEN_AGGREGATION uses the injector's "count" mode only; the orphan
    mode is inexpressible in a rendered EPCIS AggregationEvent.
  - QTY_MISMATCH uses flat (non-aggregation) bundles only; unit-level
    quantities aren't rendered in aggregation EPCIS.

Splits (frozen; SHA-256 manifest in stats.md):
  - test_heldout_defect: ALL RECALLED_LOT samples (type never seen in train)
  - test_heldout_format: remaining partner_e-dialect CSV samples
  - train / val / test_iid: 80/10/10 seeded shuffle of the rest

Usage:
    uv run python -m datagen.assemble                # 10k samples, seed 42
    uv run python -m datagen.assemble --total 200 --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from datagen.generate import generate_clean_bundle
from datagen.inject import inject, inject_broken_aggregation
from datagen.render import PARTNER_DIALECTS, render_csv, render_epcis_xml, render_x12_856
from rules.engine import detect_exceptions
from schema import CanonicalTransaction, ExceptionType, MessageFormat

OUT_DIR = Path("data/corpus")
NDC_PATH = Path("data/ref/ndc.parquet")
RECALLS_PATH = Path("data/ref/recalled_lots.parquet")

NONE_FRACTION = 0.3
HELDOUT_DEFECT = ExceptionType.RECALLED_LOT
HELDOUT_DIALECT = "partner_e"
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # remainder -> test_iid

DEFECT_TYPES = [t for t in ExceptionType if t != ExceptionType.NONE]


def _draw_clean_bundle(
    rng: random.Random,
    ndc_df: pd.DataFrame,
    valid_ndcs: set[str],
    recalled_lots: set[str],
    aggregation: bool | None,
) -> CanonicalTransaction:
    """Draw until the bundle matches the aggregation constraint AND the engine
    calls it clean (guards against e.g. a random lot colliding with a real
    recalled lot)."""
    for _ in range(1000):
        bundle = generate_clean_bundle(rng, ndc_df)
        is_agg = bundle.transaction_type.value == "aggregation"
        if aggregation is not None and is_agg != aggregation:
            continue
        detected = detect_exceptions(bundle, valid_ndcs, recalled_lots)
        if [e.type for e in detected] == [ExceptionType.NONE]:
            return bundle
    raise RuntimeError("could not draw an engine-clean bundle in 1000 tries")


def _format_slots(total: int, n_broken_agg: int, rng: random.Random) -> list[MessageFormat]:
    """Format assignments for all non-BROKEN_AGGREGATION samples, sized so the
    overall corpus (including the EPCIS-only BROKEN_AGGREGATION block) lands
    at ~1/3 per format."""
    remaining = total - n_broken_agg
    epcis = max(0, round(total / 3) - n_broken_agg)
    x12 = (remaining - epcis + 1) // 2
    csv_n = remaining - epcis - x12
    assert min(epcis, x12, csv_n) >= 0
    slots = (
        [MessageFormat.EPCIS_XML] * epcis
        + [MessageFormat.X12_856] * x12
        + [MessageFormat.CSV] * csv_n
    )
    rng.shuffle(slots)
    return slots


def assemble_corpus(
    total: int,
    seed: int,
    ndc_df: pd.DataFrame,
    recalled_lots: set[str],
) -> list[dict]:
    rng = random.Random(seed)
    valid_ndcs = set(ndc_df["ndc"])

    n_none = round(total * NONE_FRACTION)
    per_defect = (total - n_none) // len(DEFECT_TYPES)
    n_none = total - per_defect * len(DEFECT_TYPES)  # remainder folds into NONE

    plan = [(ExceptionType.NONE, n_none)] + [(t, per_defect) for t in DEFECT_TYPES]
    slots = _format_slots(total, per_defect, rng)  # per_defect == BROKEN_AGGREGATION count
    samples: list[dict] = []

    for exc_type, count in plan:
        for _ in range(count):
            if exc_type == ExceptionType.BROKEN_AGGREGATION:
                fmt = MessageFormat.EPCIS_XML  # aggregation hierarchy only renders in EPCIS
                bundle = _draw_clean_bundle(rng, ndc_df, valid_ndcs, recalled_lots, True)
                txn = inject_broken_aggregation(bundle, rng, mode="count")
            else:
                fmt = slots.pop()
                # X12/CSV carry a single product; QTY_MISMATCH evidence lives in
                # unit-level quantities that aggregation EPCIS doesn't render.
                flat_only = fmt != MessageFormat.EPCIS_XML or exc_type == ExceptionType.QTY_MISMATCH
                bundle = _draw_clean_bundle(
                    rng, ndc_df, valid_ndcs, recalled_lots, False if flat_only else None
                )
                txn = (
                    bundle
                    if exc_type == ExceptionType.NONE
                    else inject(exc_type, bundle, rng, valid_ndcs, recalled_lots)
                )

            # The rule engine is the ground-truth oracle: every label ships
            # only after the engine independently re-derives it.
            detected = detect_exceptions(txn, valid_ndcs, recalled_lots)
            assert detected == txn.exceptions, f"oracle disagreement on {exc_type}"

            txn.message_format = fmt
            dialect = rng.choice(list(PARTNER_DIALECTS)) if fmt == MessageFormat.CSV else None
            if fmt == MessageFormat.EPCIS_XML:
                raw = render_epcis_xml(txn)
            elif fmt == MessageFormat.X12_856:
                raw = render_x12_856(txn)
            else:
                raw = render_csv(txn, dialect=dialect)

            samples.append(
                {
                    "id": txn.message_id,
                    "format": fmt.value,
                    "csv_dialect": dialect,
                    "raw_message": raw,
                    "target": json.loads(txn.model_dump_json()),
                }
            )

    assert not slots
    return samples


def make_splits(samples: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)

    def exc_types(s: dict) -> list[str]:
        return [e["type"] for e in s["target"]["exceptions"]]

    heldout_defect = [s for s in samples if HELDOUT_DEFECT.value in exc_types(s)]
    rest = [s for s in samples if HELDOUT_DEFECT.value not in exc_types(s)]
    heldout_format = [s for s in rest if s["csv_dialect"] == HELDOUT_DIALECT]
    rest = [s for s in rest if s["csv_dialect"] != HELDOUT_DIALECT]

    rng.shuffle(rest)
    n_train = round(len(rest) * TRAIN_FRAC)
    n_val = round(len(rest) * VAL_FRAC)
    return {
        "train": rest[:n_train],
        "val": rest[n_train : n_train + n_val],
        "test_iid": rest[n_train + n_val :],
        "test_heldout_defect": heldout_defect,
        "test_heldout_format": heldout_format,
    }


def write_corpus(splits: dict[str, list[dict]], out_dir: Path, seed: int, total: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    lines = [
        "# Corpus stats",
        "",
        f"Generated by `datagen/assemble.py` — total={total}, seed={seed}.",
        f"Held-out defect: {HELDOUT_DEFECT.value}. Held-out CSV dialect: {HELDOUT_DIALECT}.",
        "",
        "## Samples per split x exception type",
        "",
    ]
    all_types = [t.value for t in ExceptionType]
    lines.append("| split | total | " + " | ".join(all_types) + " |")
    lines.append("|" + "---|" * (len(all_types) + 2))
    for name, rows in splits.items():
        counts = Counter(e["type"] for s in rows for e in s["target"]["exceptions"])
        lines.append(
            f"| {name} | {len(rows)} | " + " | ".join(str(counts.get(t, 0)) for t in all_types) + " |"
        )

    lines += ["", "## Samples per split x format", ""]
    lines.append("| split | EPCIS_XML | X12_856 | CSV | CSV dialects |")
    lines.append("|---|---|---|---|---|")
    for name, rows in splits.items():
        fmt_counts = Counter(s["format"] for s in rows)
        dialects = Counter(s["csv_dialect"] for s in rows if s["csv_dialect"])
        dialect_str = ", ".join(f"{d}:{n}" for d, n in sorted(dialects.items())) or "-"
        lines.append(
            f"| {name} | {fmt_counts.get('EPCIS_XML', 0)} | {fmt_counts.get('X12_856', 0)} "
            f"| {fmt_counts.get('CSV', 0)} | {dialect_str} |"
        )

    lines += ["", "## Split manifest (SHA-256) — splits are frozen; do not regenerate", ""]
    for name, digest in hashes.items():
        lines.append(f"- `{name}.jsonl`: `{digest}`")
    lines.append("")

    (out_dir / "stats.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ndc_df = pd.read_parquet(NDC_PATH)
    recalled_lots = set(pd.read_parquet(RECALLS_PATH)["lot_number"])

    samples = assemble_corpus(args.total, args.seed, ndc_df, recalled_lots)
    splits = make_splits(samples, args.seed)
    write_corpus(splits, OUT_DIR, args.seed, args.total)

    for name, rows in splits.items():
        print(f"{name}: {len(rows)}")
    print(f"wrote splits + stats.md to {OUT_DIR}/")


if __name__ == "__main__":
    main()
