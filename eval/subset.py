"""Build (and freeze) the stratified evaluation subset of test_iid.

Evaluating every model on the full 843-sample test_iid split is unnecessary for
a baseline floor and multiplies frontier-API cost. Instead we draw a ~200-sample
subset stratified across (format x primary-exception-type) so every format and
every defect class is represented in proportion, then freeze it to disk with a
SHA-256 so all later phases score the identical samples.

Deterministic: seed=42. Idempotent: if the subset already exists it is loaded
and its hash verified rather than regenerated.

Usage:
    uv run python -m eval.subset            # build/verify default 200-sample subset
    uv run python -m eval.subset --n 100    # different target size
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

CORPUS_DIR = Path("data/corpus")
SOURCE = CORPUS_DIR / "test_iid.jsonl"
SUBSET = CORPUS_DIR / "test_iid_subset.jsonl"
SEED = 42


def primary_exception(sample: dict) -> str:
    """The sample's headline defect class: first non-NONE exception, else NONE."""
    for exc in sample["target"].get("exceptions", []):
        if exc.get("type") and exc["type"] != "NONE":
            return exc["type"]
    return "NONE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_subset(n: int, seed: int = SEED) -> list[dict]:
    """Stratified sample of ~n records across (format, primary-exception) strata."""
    with open(SOURCE, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        strata[(s["format"], primary_exception(s))].append(s)

    rng = random.Random(seed)
    total = len(samples)
    picked: list[dict] = []
    # Proportional allocation with a floor of 1 per non-empty stratum, so small
    # classes (and each format) are never dropped entirely.
    for key in sorted(strata):  # sorted → deterministic allocation order
        bucket = strata[key]
        take = max(1, round(n * len(bucket) / total))
        take = min(take, len(bucket))
        picked.extend(rng.sample(bucket, take))

    rng.shuffle(picked)  # mix strata so progress output isn't class-ordered
    return picked


def write_subset(records: list[dict]) -> None:
    with open(SUBSET, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _class_balance(records: list[dict]) -> None:
    by_fmt: dict[str, int] = defaultdict(int)
    by_exc: dict[str, int] = defaultdict(int)
    for r in records:
        by_fmt[r["format"]] += 1
        by_exc[primary_exception(r)] += 1
    print(f"subset size: {len(records)}")
    print("  by format:", dict(sorted(by_fmt.items())))
    print("  by exception:", dict(sorted(by_exc.items())))


def ensure_subset(n: int = 200, seed: int = SEED) -> list[dict]:
    """Return the frozen subset, building it once if absent. Used by run_baselines."""
    if SUBSET.exists():
        with open(SUBSET, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    records = build_subset(n, seed)
    write_subset(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="target subset size")
    parser.add_argument("--force", action="store_true", help="rebuild even if subset exists")
    args = parser.parse_args()

    if SUBSET.exists() and not args.force:
        records = ensure_subset(args.n)
        print(f"subset already exists ({SUBSET}); sha256={_sha256(SUBSET)}")
    else:
        records = build_subset(args.n)
        write_subset(records)
        print(f"wrote {SUBSET}; sha256={_sha256(SUBSET)}")
    _class_balance(records)


if __name__ == "__main__":
    main()
