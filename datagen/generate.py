"""Generate internally-consistent, schema-valid "clean" transactions.

Each generated sample is one `CanonicalTransaction` (schema.py) with no
injected defects: exceptions=[NONE], all DSCSA statements present, quantities
and serials self-consistent. This is the ground truth that Phase 2.3's defect
injectors will later mutate, and what Phase 2.2's renderers turn into raw
EPCIS XML / X12-856 / CSV text.

Simplification vs. tasks.md's "commissioning event + shipping event + ASN +
optional T3" bundle description: schema.py's CanonicalTransaction models a
single transaction, not a linked multi-message set. So one generated sample =
one self-consistent transaction of a randomly chosen transaction_type, not
four cross-referenced messages. See claude/executions/phase_2.md for the
full rationale.

Usage:
    uv run python -m datagen.generate                 # 1000 bundles (default)
    uv run python -m datagen.generate --n 5000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from schema import (
    CanonicalTransaction,
    Dscsa,
    Exception_,
    ExceptionType,
    MessageFormat,
    PackagingLevel,
    Product,
    Severity,
    TransactionType,
)

NDC_PATH = Path("data/ref/ndc.parquet")
OUT_PATH = Path("data/corpus/clean_bundles.jsonl")

# GS1 company prefixes to draw fake GLNs/GTINs from -- fixed pool of
# plausible-looking prefixes rather than fully random digits, so generated
# IDs read like real GS1 identifiers (this matters for the renderers, which
# encode these into EPCIS URIs / X12 segments / CSV columns).
COMPANY_PREFIXES = ["0614141", "0037000", "0012345", "4012345", "0300060"]


def _luhn_check_digit(digits: str) -> str:
    """Compute the GS1 mod-10 check digit for a digit string (used for GLNs/GTINs)."""
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        total += n * (3 if i % 2 == 0 else 1)
    return str((10 - total % 10) % 10)


def _make_gln(rng: random.Random) -> str:
    prefix = rng.choice(COMPANY_PREFIXES)
    location_ref = f"{rng.randrange(10**6):06d}"
    body = f"{prefix}{location_ref}"[:12]
    return body + _luhn_check_digit(body)


def _make_gtin(rng: random.Random, ndc: str) -> str:
    """Synthesize a GTIN-14 that encodes the NDC digits (NDCs aren't real GTINs)."""
    ndc_digits = "".join(ch for ch in ndc if ch.isdigit()).ljust(10, "0")[:10]
    indicator = str(rng.randrange(1, 9))
    prefix = rng.choice(COMPANY_PREFIXES)
    body = (indicator + prefix + ndc_digits)[:13]
    return body + _luhn_check_digit(body)


def _make_serial(rng: random.Random) -> str:
    return f"{rng.randrange(10**8):08d}"


def _make_lot(rng: random.Random) -> str:
    return f"L{rng.randrange(10**6):06d}"


def _make_event_time(rng: random.Random) -> datetime:
    # Spread events over the last two years so expiry dates below always land
    # comfortably in the future for a "clean" (non-expired) transaction.
    days_ago = rng.randrange(0, 730)
    return datetime(2026, 7, 6) - timedelta(days=days_ago)


def generate_clean_bundle(rng: random.Random, ndc_df: pd.DataFrame) -> CanonicalTransaction:
    """Build one internally-consistent, defect-free CanonicalTransaction."""
    transaction_type = rng.choice(list(TransactionType))
    event_time = _make_event_time(rng)
    expiration_date = (event_time + timedelta(days=rng.randrange(180, 900))).date()

    ndc_row = ndc_df.sample(1, random_state=rng.randrange(2**31)).iloc[0]
    lot_number = _make_lot(rng)
    gtin = _make_gtin(rng, ndc_row["ndc"])

    unit_count = rng.randrange(3, 11)
    unit_serials = [_make_serial(rng) for _ in range(unit_count)]

    if transaction_type == TransactionType.AGGREGATION:
        # 2-level tree: N units aggregated under 1 case. Pallet level
        # (case->pallet) is intentionally omitted -- same recursive pattern,
        # not needed to prove parent/child linkage for BROKEN_AGGREGATION.
        case_serial = _make_serial(rng)
        products = [
            Product(
                ndc=ndc_row["ndc"],
                gtin=gtin,
                product_name=ndc_row["product_name"],
                lot_number=lot_number,
                expiration_date=expiration_date.isoformat(),
                serial_numbers=[serial],
                quantity=1,
                packaging_level=PackagingLevel.UNIT,
                parent_serial=case_serial,
            )
            for serial in unit_serials
        ] + [
            Product(
                ndc=ndc_row["ndc"],
                gtin=gtin,
                product_name=ndc_row["product_name"],
                lot_number=lot_number,
                expiration_date=expiration_date.isoformat(),
                serial_numbers=[case_serial],
                quantity=unit_count,
                packaging_level=PackagingLevel.CASE,
                parent_serial=None,
            )
        ]
    else:
        products = [
            Product(
                ndc=ndc_row["ndc"],
                gtin=gtin,
                product_name=ndc_row["product_name"],
                lot_number=lot_number,
                expiration_date=expiration_date.isoformat(),
                serial_numbers=unit_serials,
                quantity=unit_count,
                packaging_level=PackagingLevel.UNIT,
                parent_serial=None,
            )
        ]

    return CanonicalTransaction(
        message_id=str(uuid.uuid4()),
        message_format=MessageFormat.EPCIS_XML,  # renderer (2.2) overwrites per output format
        transaction_type=transaction_type,
        sender_gln=_make_gln(rng),
        receiver_gln=_make_gln(rng),
        event_time=event_time.isoformat() + "Z",
        products=products,
        dscsa=Dscsa(t1_present=True, t2_present=True, t3_present=True),
        exceptions=[
            Exception_(
                type=ExceptionType.NONE,
                severity=Severity.MINOR,
                rationale="No violated rules detected.",
                recommended_resolution="No action needed.",
            )
        ],
        confidence=1.0,
    )


def generate_corpus(n: int, seed: int, ndc_df: pd.DataFrame) -> list[CanonicalTransaction]:
    rng = random.Random(seed)
    return [generate_clean_bundle(rng, ndc_df) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="number of clean bundles to generate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ndc_df = pd.read_parquet(NDC_PATH)
    bundles = generate_corpus(args.n, args.seed, ndc_df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for bundle in bundles:
            f.write(bundle.model_dump_json() + "\n")

    print(f"wrote {len(bundles)} clean bundles to {OUT_PATH}")


if __name__ == "__main__":
    main()
