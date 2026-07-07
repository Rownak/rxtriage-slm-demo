"""Defect injectors — one per exception type in the taxonomy.

Each injector deep-copies a clean bundle (datagen/generate.py), applies the
minimal data mutation for its defect, and stamps the expected ground-truth
label using the SAME `build_exception()` templates the rule engine uses
(rules/engine.py). The engine then independently re-detects the defect from
the mutated data; tests/test_engine.py proves both sides agree exactly.

Mutations are deliberately non-colliding — each injector triggers its own
rule and only its own rule (e.g. DUPLICATE_SERIAL overwrites a serial rather
than appending one, so the quantity/serial-count consistency that
QTY_MISMATCH checks is preserved).

BROKEN_AGGREGATION requires an aggregation bundle (the only kind with a
parent/child hierarchy) and raises ValueError otherwise — corpus assembly
(task 2.5) is responsible for feeding it suitable bundles.
"""

from __future__ import annotations

import random
from datetime import timedelta

from rules.engine import _event_date, _units_and_cases, build_exception
from schema import CanonicalTransaction, ExceptionType


def _fresh_serial(rng: random.Random, taken: set[str]) -> str:
    while True:
        serial = f"{rng.randrange(10**8):08d}"
        if serial not in taken:
            return serial


def inject_qty_mismatch(bundle: CanonicalTransaction, rng: random.Random) -> CanonicalTransaction:
    """Perturb a unit-level product's declared quantity away from its serial count."""
    txn = bundle.model_copy(deep=True)
    units, _ = _units_and_cases(txn)
    target = units[0]  # first unit = the one the engine reports on
    target.quantity += rng.randrange(1, 4)
    txn.exceptions = [
        build_exception(
            ExceptionType.QTY_MISMATCH,
            declared=target.quantity,
            actual=len(target.serial_numbers),
        )
    ]
    return txn


def inject_missing_t3(bundle: CanonicalTransaction, rng: random.Random) -> CanonicalTransaction:
    """Drop the DSCSA transaction statement."""
    txn = bundle.model_copy(deep=True)
    txn.dscsa.t3_present = False
    txn.exceptions = [build_exception(ExceptionType.MISSING_T3)]
    return txn


def inject_expired_lot(bundle: CanonicalTransaction, rng: random.Random) -> CanonicalTransaction:
    """Move the lot's expiration date to before the event date."""
    txn = bundle.model_copy(deep=True)
    expired = _event_date(txn) - timedelta(days=rng.randrange(1, 366))
    for p in txn.products:  # all products share one lot, so expire them together
        p.expiration_date = expired.isoformat()
    txn.exceptions = [
        build_exception(
            ExceptionType.EXPIRED_LOT,
            event_date=_event_date(txn).isoformat(),
            lot=txn.products[0].lot_number,
            expiration_date=expired.isoformat(),
        )
    ]
    return txn


def inject_broken_aggregation(
    bundle: CanonicalTransaction, rng: random.Random, mode: str = "random"
) -> CanonicalTransaction:
    """Break the unit->case hierarchy: orphan a child OR remove one child.

    Only meaningful on aggregation bundles (the only kind with parent/child
    links); raises ValueError for anything else.

    mode: "orphan" | "count" | "random". Corpus assembly (2.5) uses "count"
    because the orphan defect lives only in the canonical JSON's parent_serial
    and is structurally inexpressible in a rendered EPCIS AggregationEvent
    (single parentID) — the raw message would carry no evidence of it.
    """
    txn = bundle.model_copy(deep=True)
    units, cases = _units_and_cases(txn)
    if not cases:
        raise ValueError("BROKEN_AGGREGATION requires an aggregation bundle with a case")
    case = cases[0]

    if mode == "orphan" or (mode == "random" and rng.random() < 0.5):
        # Orphan mode: point the first unit at a parent that isn't in the message.
        taken = {s for p in txn.products for s in p.serial_numbers}
        orphan = units[0]
        orphan.parent_serial = _fresh_serial(rng, taken)
        detail = (
            f"Unit {orphan.serial_numbers[0]} references parent "
            f"{orphan.parent_serial}, which is not present in this message"
        )
    else:
        # Count mode: remove one child unit; the case still declares the old count.
        removed = rng.choice(units)
        txn.products.remove(removed)
        detail = (
            f"Case {case.serial_numbers[0]} declares {case.quantity} children "
            f"but {case.quantity - 1} linked units are present"
        )

    txn.exceptions = [build_exception(ExceptionType.BROKEN_AGGREGATION, detail=detail)]
    return txn


def inject_duplicate_serial(
    bundle: CanonicalTransaction, rng: random.Random
) -> CanonicalTransaction:
    """Overwrite the message's second serial slot with a copy of the first.

    Overwriting (not appending) keeps quantity == serial count, so this never
    also trips QTY_MISMATCH.
    """
    txn = bundle.model_copy(deep=True)
    slots = [
        (p, i) for p in txn.products for i in range(len(p.serial_numbers))
    ]  # flattened in product order — matches the engine's first-seen scan
    dup = slots[0][0].serial_numbers[slots[0][1]]
    second_product, second_idx = slots[1]
    second_product.serial_numbers[second_idx] = dup
    txn.exceptions = [build_exception(ExceptionType.DUPLICATE_SERIAL, serial=dup, count=2)]
    return txn


def inject_invalid_ndc(
    bundle: CanonicalTransaction, rng: random.Random, valid_ndcs: set[str]
) -> CanonicalTransaction:
    """Replace the NDC with a fabricated code absent from the FDA directory."""
    txn = bundle.model_copy(deep=True)
    while True:
        fake = f"{rng.randrange(10**5):05d}-{rng.randrange(10**4):04d}"
        if fake not in valid_ndcs:
            break
    for p in txn.products:  # all products share one NDC, keep them consistent
        p.ndc = fake
    txn.exceptions = [build_exception(ExceptionType.INVALID_NDC, ndc=fake)]
    return txn


def inject_recalled_lot(
    bundle: CanonicalTransaction, rng: random.Random, recalled_lots: set[str]
) -> CanonicalTransaction:
    """Swap the lot number for a real recalled lot from openFDA enforcement data."""
    txn = bundle.model_copy(deep=True)
    lot = rng.choice(sorted(recalled_lots))  # sorted -> deterministic under a seeded rng
    for p in txn.products:
        p.lot_number = lot
    txn.exceptions = [build_exception(ExceptionType.RECALLED_LOT, lot=lot)]
    return txn


def inject(
    exc_type: ExceptionType,
    bundle: CanonicalTransaction,
    rng: random.Random,
    valid_ndcs: set[str],
    recalled_lots: set[str],
) -> CanonicalTransaction:
    """Dispatch to the injector for `exc_type` (corpus assembly entry point)."""
    if exc_type == ExceptionType.INVALID_NDC:
        return inject_invalid_ndc(bundle, rng, valid_ndcs)
    if exc_type == ExceptionType.RECALLED_LOT:
        return inject_recalled_lot(bundle, rng, recalled_lots)
    simple = {
        ExceptionType.QTY_MISMATCH: inject_qty_mismatch,
        ExceptionType.MISSING_T3: inject_missing_t3,
        ExceptionType.EXPIRED_LOT: inject_expired_lot,
        ExceptionType.BROKEN_AGGREGATION: inject_broken_aggregation,
        ExceptionType.DUPLICATE_SERIAL: inject_duplicate_serial,
    }
    return simple[exc_type](bundle, rng)


INJECTABLE_TYPES = [t for t in ExceptionType if t != ExceptionType.NONE]
