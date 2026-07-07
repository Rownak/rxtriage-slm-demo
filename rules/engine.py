"""Deterministic exception rule engine — the ground-truth oracle.

`detect_exceptions()` inspects one canonical transaction (schema.py) plus two
reference sets (valid NDCs, recalled lots — from the Phase 1 parquets) and
returns the list of exceptions present. This is the ONLY source of ground-truth
labels in the project (see tasks.md working agreements): the defect injectors
in datagen/inject.py stamp their expected label using the same
`build_exception()` templates below, and tests/test_engine.py proves the
engine independently re-derives identical labels from the mutated data.

All rules are single-message rules: the engine is a pure function of one
canonical JSON, which mirrors what the model sees at inference time (one raw
message in, one triaged JSON out). The two taxonomy entries originally worded
cross-message (QTY_MISMATCH, DUPLICATE_SERIAL) are redefined within-message —
see claude/executions/phase_2.md.
"""

from __future__ import annotations

from datetime import date, datetime

from schema import (
    CanonicalTransaction,
    Exception_,
    ExceptionType,
    PackagingLevel,
    Product,
    Severity,
)

# One entry per exception type: severity + rationale/resolution templates.
# Rationales cite the violated rule, per tasks.md 2.3. Injectors and the
# engine both build exceptions from these templates via build_exception(),
# so label text can never drift between the two.
EXCEPTION_META: dict[ExceptionType, dict] = {
    ExceptionType.QTY_MISMATCH: {
        "severity": Severity.MAJOR,
        "rationale": (
            "Declared quantity ({declared}) does not match the {actual} serial "
            "numbers listed; DSCSA §582 transaction information must accurately "
            "reflect the product transferred."
        ),
        "resolution": (
            "Request a corrected message from the sender reconciling the declared "
            "quantity with the serialized units."
        ),
    },
    ExceptionType.MISSING_T3: {
        "severity": Severity.CRITICAL,
        "rationale": (
            "DSCSA §582(b)(1) requires a transaction statement (T3) with each "
            "change of ownership, but none is present in this message."
        ),
        "resolution": (
            "Hold the shipment and request the missing transaction statement from "
            "the sending trading partner before accepting ownership."
        ),
    },
    ExceptionType.EXPIRED_LOT: {
        "severity": Severity.CRITICAL,
        "rationale": (
            "Event dated {event_date} occurred after lot {lot} expired on "
            "{expiration_date}; distributing expired product violates FD&C Act "
            "§501/§502 adulteration and misbranding provisions."
        ),
        "resolution": (
            "Quarantine the expired units and initiate a return or destruction "
            "per the trading partner agreement."
        ),
    },
    ExceptionType.BROKEN_AGGREGATION: {
        "severity": Severity.MAJOR,
        "rationale": (
            "{detail}; DSCSA serialized tracing requires an intact aggregation "
            "hierarchy from unit to case."
        ),
        "resolution": (
            "Request a corrected aggregation (packing) event from the sender "
            "re-establishing the parent-child hierarchy."
        ),
    },
    ExceptionType.DUPLICATE_SERIAL: {
        "severity": Severity.CRITICAL,
        "rationale": (
            "Serial number {serial} appears {count} times in this message; DSCSA "
            "§581(14) requires a unique product identifier per saleable unit."
        ),
        "resolution": (
            "Quarantine the affected units and verify serials with the "
            "manufacturer — duplicates may indicate counterfeit product."
        ),
    },
    ExceptionType.INVALID_NDC: {
        "severity": Severity.MAJOR,
        "rationale": (
            "NDC {ndc} is not found in the FDA NDC Directory; DSCSA transaction "
            "information requires a valid product identifier (21 CFR Part 207)."
        ),
        "resolution": (
            "Verify the product code with the sender and request a corrected "
            "message with a registered NDC."
        ),
    },
    ExceptionType.RECALLED_LOT: {
        "severity": Severity.CRITICAL,
        "rationale": (
            "Lot {lot} appears in FDA drug enforcement (recall) records; recalled "
            "product must not be further distributed (21 CFR Part 7)."
        ),
        "resolution": (
            "Quarantine the lot immediately and follow the recall instructions in "
            "the FDA enforcement report."
        ),
    },
    ExceptionType.NONE: {
        "severity": Severity.MINOR,
        "rationale": "No violated rules detected.",
        "resolution": "No action needed.",
    },
}

# Deterministic output order for multi-exception messages.
_TYPE_ORDER = {t: i for i, t in enumerate(ExceptionType)}


def build_exception(exc_type: ExceptionType, **fmt) -> Exception_:
    """Build an Exception_ from the shared templates. Used by engine AND injectors."""
    meta = EXCEPTION_META[exc_type]
    return Exception_(
        type=exc_type,
        severity=meta["severity"],
        rationale=meta["rationale"].format(**fmt),
        recommended_resolution=meta["resolution"],
    )


def _event_date(txn: CanonicalTransaction) -> date:
    # event_time is ISO-8601, possibly with a trailing Z.
    return datetime.fromisoformat(txn.event_time.replace("Z", "+00:00")).date()


def _units_and_cases(txn: CanonicalTransaction) -> tuple[list[Product], list[Product]]:
    units = [p for p in txn.products if p.packaging_level == PackagingLevel.UNIT]
    cases = [p for p in txn.products if p.packaging_level == PackagingLevel.CASE]
    return units, cases


def detect_exceptions(
    txn: CanonicalTransaction,
    valid_ndcs: set[str],
    recalled_lots: set[str],
) -> list[Exception_]:
    """Detect all exceptions in one canonical transaction.

    Returns detected exceptions sorted by taxonomy order, or [NONE] if clean.
    Each rule fires at most once per message (first offending product wins,
    in product order) — one exception entry per defect *type*, matching how
    the injectors label.
    """
    found: list[Exception_] = []
    units, cases = _units_and_cases(txn)

    # QTY_MISMATCH — unit-level products only: a case legitimately declares
    # quantity = number of children while carrying its own single serial.
    for p in units:
        if p.quantity != len(p.serial_numbers):
            found.append(
                build_exception(
                    ExceptionType.QTY_MISMATCH,
                    declared=p.quantity,
                    actual=len(p.serial_numbers),
                )
            )
            break

    # MISSING_T3
    if not txn.dscsa.t3_present:
        found.append(build_exception(ExceptionType.MISSING_T3))

    # EXPIRED_LOT
    event_date = _event_date(txn)
    for p in txn.products:
        if event_date > date.fromisoformat(p.expiration_date):
            found.append(
                build_exception(
                    ExceptionType.EXPIRED_LOT,
                    event_date=event_date.isoformat(),
                    lot=p.lot_number,
                    expiration_date=p.expiration_date,
                )
            )
            break

    # BROKEN_AGGREGATION — (a) orphaned child: unit's parent_serial doesn't
    # match any case serial; (b) count mismatch: case quantity != linked units.
    case_serials = {s for c in cases for s in c.serial_numbers}
    broken: Exception_ | None = None
    for p in units:
        if p.parent_serial is not None and p.parent_serial not in case_serials:
            broken = build_exception(
                ExceptionType.BROKEN_AGGREGATION,
                detail=(
                    f"Unit {p.serial_numbers[0]} references parent "
                    f"{p.parent_serial}, which is not present in this message"
                ),
            )
            break
    if broken is None:
        for c in cases:
            linked = sum(1 for p in units if p.parent_serial == c.serial_numbers[0])
            if c.quantity != linked:
                broken = build_exception(
                    ExceptionType.BROKEN_AGGREGATION,
                    detail=(
                        f"Case {c.serial_numbers[0]} declares {c.quantity} children "
                        f"but {linked} linked units are present"
                    ),
                )
                break
    if broken is not None:
        found.append(broken)

    # DUPLICATE_SERIAL — same serial listed more than once across the message.
    seen: dict[str, int] = {}
    for p in txn.products:
        for s in p.serial_numbers:
            seen[s] = seen.get(s, 0) + 1
    for s, count in seen.items():  # dict preserves first-seen order → deterministic
        if count > 1:
            found.append(build_exception(ExceptionType.DUPLICATE_SERIAL, serial=s, count=count))
            break

    # INVALID_NDC
    for p in txn.products:
        if p.ndc not in valid_ndcs:
            found.append(build_exception(ExceptionType.INVALID_NDC, ndc=p.ndc))
            break

    # RECALLED_LOT
    for p in txn.products:
        if p.lot_number in recalled_lots:
            found.append(build_exception(ExceptionType.RECALLED_LOT, lot=p.lot_number))
            break

    if not found:
        return [build_exception(ExceptionType.NONE)]
    return sorted(found, key=lambda e: _TYPE_ORDER[e.type])
