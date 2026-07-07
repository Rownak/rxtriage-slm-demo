"""Scoring core for RxTriage model outputs — the Phase 5 harness stub.

Given a model's raw text output and the oracle ground-truth `target` (the
canonical JSON stored in each corpus sample), this module computes the four
metrics the project cares about:

  1. schema-validity     — does the output parse AND satisfy schema.py?
  2. exception F1         — per-class + macro/micro F1 over the 8-class taxonomy
  3. hallucination rate   — predicted identifier values absent from the source
  4. exact-match rate     — normalization fields match the oracle exactly

Kept dependency-light on purpose (CLAUDE.md Simplicity Rule): F1 is computed by
hand from tp/fp/fn over 8 fixed classes rather than pulling in scikit-learn.
`aggregate()` is the function Phase 5.1 will extend to more models/splits.
"""

from __future__ import annotations

import json
import re

from schema import CanonicalTransaction, ExceptionType

# The 8-class exception taxonomy, in schema order. Fixed set → hand-rolled F1.
EXCEPTION_CLASSES = [t.value for t in ExceptionType]

# Identifier fields whose values must appear verbatim in the raw source message.
# A predicted value not found as a substring of the source is a hallucination
# (project_summary.md: "fields present in output but absent from source").
# We deliberately exclude free-text fields (product_name, rationale) and
# derived/format-translated fields (gtin, event_time) — those legitimately
# differ from any source substring.
_HALLUCINATION_FIELDS = ("ndc", "lot_number", "sender_gln", "receiver_gln")

# Field groups compared for exact-match. Exceptions are scored separately via F1,
# so exact-match here means "did the model normalize the transaction correctly".
_HEADER_FIELDS = (
    "message_format",
    "transaction_type",
    "sender_gln",
    "receiver_gln",
    "event_time",
)


def parse_output(text: str) -> dict | None:
    """Best-effort parse of a model's text into a JSON object. None on failure.

    Strips ```json fences some models emit despite JSON-mode, then json.loads.
    """
    if text is None:
        return None
    cleaned = text.strip()
    # Remove a leading/trailing markdown code fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def is_schema_valid(obj: dict | None) -> bool:
    """True iff obj validates against the canonical schema (schema.py contract)."""
    if obj is None:
        return False
    try:
        CanonicalTransaction.model_validate(obj)
        return True
    except Exception:  # pydantic ValidationError (and any malformed-input error)
        return False


def exception_labels(obj: dict | None) -> set[str]:
    """Set of exception `type` strings in a (predicted or gold) transaction.

    Drops NONE when any real exception is also present, matching oracle
    semantics: NONE is only meaningful as the sole label of a clean message.
    Unknown/garbage type strings are kept as-is so they count as false positives.
    """
    if not obj or not isinstance(obj.get("exceptions"), list):
        return set()
    types = set()
    for exc in obj["exceptions"]:
        if isinstance(exc, dict) and "type" in exc:
            types.add(exc["type"])
    if len(types) > 1:
        types.discard(ExceptionType.NONE.value)
    return types


def hallucinated_fields(obj: dict | None, raw_message: str) -> tuple[int, int]:
    """(hallucinated_count, total_count) of identifier values absent from source.

    Compares case-insensitively — CSV partners lowercase some fields. Only counts
    fields we can ground against the source; a model that omits products simply
    has fewer values checked (that surfaces as an exact-match miss instead).
    """
    if not obj or not isinstance(obj.get("products"), list):
        return (0, 0)
    src = raw_message.lower()
    hallucinated = 0
    total = 0
    for prod in obj["products"]:
        if not isinstance(prod, dict):
            continue
        for field in _HALLUCINATION_FIELDS:
            val = prod.get(field)
            if not val or not isinstance(val, str):
                continue
            total += 1
            if val.lower() not in src:
                hallucinated += 1
    return (hallucinated, total)


def _header_exact_match(pred: dict | None, target: dict) -> bool:
    """True iff all header (non-product, non-exception) fields match the oracle."""
    if not pred:
        return False
    return all(pred.get(f) == target.get(f) for f in _HEADER_FIELDS)


def score_sample(pred_obj: dict | None, target_obj: dict, raw_message: str) -> dict:
    """Score one model output against the oracle target. Returns a per-sample dict."""
    hall_count, hall_total = hallucinated_fields(pred_obj, raw_message)
    return {
        "parse_ok": pred_obj is not None,
        "schema_valid": is_schema_valid(pred_obj),
        "pred_exceptions": sorted(exception_labels(pred_obj)),
        "gold_exceptions": sorted(exception_labels(target_obj)),
        "header_exact_match": _header_exact_match(pred_obj, target_obj),
        "hallucinated": hall_count,
        "hallucination_total": hall_total,
    }


def _f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-sample scores into the headline metrics for one model.

    Exception F1 is multi-label: each sample can carry several exception types,
    so tp/fp/fn are counted per class over the predicted vs. gold *sets*.
    """
    n = len(results)
    if n == 0:
        return {}

    # Per-class multi-label confusion counts.
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in EXCEPTION_CLASSES}
    for r in results:
        pred = set(r["pred_exceptions"])
        gold = set(r["gold_exceptions"])
        for c in EXCEPTION_CLASSES:
            if c in pred and c in gold:
                per_class[c]["tp"] += 1
            elif c in pred and c not in gold:
                per_class[c]["fp"] += 1
            elif c not in pred and c in gold:
                per_class[c]["fn"] += 1

    per_class_f1 = {c: _f1(**counts) for c, counts in per_class.items()}
    # Macro-F1 over classes that actually appear in the gold labels (fair when a
    # split excludes a class, e.g. RECALLED_LOT held out of test_iid).
    present = [c for c in EXCEPTION_CLASSES if (per_class[c]["tp"] + per_class[c]["fn"]) > 0]
    macro_f1 = sum(per_class_f1[c]["f1"] for c in present) / len(present) if present else 0.0
    micro = _f1(
        tp=sum(per_class[c]["tp"] for c in EXCEPTION_CLASSES),
        fp=sum(per_class[c]["fp"] for c in EXCEPTION_CLASSES),
        fn=sum(per_class[c]["fn"] for c in EXCEPTION_CLASSES),
    )

    hall_total = sum(r["hallucination_total"] for r in results)
    hall_count = sum(r["hallucinated"] for r in results)

    return {
        "n": n,
        "schema_valid_rate": sum(r["schema_valid"] for r in results) / n,
        "parse_ok_rate": sum(r["parse_ok"] for r in results) / n,
        "header_exact_match_rate": sum(r["header_exact_match"] for r in results) / n,
        "exception_macro_f1": macro_f1,
        "exception_micro_f1": micro["f1"],
        "exception_micro": micro,
        "exception_per_class_f1": {c: per_class_f1[c]["f1"] for c in EXCEPTION_CLASSES},
        "exception_per_class": per_class_f1,
        "hallucination_rate": hall_count / hall_total if hall_total else 0.0,
        "hallucinated_values": hall_count,
        "hallucination_denominator": hall_total,
    }
