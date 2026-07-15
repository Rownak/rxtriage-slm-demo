import random

import pandas as pd
import pytest

from datagen.generate import generate_clean_bundle
from datagen.inject import INJECTABLE_TYPES, inject, inject_broken_aggregation
from rules.engine import detect_exceptions
from schema import ExceptionType

VALID_NDCS = {"0002-1433-80", "0069-3150-83", "50242-0079-01"}
RECALLED_LOTS = {"072915", "9LK442", "A1B2C3"}


@pytest.fixture
def ndc_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ndc": "0002-1433-80", "product_name": "Amoxicillin 500mg"},
            {"ndc": "0069-3150-83", "product_name": "Atorvastatin 20mg"},
            {"ndc": "50242-0079-01", "product_name": "Insulin Glargine"},
        ]
    )


def _bundle(ndc_df, rng, aggregation: bool | None = None):
    """Draw bundles until the aggregation-ness matches (None = don't care)."""
    for _ in range(500):
        bundle = generate_clean_bundle(rng, ndc_df)
        is_agg = bundle.transaction_type.value == "aggregation"
        if aggregation is None or is_agg == aggregation:
            return bundle
    raise AssertionError("could not draw a suitable bundle in 500 tries")


@pytest.mark.parametrize("exc_type", INJECTABLE_TYPES)
def test_injector_triggers_exactly_its_own_rule(ndc_df, exc_type):
    rng = random.Random(hash(exc_type.value) % 2**31)
    needs_aggregation = exc_type == ExceptionType.BROKEN_AGGREGATION
    bundle = _bundle(ndc_df, rng, aggregation=True if needs_aggregation else None)

    injected = inject(exc_type, bundle, rng, VALID_NDCS, RECALLED_LOTS)
    detected = detect_exceptions(injected, VALID_NDCS, RECALLED_LOTS)

    assert [e.type for e in detected] == [exc_type]
    assert detected == injected.exceptions  # full label agreement, not just the type


@pytest.mark.parametrize("exc_type", INJECTABLE_TYPES)
def test_injected_bundle_stays_schema_valid(ndc_df, exc_type):
    rng = random.Random(7)
    needs_aggregation = exc_type == ExceptionType.BROKEN_AGGREGATION
    bundle = _bundle(ndc_df, rng, aggregation=True if needs_aggregation else None)

    injected = inject(exc_type, bundle, rng, VALID_NDCS, RECALLED_LOTS)
    # pydantic re-validation is the schema check (schema.json derives from the model)
    type(injected).model_validate(injected.model_dump())


def test_injection_does_not_mutate_the_original(ndc_df):
    rng = random.Random(3)
    bundle = _bundle(ndc_df, rng)
    before = bundle.model_dump_json()
    inject(ExceptionType.QTY_MISMATCH, bundle, rng, VALID_NDCS, RECALLED_LOTS)
    assert bundle.model_dump_json() == before


def test_broken_aggregation_requires_aggregation_bundle(ndc_df):
    rng = random.Random(5)
    bundle = _bundle(ndc_df, rng, aggregation=False)
    with pytest.raises(ValueError):
        inject_broken_aggregation(bundle, rng)


def test_broken_aggregation_both_modes_detected(ndc_df):
    # Run enough seeds to exercise both the orphan and the removed-child modes.
    details_seen = set()
    for seed in range(20):
        rng = random.Random(seed)
        bundle = _bundle(ndc_df, rng, aggregation=True)
        injected = inject_broken_aggregation(bundle, rng)
        detected = detect_exceptions(injected, VALID_NDCS, RECALLED_LOTS)
        assert [e.type for e in detected] == [ExceptionType.BROKEN_AGGREGATION]
        assert detected == injected.exceptions
        details_seen.add("references parent" in detected[0].rationale)
    assert details_seen == {True, False}, "expected both orphan and count-mismatch modes"
