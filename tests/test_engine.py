import random

import pandas as pd
import pytest

from datagen.generate import generate_clean_bundle, generate_corpus
from datagen.inject import INJECTABLE_TYPES, inject
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


def test_no_false_positives_on_1000_clean_bundles(ndc_df):
    """Property test: the engine never flags a defect on clean generator output."""
    bundles = generate_corpus(n=1000, seed=11, ndc_df=ndc_df)
    for bundle in bundles:
        detected = detect_exceptions(bundle, VALID_NDCS, RECALLED_LOTS)
        assert [e.type for e in detected] == [ExceptionType.NONE]
        assert detected == bundle.exceptions  # generator's NONE label matches too


def test_100_percent_agreement_on_1000_injected_samples(ndc_df):
    """tasks.md 2.4 acceptance: engine output == injector labels on 1k samples."""
    rng = random.Random(99)
    disagreements = 0
    for i in range(1000):
        exc_type = INJECTABLE_TYPES[i % len(INJECTABLE_TYPES)]
        bundle = generate_clean_bundle(rng, ndc_df)
        while (
            exc_type == ExceptionType.BROKEN_AGGREGATION
            and bundle.transaction_type.value != "aggregation"
        ):
            bundle = generate_clean_bundle(rng, ndc_df)

        injected = inject(exc_type, bundle, rng, VALID_NDCS, RECALLED_LOTS)
        detected = detect_exceptions(injected, VALID_NDCS, RECALLED_LOTS)
        if detected != injected.exceptions:
            disagreements += 1

    assert disagreements == 0


def test_multiple_defects_all_detected_in_taxonomy_order(ndc_df):
    """The engine reports every defect present, not just the injected one."""
    rng = random.Random(21)
    bundle = generate_clean_bundle(rng, ndc_df)
    stacked = inject(ExceptionType.MISSING_T3, bundle, rng, VALID_NDCS, RECALLED_LOTS)
    stacked = inject(ExceptionType.EXPIRED_LOT, stacked, rng, VALID_NDCS, RECALLED_LOTS)

    detected = detect_exceptions(stacked, VALID_NDCS, RECALLED_LOTS)
    assert [e.type for e in detected] == [ExceptionType.MISSING_T3, ExceptionType.EXPIRED_LOT]
