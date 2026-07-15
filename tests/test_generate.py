import json
import random
import time

import jsonschema
import pandas as pd
import pytest

from datagen.generate import generate_clean_bundle, generate_corpus


@pytest.fixture
def ndc_df() -> pd.DataFrame:
    # Small fake NDC pool -- the real data/ref/ndc.parquet is gitignored and
    # not guaranteed to exist in a fresh checkout, so tests don't depend on it.
    return pd.DataFrame(
        [
            {"ndc": "0002-1433-80", "product_name": "Amoxicillin 500mg"},
            {"ndc": "0069-3150-83", "product_name": "Atorvastatin 20mg"},
            {"ndc": "50242-0079-01", "product_name": "Insulin Glargine"},
        ]
    )


@pytest.fixture
def json_schema():
    with open("schema.json") as f:
        return json.load(f)


def test_single_bundle_is_schema_valid(ndc_df, json_schema):
    rng = random.Random(42)
    bundle = generate_clean_bundle(rng, ndc_df)
    payload = json.loads(bundle.model_dump_json())
    jsonschema.validate(instance=payload, schema=json_schema)
    assert bundle.exceptions[0].type.value == "NONE"
    assert bundle.confidence == 1.0
    assert bundle.dscsa.t1_present and bundle.dscsa.t2_present and bundle.dscsa.t3_present


def test_aggregation_bundle_has_consistent_parent_child_links(ndc_df):
    # Force aggregation bundles until we see one, since transaction_type is random.
    rng = random.Random(1)
    bundle = None
    for _ in range(200):
        candidate = generate_clean_bundle(rng, ndc_df)
        if candidate.transaction_type.value == "aggregation":
            bundle = candidate
            break
    assert bundle is not None, "expected at least one aggregation bundle in 200 draws"

    units = [p for p in bundle.products if p.packaging_level.value == "unit"]
    cases = [p for p in bundle.products if p.packaging_level.value == "case"]
    assert len(cases) == 1
    case = cases[0]
    case_serial = case.serial_numbers[0]

    assert all(u.parent_serial == case_serial for u in units)
    assert case.quantity == len(units)
    assert case.parent_serial is None


def test_1000_bundles_generate_quickly_and_are_all_schema_valid(ndc_df, json_schema):
    start = time.time()
    bundles = generate_corpus(n=1000, seed=0, ndc_df=ndc_df)
    elapsed = time.time() - start

    assert elapsed < 300, f"1000 bundles took {elapsed:.1f}s, expected < 5 min"
    for bundle in bundles:
        jsonschema.validate(instance=json.loads(bundle.model_dump_json()), schema=json_schema)

    message_ids = {b.message_id for b in bundles}
    assert len(message_ids) == 1000  # all unique
