from collections import Counter

import pandas as pd
import pytest

from datagen.assemble import HELDOUT_DIALECT, assemble_corpus, make_splits
from rules.engine import detect_exceptions
from schema import CanonicalTransaction

RECALLED_LOTS = {"072915", "9LK442", "A1B2C3"}
TOTAL = 170  # small but enough for every (type, format) combination to appear


@pytest.fixture
def ndc_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ndc": "0002-1433-80", "product_name": "Amoxicillin 500mg"},
            {"ndc": "0069-3150-83", "product_name": "Atorvastatin 20mg"},
            {"ndc": "50242-0079-01", "product_name": "Insulin Glargine"},
        ]
    )


@pytest.fixture(scope="module")
def corpus():
    ndc_df = pd.DataFrame(
        [
            {"ndc": "0002-1433-80", "product_name": "Amoxicillin 500mg"},
            {"ndc": "0069-3150-83", "product_name": "Atorvastatin 20mg"},
        ]
    )
    return assemble_corpus(TOTAL, seed=5, ndc_df=ndc_df, recalled_lots=RECALLED_LOTS), ndc_df


def _types(sample) -> list[str]:
    return [e["type"] for e in sample["target"]["exceptions"]]


def test_class_balance(corpus):
    samples, _ = corpus
    assert len(samples) == TOTAL
    counts = Counter(t for s in samples for t in _types(s))
    per_defect = (TOTAL - round(TOTAL * 0.3)) // 7
    defect_counts = {t: n for t, n in counts.items() if t != "NONE"}
    assert set(defect_counts.values()) == {per_defect}
    assert counts["NONE"] == TOTAL - 7 * per_defect  # ~30% + rounding remainder


def test_format_mix_roughly_equal(corpus):
    samples, _ = corpus
    counts = Counter(s["format"] for s in samples)
    assert set(counts) == {"EPCIS_XML", "X12_856", "CSV"}
    assert max(counts.values()) - min(counts.values()) <= 2


def test_broken_aggregation_only_in_epcis(corpus):
    samples, _ = corpus
    for s in samples:
        if "BROKEN_AGGREGATION" in _types(s):
            assert s["format"] == "EPCIS_XML"


def test_x12_and_csv_targets_are_single_flat_product(corpus):
    """Non-EPCIS raw messages render products[0] only, so their targets must
    contain exactly one unit-level product — anything more would train the
    model to hallucinate products absent from the source."""
    samples, _ = corpus
    for s in samples:
        if s["format"] != "EPCIS_XML":
            products = s["target"]["products"]
            assert len(products) == 1
            assert products[0]["packaging_level"] == "unit"
            assert products[0]["parent_serial"] is None


def test_engine_agrees_with_every_label(corpus):
    samples, ndc_df = corpus
    valid_ndcs = set(ndc_df["ndc"])
    for s in samples:
        txn = CanonicalTransaction.model_validate(s["target"])
        detected = detect_exceptions(txn, valid_ndcs, RECALLED_LOTS)
        assert detected == txn.exceptions


def test_target_format_tag_matches_sample_format(corpus):
    samples, _ = corpus
    for s in samples:
        assert s["target"]["message_format"] == s["format"]
        assert (s["csv_dialect"] is not None) == (s["format"] == "CSV")


def test_splits_are_disjoint_and_complete(corpus):
    samples, _ = corpus
    splits = make_splits(samples, seed=5)
    all_ids = [s["id"] for rows in splits.values() for s in rows]
    assert len(all_ids) == TOTAL
    assert len(set(all_ids)) == TOTAL


def test_heldout_defect_never_in_other_splits(corpus):
    samples, _ = corpus
    splits = make_splits(samples, seed=5)
    for name, rows in splits.items():
        recalled = [s for s in rows if "RECALLED_LOT" in _types(s)]
        if name == "test_heldout_defect":
            assert len(recalled) == len(rows) > 0
        else:
            assert recalled == []


def test_heldout_dialect_never_in_train_val_iid(corpus):
    samples, _ = corpus
    splits = make_splits(samples, seed=5)
    for name in ("train", "val", "test_iid"):
        assert all(s["csv_dialect"] != HELDOUT_DIALECT for s in splits[name])
    assert all(s["csv_dialect"] == HELDOUT_DIALECT for s in splits["test_heldout_format"])
