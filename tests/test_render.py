import random

import pandas as pd
import pytest

from datagen.generate import generate_clean_bundle
from datagen.render import (
    PARTNER_DIALECTS,
    parse_csv,
    parse_epcis_xml,
    parse_x12_856,
    render_csv,
    render_epcis_xml,
    render_x12_856,
)


@pytest.fixture
def ndc_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ndc": "0002-1433-80", "product_name": "Amoxicillin 500mg"},
            {"ndc": "0069-3150-83", "product_name": "Atorvastatin 20mg"},
        ]
    )


def _non_aggregation_bundle(ndc_df, seed=0):
    rng = random.Random(seed)
    for _ in range(200):
        bundle = generate_clean_bundle(rng, ndc_df)
        if bundle.transaction_type.value != "aggregation":
            return bundle
    raise AssertionError("expected a non-aggregation bundle in 200 draws")


def _aggregation_bundle(ndc_df, seed=1):
    rng = random.Random(seed)
    for _ in range(200):
        bundle = generate_clean_bundle(rng, ndc_df)
        if bundle.transaction_type.value == "aggregation":
            return bundle
    raise AssertionError("expected an aggregation bundle in 200 draws")


# ---------------------------------------------------------------------------
# EPCIS XML
# ---------------------------------------------------------------------------


def test_epcis_xml_round_trip_object_event(ndc_df):
    bundle = _non_aggregation_bundle(ndc_df)
    product = bundle.products[0]

    xml_text = render_epcis_xml(bundle)
    parsed = parse_epcis_xml(xml_text)

    assert parsed["event_time"] == bundle.event_time
    assert parsed["transaction_type"] == bundle.transaction_type.value
    assert parsed["ndc"] == product.ndc
    assert parsed["product_name"] == product.product_name
    assert parsed["lot_number"] == product.lot_number
    assert parsed["expiration_date"] == product.expiration_date
    assert parsed["quantity"] == product.quantity
    assert sorted(parsed["serial_numbers"]) == sorted(product.serial_numbers)
    assert parsed["t3_present"] == bundle.dscsa.t3_present


def test_epcis_xml_round_trip_aggregation_event(ndc_df):
    bundle = _aggregation_bundle(ndc_df)
    case = next(p for p in bundle.products if p.packaging_level.value == "case")
    units = [p for p in bundle.products if p.packaging_level.value == "unit"]

    xml_text = render_epcis_xml(bundle)
    parsed = parse_epcis_xml(xml_text)

    assert parsed["transaction_type"] == "aggregation"
    assert parsed["lot_number"] == case.lot_number
    assert parsed["quantity"] == case.quantity
    assert sorted(parsed["serial_numbers"]) == sorted(u.serial_numbers[0] for u in units)


def test_epcis_xml_parses_with_lxml(ndc_df):
    from lxml import etree

    bundle = _non_aggregation_bundle(ndc_df)
    xml_text = render_epcis_xml(bundle)
    etree.fromstring(xml_text.encode())  # raises on malformed XML


# ---------------------------------------------------------------------------
# X12-856
# ---------------------------------------------------------------------------


def test_x12_856_round_trip(ndc_df):
    bundle = _non_aggregation_bundle(ndc_df)
    product = bundle.products[0]

    text = render_x12_856(bundle)
    parsed = parse_x12_856(text)

    assert parsed["event_time"] == bundle.event_time
    assert parsed["transaction_type"] == bundle.transaction_type.value
    assert parsed["sender_gln"] == bundle.sender_gln
    assert parsed["receiver_gln"] == bundle.receiver_gln
    assert parsed["ndc"] == product.ndc
    assert parsed["gtin"] == product.gtin
    assert parsed["product_name"] == product.product_name
    assert parsed["quantity"] == product.quantity
    assert parsed["lot_number"] == product.lot_number
    assert parsed["expiration_date"] == product.expiration_date
    assert parsed["packaging_level"] == product.packaging_level.value
    assert sorted(parsed["serial_numbers"]) == sorted(product.serial_numbers)
    assert parsed["t3_present"] == bundle.dscsa.t3_present


def test_x12_856_round_trip_carries_parent_serial(ndc_df):
    bundle = _aggregation_bundle(ndc_df)
    unit = next(p for p in bundle.products if p.packaging_level.value == "unit")
    # Render a single-product view (renderer takes bundle.products[0]) --
    # construct a bundle whose only product is the unit, to check parent_serial wiring.
    bundle.products = [unit]

    text = render_x12_856(bundle)
    parsed = parse_x12_856(text)
    assert parsed["parent_serial"] == unit.parent_serial


# ---------------------------------------------------------------------------
# CSV (all 5 partner dialects)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", list(PARTNER_DIALECTS.keys()))
def test_csv_round_trip_all_dialects(ndc_df, dialect):
    bundle = _non_aggregation_bundle(ndc_df)
    product = bundle.products[0]

    text = render_csv(bundle, dialect=dialect)
    parsed = parse_csv(text, dialect=dialect)

    assert parsed["ndc"] == product.ndc
    assert parsed["gtin"] == product.gtin
    assert parsed["product_name"] == product.product_name
    assert parsed["lot_number"] == product.lot_number
    assert parsed["expiration_date"] == product.expiration_date
    assert parsed["quantity"] == product.quantity
    assert parsed["packaging_level"] == product.packaging_level.value
    assert sorted(parsed["serial_numbers"]) == sorted(product.serial_numbers)
    assert parsed["sender_gln"] == bundle.sender_gln
    assert parsed["receiver_gln"] == bundle.receiver_gln
    assert parsed["event_time"] == bundle.event_time
    assert parsed["transaction_type"] == bundle.transaction_type.value
    assert parsed["t3_present"] == bundle.dscsa.t3_present


def test_at_least_5_partner_dialects_defined():
    assert len(PARTNER_DIALECTS) >= 5
