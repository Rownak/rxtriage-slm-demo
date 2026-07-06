import json

import jsonschema
import pytest

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


@pytest.fixture
def json_schema():
    with open("schema.json") as f:
        return json.load(f)


def make_sample() -> CanonicalTransaction:
    return CanonicalTransaction(
        message_id="MSG-001",
        message_format=MessageFormat.EPCIS_XML,
        transaction_type=TransactionType.SHIPPING,
        sender_gln="0614141000012",
        receiver_gln="0614141000029",
        event_time="2026-01-15T10:30:00Z",
        products=[
            Product(
                ndc="0002-1433-80",
                gtin="00300020114113",
                product_name="Amoxicillin 500mg",
                lot_number="LOT12345",
                expiration_date="2027-06-30",
                serial_numbers=["SN0001", "SN0002"],
                quantity=2,
                packaging_level=PackagingLevel.CASE,
                parent_serial=None,
            )
        ],
        dscsa=Dscsa(t1_present=True, t2_present=True, t3_present=True),
        exceptions=[
            Exception_(
                type=ExceptionType.NONE,
                severity=Severity.MINOR,
                rationale="No violated rules detected.",
                recommended_resolution="No action needed.",
            )
        ],
        confidence=0.98,
    )


def test_pydantic_round_trip_validates_against_json_schema(json_schema):
    sample = make_sample()
    payload = json.loads(sample.model_dump_json())
    jsonschema.validate(instance=payload, schema=json_schema)


def test_confidence_bounds_enforced():
    sample = make_sample()
    payload = json.loads(sample.model_dump_json())
    payload["confidence"] = 1.5
    with pytest.raises(ValueError):
        CanonicalTransaction.model_validate(payload)
