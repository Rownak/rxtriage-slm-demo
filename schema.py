"""Canonical JSON schema (v1) for RxTriage — pydantic models.

This is a frozen contract (see CLAUDE.md Hard Rule #2). Any change here must be
mirrored in schema.json, every generator/renderer, and the eval harness.

Field definitions and the exception taxonomy are copied from project_summary.md.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MessageFormat(str, Enum):
    EPCIS_XML = "EPCIS_XML"
    X12_856 = "X12_856"
    CSV = "CSV"


class TransactionType(str, Enum):
    COMMISSIONING = "commissioning"
    SHIPPING = "shipping"
    RECEIVING = "receiving"
    AGGREGATION = "aggregation"


class PackagingLevel(str, Enum):
    UNIT = "unit"
    CASE = "case"
    PALLET = "pallet"


class ExceptionType(str, Enum):
    QTY_MISMATCH = "QTY_MISMATCH"
    MISSING_T3 = "MISSING_T3"
    EXPIRED_LOT = "EXPIRED_LOT"
    BROKEN_AGGREGATION = "BROKEN_AGGREGATION"
    DUPLICATE_SERIAL = "DUPLICATE_SERIAL"
    INVALID_NDC = "INVALID_NDC"
    RECALLED_LOT = "RECALLED_LOT"
    NONE = "NONE"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class Product(BaseModel):
    ndc: str
    gtin: str
    product_name: str
    lot_number: str
    expiration_date: str  # YYYY-MM-DD
    serial_numbers: list[str]
    quantity: int
    packaging_level: PackagingLevel
    parent_serial: str | None = None


class Dscsa(BaseModel):
    t1_present: bool
    t2_present: bool
    t3_present: bool


class Exception_(BaseModel):
    """Named with a trailing underscore to avoid shadowing the builtin `Exception`."""

    type: ExceptionType
    severity: Severity
    rationale: str
    recommended_resolution: str


class CanonicalTransaction(BaseModel):
    message_id: str
    message_format: MessageFormat
    transaction_type: TransactionType
    sender_gln: str
    receiver_gln: str
    event_time: str  # ISO-8601
    products: list[Product]
    dscsa: Dscsa
    exceptions: list[Exception_]
    confidence: float = Field(ge=0.0, le=1.0)
