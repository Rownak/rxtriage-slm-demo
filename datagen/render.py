"""Render a CanonicalTransaction into one of the three raw input formats.

Three renderers, one per `MessageFormat`:
  - render_epcis_xml : GS1 EPCIS 2.0 ObjectEvent/AggregationEvent XML
  - render_x12_856   : simplified flat X12-856 ASN segments (ST/BSN/HL/LIN/SN1/REF)
  - render_csv       : one row per product, in one of 5 partner column dialects

Field choices follow datagen/templates/MAPPING.md (the EPCIS <-> canonical
schema mapping doc from Phase 1). Only fields present in the canonical schema
are emitted -- these are synthetic messages generated FROM the ground truth,
not independent format implementations, so there's nothing to round-trip
except what the generator put in.

Each renderer has a matching minimal parser (parse_epcis_xml / parse_x12_856
/ parse_csv) used only by tests/test_render.py to prove the round-trip
property from tasks.md 2.2: parse(render(bundle)) reproduces the bundle's key
fields. The parsers are intentionally not full inverse-schema reconstructors
(that's the training-data-assembly job in a later phase) -- they just pull
back enough fields to verify the renderer didn't lose or corrupt information.
"""

from __future__ import annotations

import csv
import io

from lxml import etree

from schema import CanonicalTransaction, PackagingLevel

EPCIS_NS = "urn:epcglobal:epcis:xsd:2"
BIZSTEP_BY_TXN_TYPE = {
    "commissioning": "urn:epcglobal:cbv:bizstep:commissioning",
    "shipping": "urn:epcglobal:cbv:bizstep:shipping",
    "receiving": "urn:epcglobal:cbv:bizstep:receiving",
    "aggregation": "urn:epcglobal:cbv:bizstep:packing",
}
TXN_TYPE_BY_BIZSTEP = {v: k for k, v in BIZSTEP_BY_TXN_TYPE.items()}


def _sgtin(gtin: str, serial: str) -> str:
    # Company prefix = first 7 digits, item ref = remaining digits sans check digit.
    return f"urn:epc:id:sgtin:{gtin[:7]}.{gtin[7:-1]}.{serial}"


def _sgln(gln: str) -> str:
    return f"urn:epc:id:sgln:{gln[:7]}.{gln[7:-1]}.0"


def _sscc(serial: str, gln: str) -> str:
    return f"urn:epc:id:sscc:{gln[:7]}.{serial}"


# ---------------------------------------------------------------------------
# EPCIS XML
# ---------------------------------------------------------------------------


def render_epcis_xml(bundle: CanonicalTransaction) -> str:
    """Render as a minimal GS1 EPCIS 2.0 ObjectEvent or AggregationEvent."""
    nsmap = {"epcis": EPCIS_NS}
    root = etree.Element(f"{{{EPCIS_NS}}}EPCISDocument", nsmap=nsmap, schemaVersion="2.0")
    body = etree.SubElement(root, f"{{{EPCIS_NS}}}EPCISBody")
    event_list = etree.SubElement(body, f"{{{EPCIS_NS}}}EventList")

    is_aggregation = bundle.transaction_type.value == "aggregation"
    event = etree.SubElement(event_list, "AggregationEvent" if is_aggregation else "ObjectEvent")

    etree.SubElement(event, "eventTime").text = bundle.event_time
    etree.SubElement(event, "bizStep").text = BIZSTEP_BY_TXN_TYPE[bundle.transaction_type.value]

    biz_txn_list = etree.SubElement(event, "bizTransactionList")
    if bundle.dscsa.t2_present:
        etree.SubElement(biz_txn_list, "bizTransaction", type="urn:epcglobal:cbv:btt:po").text = (
            f"urn:epc:id:gdti:{bundle.sender_gln[:7]}.00001.1"
        )
    if bundle.dscsa.t3_present:
        etree.SubElement(
            biz_txn_list, "bizTransaction", type="urn:epcglobal:cbv:btt:pedigree"
        ).text = f"urn:epc:id:gsrn:{bundle.sender_gln[:7]}.000001"

    source_list = etree.SubElement(event, "sourceList")
    etree.SubElement(source_list, "source", type="urn:epcglobal:cbv:sdt:owning_party").text = (
        f"urn:epc:id:pgln:{bundle.sender_gln}"
    )
    dest_list = etree.SubElement(event, "destinationList")
    etree.SubElement(dest_list, "destination", type="urn:epcglobal:cbv:sdt:owning_party").text = (
        f"urn:epc:id:pgln:{bundle.receiver_gln}"
    )

    if is_aggregation:
        case = next(p for p in bundle.products if p.packaging_level == PackagingLevel.CASE)
        units = [p for p in bundle.products if p.packaging_level == PackagingLevel.UNIT]
        etree.SubElement(event, "parentID").text = _sscc(case.serial_numbers[0], bundle.sender_gln)
        child_epcs = etree.SubElement(event, "childEPCs")
        for unit in units:
            etree.SubElement(child_epcs, "epc").text = _sgtin(unit.gtin, unit.serial_numbers[0])
        # lot/expiry/ndc/product_name carried as a childQuantityList quantityElement,
        # per MAPPING.md's epcClass-based lot encoding.
        qty_list = etree.SubElement(event, "childQuantityList")
        qty_el = etree.SubElement(qty_list, "quantityElement")
        etree.SubElement(qty_el, "epcClass").text = (
            f"urn:epc:class:lgtin:{case.gtin[:7]}.{case.gtin[7:-1]}.{case.lot_number}"
        )
        etree.SubElement(qty_el, "quantity").text = str(case.quantity)
        etree.SubElement(qty_el, "ndc").text = case.ndc
        etree.SubElement(qty_el, "productName").text = case.product_name
        etree.SubElement(qty_el, "expirationDate").text = case.expiration_date
    else:
        product = bundle.products[0]
        epc_list = etree.SubElement(event, "epcList")
        for serial in product.serial_numbers:
            etree.SubElement(epc_list, "epc").text = _sgtin(product.gtin, serial)
        qty_list = etree.SubElement(event, "quantityList")
        qty_el = etree.SubElement(qty_list, "quantityElement")
        etree.SubElement(qty_el, "epcClass").text = (
            f"urn:epc:class:lgtin:{product.gtin[:7]}.{product.gtin[7:-1]}.{product.lot_number}"
        )
        etree.SubElement(qty_el, "quantity").text = str(product.quantity)
        etree.SubElement(qty_el, "ndc").text = product.ndc
        etree.SubElement(qty_el, "productName").text = product.product_name
        etree.SubElement(qty_el, "expirationDate").text = product.expiration_date

    return etree.tostring(root, pretty_print=True, encoding="unicode")


def parse_epcis_xml(xml_text: str) -> dict:
    """Pull back the fields render_epcis_xml wrote in, for round-trip testing."""
    root = etree.fromstring(xml_text.encode())
    event = root.find(".//ObjectEvent")
    is_aggregation = event is None
    if is_aggregation:
        event = root.find(".//AggregationEvent")

    biz_step = event.findtext("bizStep")
    txn_type = TXN_TYPE_BY_BIZSTEP[biz_step]
    t2_present = event.find('.//bizTransaction[@type="urn:epcglobal:cbv:btt:po"]') is not None
    t3_present = (
        event.find('.//bizTransaction[@type="urn:epcglobal:cbv:btt:pedigree"]') is not None
    )

    qty_el = event.find(".//quantityElement")
    result = {
        "event_time": event.findtext("eventTime"),
        "transaction_type": txn_type,
        "t2_present": t2_present,
        "t3_present": t3_present,
        "ndc": qty_el.findtext("ndc"),
        "product_name": qty_el.findtext("productName"),
        "lot_number": qty_el.findtext("epcClass").rsplit(".", 1)[-1],
        "expiration_date": qty_el.findtext("expirationDate"),
        "quantity": int(qty_el.findtext("quantity")),
    }

    if is_aggregation:
        result["serial_numbers"] = [
            epc.text.rsplit(".", 1)[-1] for epc in event.findall(".//childEPCs/epc")
        ]
    else:
        result["serial_numbers"] = [
            epc.text.rsplit(".", 1)[-1] for epc in event.findall(".//epcList/epc")
        ]

    return result


# ---------------------------------------------------------------------------
# X12-856 (simplified flat segments)
# ---------------------------------------------------------------------------


def render_x12_856(bundle: CanonicalTransaction) -> str:
    """Render as simplified X12-856 ASN segments: ST/BSN/HL/LIN/SN1/REF.

    Real X12-856 is far more elaborate (envelopes, loops, hundreds of
    qualifier codes); this keeps only the segments that carry fields our
    canonical schema actually models, per the Simplicity Rule.
    """
    product = bundle.products[0]
    lines = [
        f"ST*856*{bundle.message_id[:9]}",
        f"BSN*00*{bundle.message_id[:9]}*{bundle.event_time}*{bundle.transaction_type.value}",
        f"HL*1**S*{bundle.sender_gln}",
        f"HL*2**T*{bundle.receiver_gln}",
        f"LIN*1*NX*{product.ndc}*UP*{product.gtin}*{product.product_name}",
        f"SN1*1*{product.quantity}*EA",
        f"REF*LT*{product.lot_number}",
        f"REF*EX*{product.expiration_date}",
        f"REF*PK*{product.packaging_level.value}",
    ]
    for serial in product.serial_numbers:
        lines.append(f"REF*SN*{serial}")
    if product.parent_serial:
        lines.append(f"REF*PS*{product.parent_serial}")
    lines.append(f"REF*T3*{'Y' if bundle.dscsa.t3_present else 'N'}")
    lines.append("SE*" + str(len(lines) + 1))
    return "\n".join(lines)


def parse_x12_856(text: str) -> dict:
    """Pull back the fields render_x12_856 wrote in, for round-trip testing."""
    serials: list[str] = []
    result: dict = {"serial_numbers": serials}

    for line in text.splitlines():
        segs = line.split("*")
        tag = segs[0]

        if tag == "BSN":
            result["event_time"] = segs[3]
            result["transaction_type"] = segs[4]
        elif tag == "HL" and segs[3] == "S":
            result["sender_gln"] = segs[4]
        elif tag == "HL" and segs[3] == "T":
            result["receiver_gln"] = segs[4]
        elif tag == "LIN":
            result["ndc"] = segs[3]
            result["gtin"] = segs[5]
            result["product_name"] = segs[6]
        elif tag == "SN1":
            result["quantity"] = int(segs[2])
        elif tag == "REF" and segs[1] == "LT":
            result["lot_number"] = segs[2]
        elif tag == "REF" and segs[1] == "EX":
            result["expiration_date"] = segs[2]
        elif tag == "REF" and segs[1] == "PK":
            result["packaging_level"] = segs[2]
        elif tag == "REF" and segs[1] == "SN":
            serials.append(segs[2])
        elif tag == "REF" and segs[1] == "PS":
            result["parent_serial"] = segs[2]
        elif tag == "REF" and segs[1] == "T3":
            result["t3_present"] = segs[2] == "Y"

    return result


# ---------------------------------------------------------------------------
# CSV (5 partner dialects: different column names/order)
# ---------------------------------------------------------------------------

# Each dialect maps a canonical field name -> the partner's own column header.
# Order of the dict also fixes the column order in that partner's CSV.
PARTNER_DIALECTS: dict[str, dict[str, str]] = {
    "partner_a": {
        "ndc": "NDC",
        "gtin": "GTIN",
        "product_name": "ProductName",
        "lot_number": "LotNumber",
        "expiration_date": "ExpirationDate",
        "quantity": "Quantity",
        "packaging_level": "PackagingLevel",
        "serial_numbers": "SerialNumbers",
        "parent_serial": "ParentSerial",
        "sender_gln": "SenderGLN",
        "receiver_gln": "ReceiverGLN",
        "event_time": "EventTime",
        "transaction_type": "TransactionType",
        "t3_present": "T3Present",
    },
    "partner_b": {
        "ndc": "ndc_code",
        "gtin": "gtin14",
        "product_name": "item_desc",
        "lot_number": "lot",
        "expiration_date": "exp_date",
        "quantity": "qty",
        "packaging_level": "pkg_level",
        "serial_numbers": "serials",
        "parent_serial": "parent_sn",
        "sender_gln": "ship_from_gln",
        "receiver_gln": "ship_to_gln",
        "event_time": "txn_datetime",
        "transaction_type": "txn_type",
        "t3_present": "has_t3",
    },
    "partner_c": {
        # deliberately reordered + fewer columns to prove renderer handles
        # dialect variety, not just relabeling
        "transaction_type": "Type",
        "event_time": "Timestamp",
        "sender_gln": "FromGLN",
        "receiver_gln": "ToGLN",
        "ndc": "NDC",
        "lot_number": "Lot",
        "quantity": "Qty",
        "expiration_date": "Expiry",
        "gtin": "GTIN14",
        "product_name": "Description",
        "packaging_level": "Level",
        "serial_numbers": "Serials",
        "parent_serial": "ParentSN",
        "t3_present": "T3",
    },
    "partner_d": {
        "gtin": "GTIN",
        "ndc": "NationalDrugCode",
        "product_name": "Product",
        "lot_number": "Batch",
        "expiration_date": "BestBefore",
        "quantity": "Count",
        "packaging_level": "UnitType",
        "serial_numbers": "SGTINs",
        "parent_serial": "AggregateParent",
        "sender_gln": "Origin",
        "receiver_gln": "Destination",
        "event_time": "When",
        "transaction_type": "EventKind",
        "t3_present": "TransactionStatement",
    },
    "partner_e": {
        "ndc": "NDC11",
        "gtin": "GTIN",
        "product_name": "DrugName",
        "lot_number": "LOT#",
        "expiration_date": "EXPDATE",
        "quantity": "UNITS",
        "packaging_level": "PKGLVL",
        "serial_numbers": "SERIALNOS",
        "parent_serial": "PARENTID",
        "sender_gln": "SHIPPER_GLN",
        "receiver_gln": "CONSIGNEE_GLN",
        "event_time": "EVENT_TS",
        "transaction_type": "EVENT_TYPE",
        "t3_present": "DSCSA_T3",
    },
}

SERIAL_SEP = "|"


def render_csv(bundle: CanonicalTransaction, dialect: str = "partner_a") -> str:
    """Render as a single-row CSV using the given partner dialect's column mapping."""
    columns = PARTNER_DIALECTS[dialect]
    product = bundle.products[0]

    values = {
        "ndc": product.ndc,
        "gtin": product.gtin,
        "product_name": product.product_name,
        "lot_number": product.lot_number,
        "expiration_date": product.expiration_date,
        "quantity": str(product.quantity),
        "packaging_level": product.packaging_level.value,
        "serial_numbers": SERIAL_SEP.join(product.serial_numbers),
        "parent_serial": product.parent_serial or "",
        "sender_gln": bundle.sender_gln,
        "receiver_gln": bundle.receiver_gln,
        "event_time": bundle.event_time,
        "transaction_type": bundle.transaction_type.value,
        "t3_present": "Y" if bundle.dscsa.t3_present else "N",
    }

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns.values())
    writer.writerow([values[field] for field in columns.keys()])
    return buf.getvalue()


def parse_csv(text: str, dialect: str = "partner_a") -> dict:
    """Pull back the fields render_csv wrote in, for round-trip testing."""
    columns = PARTNER_DIALECTS[dialect]
    header_to_field = {header: field for field, header in columns.items()}

    reader = csv.DictReader(io.StringIO(text))
    row = next(reader)

    result = {header_to_field[header]: value for header, value in row.items()}
    result["quantity"] = int(result["quantity"])
    result["serial_numbers"] = result["serial_numbers"].split(SERIAL_SEP)
    result["parent_serial"] = result["parent_serial"] or None
    result["t3_present"] = result["t3_present"] == "Y"
    return result
