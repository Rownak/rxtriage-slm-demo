# EPCIS template → canonical schema field mapping

Templates in this directory are GS1 EPCIS 2.0 example event XML files, used as
structural skeletons for the synthetic generator (Phase 2). This doc maps each
EPCIS XML element to the corresponding field in the canonical schema
(`schema.py` / `schema.json` at repo root).

## Templates

| File | EPCIS event type | Used as structural template for |
|---|---|---|
| `object_event_all_possible_fields.xml` | `ObjectEvent` | commissioning / shipping / receiving (single-EPC observation) |
| `Example_9.6.1-ObjectEvent-2020_06_18a.xml` | `ObjectEvent` (paired shipping→receiving) | shipping/receiving pair with PO + DESADV `bizTransaction` refs |
| `Mimasu-Example2-shipping-receiving.xml` | `ObjectEvent` | receiving event with explicit lot/GTIN quantity semantics |
| `aggregation_event_all_possible_fields.xml` | `AggregationEvent` | aggregation (parent/child serial hierarchy: unit→case→pallet) |
| `transaction_event_all_possible_fields.xml` | `TransactionEvent` | ASN / transaction-statement carrier (maps to DSCSA T1/T2/T3 concept) |

`_all_possible_fields.xml` templates are GS1's own field-coverage examples —
kept because they show every optional element (sensor data, extensions,
persistent disposition) even though our generator will only populate the
subset listed below. Fields not mapped are simply dropped by the generator;
they exist in real EPCIS traffic but have no home in our canonical schema.

## Field mapping

### Message-level (`CanonicalTransaction`)

| Canonical field | EPCIS XML source | Notes |
|---|---|---|
| `message_id` | *(generated)* | Not a native EPCIS field; the generator assigns a UUID/sequential ID per bundle. |
| `message_format` | *(fixed: `EPCIS_XML`)* | Set by the generator based on which renderer produced the message. |
| `transaction_type` | Derived from event type + `bizStep` | `ObjectEvent` + `bizstep:commissioning` → `commissioning`; `+ bizstep:shipping` → `shipping`; `+ bizstep:receiving` → `receiving`; `AggregationEvent` → `aggregation`. |
| `sender_gln` | `sourceList/source[type=...:possessing_party or owning_party]` | The GLN embedded in the `urn:epc:id:sgln:...` or `urn:epc:id:pgln:...` URI. |
| `receiver_gln` | `destinationList/destination[type=...:possessing_party or owning_party]` | Same URI parsing as sender. |
| `event_time` | `eventTime` (+ `eventTimeZoneOffset`) | Already ISO-8601; passed through directly. |

### Per-product (`Product`, from `quantityList`/`epcList`/`childEPCs`)

| Canonical field | EPCIS XML source | Notes |
|---|---|---|
| `ndc` | *(joined from `data/ref/ndc.parquet`)* | Not present in EPCIS; the generator picks a real NDC and encodes it into the synthetic GTIN. |
| `gtin` | `epcClass` (e.g. `urn:epc:class:lgtin:4012345.012345.998877`) or parsed from `epc` (`urn:epc:id:sgtin:...`) | GS1 SGTIN/LGTIN URI — company prefix + item ref decode to a GTIN-14. |
| `product_name` | *(joined from `data/ref/ndc.parquet`)* | EPCIS doesn't carry product names; looked up via the NDC. |
| `lot_number` | Lot segment of the `lgtin`/`sgtin` URI (last URI segment) | e.g. `998877` in `urn:epc:class:lgtin:4012345.012345.998877`. |
| `expiration_date` | *(generated)* | Not in EPCIS; the generator assigns a plausible expiry relative to `event_time`. |
| `serial_numbers` | `epcList/epc` (ObjectEvent) or `childEPCs/epc` (AggregationEvent) | Serial segment of each `sgtin` URI. |
| `quantity` | `quantityList/quantityElement/quantity` (or `childQuantityList` for aggregation) | Direct passthrough; `uom` is not modeled in canonical schema (dropped). |
| `packaging_level` | Inferred from EPC URI type / aggregation depth | `sgtin` (item-level) → `unit`; SSCC parent in `AggregationEvent.parentID` → `case` or `pallet` depending on nesting depth. |
| `parent_serial` | `AggregationEvent.parentID` | Serial segment of the `sscc` URI; `null` for un-aggregated units. |

### DSCSA block (`Dscsa`)

| Canonical field | EPCIS XML source | Notes |
|---|---|---|
| `t1_present` | *(generator-tracked)* | T1 = prior transaction history; not a discrete EPCIS field — the generator tracks whether an upstream commissioning event exists in the bundle. |
| `t2_present` | `bizTransactionList/bizTransaction[type=...:po]` presence | T2 = transaction information (PO reference). |
| `t3_present` | `bizTransactionList/bizTransaction[type=...:pedigree]` presence | T3 = transaction statement; this is the field the `MISSING_T3` defect injector removes. |

### `exceptions` / `confidence`

Not derived from EPCIS at all — these are populated by the deterministic rule
engine (`rules/engine.py`, Phase 2) after the bundle is assembled, per
CLAUDE.md's ground-truth rule (rule engine only, never an LLM).

## Known gaps (acceptable for this demo)

- `bizStep`/`disposition` CBV URIs (e.g. `urn:epcglobal:cbv:bizstep:shipping`)
  are matched by suffix string, not validated against the full CBV vocabulary
  — sufficient since the generator only ever emits the handful of `bizStep`
  values it needs.
- Sensor data, persistent disposition, and vendor extensions (`ext1:`/`ext2:`/
  `ext3:` namespaces) in the `_all_possible_fields.xml` templates are present
  in the raw XML for structural realism but are never read by the generator —
  the canonical schema has no field for them.
