"""Pull NDC product records from the openFDA NDC Directory API.

Fetches product_ndc, product name, labeler, dosage form, and package
description for ~5k finished human drug products, paging through the API
(max 1000 records/request) and caching the result to data/ref/ndc.parquet.

Usage:
    uv run python -m datagen.fetch_ndc
    uv run python -m datagen.fetch_ndc --target 5000

An OPENFDA_API_KEY env var is optional — openFDA works unauthenticated at a
lower rate limit (240 req/min, 1000 req/day), which is plenty for a one-time
~5k-record pull (5 requests). Passing a key just raises the ceiling.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.fda.gov/drug/ndc.json"
PAGE_SIZE = 1000  # openFDA max per request
OUT_PATH = Path("data/ref/ndc.parquet")

# Only finished human drug products have a stable, real-world dosage form /
# package description — this keeps the reference pool relevant to the
# commissioning/shipping scenarios we're simulating.
QUERY = 'finished:true AND product_type:"HUMAN PRESCRIPTION DRUG"'


def fetch_page(skip: int, api_key: str | None) -> list[dict]:
    params = {"search": QUERY, "limit": PAGE_SIZE, "skip": skip}
    if api_key:
        params["api_key"] = api_key

    for attempt in range(5):
        resp = requests.get(API_URL, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 2**attempt
            print(f"  rate limited, backing off {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json().get("results", [])
    raise RuntimeError("openFDA rate limit persisted after 5 retries")


def extract_row(record: dict) -> dict | None:
    ndc = record.get("product_ndc")
    product_name = record.get("brand_name") or record.get("generic_name")
    labeler = record.get("labeler_name")
    dosage_form = record.get("dosage_form")
    packaging = record.get("packaging") or []
    package_description = packaging[0].get("description") if packaging else None

    if not all([ndc, product_name, labeler, dosage_form, package_description]):
        return None  # skip records missing a key field rather than fabricate one

    return {
        "ndc": ndc,
        "product_name": product_name,
        "labeler": labeler,
        "dosage_form": dosage_form,
        "package_description": package_description,
    }


def fetch_ndc(target: int) -> pd.DataFrame:
    api_key = os.environ.get("OPENFDA_API_KEY")
    rows: list[dict] = []
    seen_ndc: set[str] = set()
    skip = 0

    while len(rows) < target:
        print(f"fetching skip={skip}...")
        page = fetch_page(skip, api_key)
        if not page:
            print("openFDA returned no more results.")
            break

        for record in page:
            row = extract_row(record)
            if row is None or row["ndc"] in seen_ndc:
                continue
            seen_ndc.add(row["ndc"])
            rows.append(row)

        skip += PAGE_SIZE
        time.sleep(0.3)  # stay well under the unauthenticated rate limit

    return pd.DataFrame(rows[:target])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=5000, help="unique NDC records to collect")
    args = parser.parse_args()

    df = fetch_ndc(args.target)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df)} unique NDC records to {OUT_PATH}")


if __name__ == "__main__":
    main()
