"""Pull drug recall (enforcement) records from the openFDA API and extract
(product, lot number) pairs for use as RECALLED_LOT exception seeds.

The API's `code_info` field is unstructured free text (e.g.
"Lot #: 072915, Exp 10/29/2015" or "Lot# A; Lot# B, Exp 05/2023") — this
script regexes out lot codes following a "Lot" keyword. Records with no
parseable lot (e.g. "All lots within expiry") are skipped rather than
guessed at.

Usage:
    uv run python -m datagen.fetch_recalls
    uv run python -m datagen.fetch_recalls --target 300
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.fda.gov/drug/enforcement.json"
PAGE_SIZE = 100
OUT_PATH = Path("data/ref/recalled_lots.parquet")

# Matches "Lot", "Lot#", "Lot #:", "Lots:", "Lot Number(s):", "Lot code(s):",
# etc. (case-insensitive), skipping the optional "number"/"code" filler word,
# then captures an alphanumeric/hyphen code. Deliberately simple: this is a
# synthetic data seed, not a regulatory parser — a missed or malformed lot
# just means that record is skipped, not a downstream data-quality problem.
LOT_PATTERN = re.compile(
    r"lots?\s*(?:number|code)?s?\s*#?:?\s*([A-Za-z0-9][A-Za-z0-9\-@]{2,})",
    re.IGNORECASE,
)

# Reject tokens that matched the pattern but are just recall-status prose,
# not real lot codes (e.g. "Lot: All" or "Lots remaining within expiry").
NON_LOT_WORDS = {
    "all", "within", "no", "distributed", "codes", "code", "products", "product",
    "remaining", "known", "of", "lot", "lots", "number", "numbers", "repackaged",
    "labeled", "compounded", "received", "information",
}


def fetch_page(skip: int, api_key: str | None) -> list[dict]:
    params = {"limit": PAGE_SIZE, "skip": skip}
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


def extract_lots(code_info: str) -> list[str]:
    lots = []
    for match in LOT_PATTERN.finditer(code_info):
        token = match.group(1).strip().rstrip(",.;")
        if token.lower() in NON_LOT_WORDS:
            continue
        lots.append(token)
    return lots


def extract_rows(record: dict) -> list[dict]:
    code_info = record.get("code_info")
    product_description = record.get("product_description")
    recalling_firm = record.get("recalling_firm")
    recall_number = record.get("recall_number")
    classification = record.get("classification")

    if not all([code_info, product_description, recalling_firm, recall_number]):
        return []

    lots = extract_lots(code_info)
    return [
        {
            "recall_number": recall_number,
            "product_description": product_description,
            "recalling_firm": recalling_firm,
            "classification": classification,
            "lot_number": lot,
        }
        for lot in lots
    ]


def fetch_recalls(target: int) -> pd.DataFrame:
    api_key = os.environ.get("OPENFDA_API_KEY")
    rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    skip = 0

    while len(rows) < target:
        print(f"fetching skip={skip}...")
        page = fetch_page(skip, api_key)
        if not page:
            print("openFDA returned no more results.")
            break

        for record in page:
            for row in extract_rows(record):
                key = (row["recall_number"], row["lot_number"])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                rows.append(row)

        skip += PAGE_SIZE
        time.sleep(0.3)

    return pd.DataFrame(rows[:target] if len(rows) > target else rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=300, help="usable (product, lot) pairs to collect")
    args = parser.parse_args()

    df = fetch_recalls(args.target)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df)} (product, lot) pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
