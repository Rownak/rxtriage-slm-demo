"""Convert corpus splits to chat-format training records + token-length audit.

Each corpus sample becomes one record:
    {"messages": [
        {"role": "system", ...fixed task+schema prompt...},
        {"role": "user", "content": "Format: <tag>\\n\\n<raw message>"},
        {"role": "assistant", "content": "<compact canonical JSON>"}
    ]}

written to data/corpus/<split>_chat.jsonl for every split (training only uses
train/val, but eval scripts read the same shape for the test splits).

The token audit tokenizes every record with the Qwen2.5-3B-Instruct chat
template (the Phase 3 training target) and writes an ASCII histogram to
data/corpus/token_lengths.md, confirming the >=99%-under-4k acceptance check.

Usage:
    uv run python -m datagen.format_chat            # convert + audit
    uv run python -m datagen.format_chat --no-audit  # convert only (offline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CORPUS_DIR = Path("data/corpus")
SPLITS = ["train", "val", "test_iid", "test_heldout_defect", "test_heldout_format"]
TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are RxTriage, a pharmaceutical supply chain transaction normalizer and exception triager.

Convert the raw transaction message (EPCIS XML, X12-856 ASN, or partner CSV) into exactly one JSON object with this schema:

{"message_id": str, "message_format": "EPCIS_XML"|"X12_856"|"CSV", "transaction_type": "commissioning"|"shipping"|"receiving"|"aggregation", "sender_gln": str, "receiver_gln": str, "event_time": ISO-8601 str, "products": [{"ndc": str, "gtin": str, "product_name": str, "lot_number": str, "expiration_date": "YYYY-MM-DD", "serial_numbers": [str], "quantity": int, "packaging_level": "unit"|"case"|"pallet", "parent_serial": str|null}], "dscsa": {"t1_present": bool, "t2_present": bool, "t3_present": bool}, "exceptions": [{"type": "QTY_MISMATCH"|"MISSING_T3"|"EXPIRED_LOT"|"BROKEN_AGGREGATION"|"DUPLICATE_SERIAL"|"INVALID_NDC"|"RECALLED_LOT"|"NONE", "severity": "critical"|"major"|"minor", "rationale": str, "recommended_resolution": str}], "confidence": float}

Rules: extract only fields present in the message — never invent values. Check for exceptions: quantity vs. serial count mismatch, missing DSCSA T3 transaction statement, movement after lot expiry, broken case/unit aggregation, duplicate serial numbers, NDC not in the FDA directory, recalled lots. If none apply, emit the single NONE exception. Each rationale must cite the violated rule in one sentence. Output only the JSON object."""


def build_chat(sample: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Format: {sample['format']}\n\n{sample['raw_message']}"},
            {
                "role": "assistant",
                "content": json.dumps(sample["target"], separators=(",", ":")),
            },
        ]
    }


def convert_split(split: str) -> list[dict]:
    in_path = CORPUS_DIR / f"{split}.jsonl"
    out_path = CORPUS_DIR / f"{split}_chat.jsonl"
    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            records.append(build_chat(json.loads(line)))
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


def audit_token_lengths(records_by_split: dict[str, list[dict]]) -> None:
    from transformers import AutoTokenizer  # heavy import — only when auditing

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    # Render the template to text, then encode — apply_chat_template's
    # tokenize=True return type changed across transformers versions.
    lengths = [
        len(tokenizer.encode(tokenizer.apply_chat_template(rec["messages"], tokenize=False)))
        for records in records_by_split.values()
        for rec in records
    ]
    lengths.sort()
    n = len(lengths)
    under = sum(1 for x in lengths if x <= MAX_TOKENS)
    pct_under = 100 * under / n

    # ASCII histogram, 256-token buckets — a plot dependency isn't worth it here.
    bucket = 256
    top = (max(lengths) // bucket + 1) * bucket
    lines = [
        "# Token-length audit (Qwen2.5-3B-Instruct chat template)",
        "",
        f"All splits combined: n={n}, min={lengths[0]}, "
        f"p50={lengths[n // 2]}, p95={lengths[int(n * 0.95)]}, max={lengths[-1]}.",
        f"**{pct_under:.2f}% fit under {MAX_TOKENS} tokens** (acceptance: >= 99%).",
        "",
        "```",
    ]
    for lo in range(0, top, bucket):
        count = sum(1 for x in lengths if lo <= x < lo + bucket)
        bar = "#" * round(60 * count / n) if count else ""
        lines.append(f"{lo:>5}-{lo + bucket - 1:<5} {count:>6} {bar}")
    lines += ["```", ""]

    (CORPUS_DIR / "token_lengths.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{pct_under:.2f}% of {n} records under {MAX_TOKENS} tokens "
          f"(p95={lengths[int(n * 0.95)]}, max={lengths[-1]})")
    if pct_under < 99:
        raise SystemExit("FAIL: <99% of records fit the 4k-token budget — trim or chunk needed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-audit", action="store_true", help="skip the tokenizer audit")
    args = parser.parse_args()

    records_by_split = {}
    for split in SPLITS:
        records_by_split[split] = convert_split(split)
        print(f"{split}_chat.jsonl: {len(records_by_split[split])} records")

    if not args.no_audit:
        audit_token_lengths(records_by_split)


if __name__ == "__main__":
    main()
