import json

from datagen.format_chat import SYSTEM_PROMPT, build_chat


def _sample() -> dict:
    return {
        "id": "abc-123",
        "format": "X12_856",
        "csv_dialect": None,
        "raw_message": "ST*856*abc\nBSN*00*abc*2026-01-01T00:00:00Z*shipping",
        "target": {"message_id": "abc-123", "confidence": 1.0},
    }


def test_chat_record_structure():
    rec = build_chat(_sample())
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == SYSTEM_PROMPT


def test_user_message_carries_format_tag_and_raw_text():
    rec = build_chat(_sample())
    user = rec["messages"][1]["content"]
    assert user.startswith("Format: X12_856\n\n")
    assert "ST*856*abc" in user


def test_assistant_message_is_compact_parseable_target_json():
    sample = _sample()
    rec = build_chat(sample)
    assistant = rec["messages"][2]["content"]
    assert json.loads(assistant) == sample["target"]
    assert "\n" not in assistant and ": " not in assistant  # compact separators
