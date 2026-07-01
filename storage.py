import json
import os
from typing import Any

SUBMISSIONS_FILE = "submissions.json"
AUDIT_LOG_FILE = "audit_log.json"


def _load_json_file(filename: str, default: Any) -> Any:
    if not os.path.exists(filename):
        return default

    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read().strip()

            if not content:
                return default

            return json.loads(content)

    except (json.JSONDecodeError, OSError):
        return default


def _save_json_file(filename: str, data: Any) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_submission(submission: dict) -> None:
    submissions = _load_json_file(SUBMISSIONS_FILE, {})
    submissions[submission["content_id"]] = submission
    _save_json_file(SUBMISSIONS_FILE, submissions)


def get_submission(content_id: str) -> dict | None:
    submissions = _load_json_file(SUBMISSIONS_FILE, {})
    return submissions.get(content_id)


def update_submission(content_id: str, updates: dict) -> dict | None:
    submissions = _load_json_file(SUBMISSIONS_FILE, {})

    if content_id not in submissions:
        return None

    submissions[content_id].update(updates)
    _save_json_file(SUBMISSIONS_FILE, submissions)

    return submissions[content_id]


def add_audit_entry(entry: dict) -> None:
    entries = _load_json_file(AUDIT_LOG_FILE, [])
    entries.append(entry)
    _save_json_file(AUDIT_LOG_FILE, entries)


def get_audit_entries(limit: int = 50) -> list[dict]:
    entries = _load_json_file(AUDIT_LOG_FILE, [])
    return entries[-limit:]