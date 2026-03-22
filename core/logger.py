import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("data/logs/chat_log.jsonl")


def _ensure_dir():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_event(session_id: str, event_type: str, payload: dict) -> None:
    _ensure_dir()
    entry = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
