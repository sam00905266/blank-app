import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.tokenizer import tokenize

DATA_PATH = Path("data/rules.json")


def _ensure_dir():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_rules() -> list:
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def save_rules(rules: list) -> None:
    _ensure_dir()
    tmp_path = DATA_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "entries": rules}, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DATA_PATH)


def add_rule(question: str, answer: str, source: str = "claude") -> dict:
    rules = load_rules()
    entry = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "tokens": tokenize(question),
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    rules.append(entry)
    save_rules(rules)
    return entry


def delete_rule(rule_id: str) -> bool:
    rules = load_rules()
    new_rules = [r for r in rules if r["id"] != rule_id]
    if len(new_rules) == len(rules):
        return False
    save_rules(new_rules)
    return True


def all_rules() -> list:
    return load_rules()
