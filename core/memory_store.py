import uuid
from datetime import datetime, timezone

from core.tokenizer import tokenize


def init_memory(session_state) -> None:
    if "memory_entries" not in session_state:
        session_state["memory_entries"] = []


def add_memory(session_state, question: str, answer: str) -> dict:
    entry = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "tokens": tokenize(question),
        "source": "memory",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    session_state["memory_entries"].append(entry)
    return entry


def get_all(session_state) -> list:
    return session_state.get("memory_entries", [])


def clear_memory(session_state) -> None:
    session_state["memory_entries"] = []
