from core import claude_client, rule_store, memory_store, logger


def learn(client, session_state: dict, question: str, answer: str,
          session_id: str) -> str:
    try:
        classification = claude_client.classify_answer(client, question, answer)
    except Exception:
        classification = "memory"

    if classification == "rule":
        rule_store.add_rule(question, answer, source="claude")
    else:
        memory_store.add_memory(session_state, question, answer)

    logger.log_event(session_id, "auto_learn", {
        "classification": classification,
        "question": question,
        "answer_preview": answer[:100],
    })
    return classification
