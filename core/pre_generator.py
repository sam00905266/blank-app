from core import claude_client, rule_store, logger


def generate_rules_from_topics(client, topics: list, session_id: str) -> list:
    saved = []
    for topic in topics:
        topic = topic.strip()
        if not topic:
            continue
        prompt = (
            f"請針對以下主題，提供一個常見問題和對應的清晰答案。\n"
            f"主題：{topic}\n\n"
            f"格式：\nQ: （問題）\nA: （答案）"
        )
        try:
            response = claude_client.chat(client, [{"role": "user", "content": prompt}])
            question, answer = _parse_qa(response, topic)
            entry = rule_store.add_rule(question, answer, source="pre_generated")
            saved.append(entry)
            logger.log_event(session_id, "pre_generate", {
                "topic": topic,
                "question": question,
                "answer_preview": answer[:100],
            })
        except Exception as e:
            logger.log_event(session_id, "pre_generate", {
                "topic": topic,
                "error": str(e),
            })
    return saved


def _parse_qa(text: str, fallback_topic: str) -> tuple:
    lines = text.strip().splitlines()
    question = ""
    answer_lines = []
    in_answer = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Q:") or stripped.startswith("Q："):
            question = stripped[2:].strip()
        elif stripped.startswith("A:") or stripped.startswith("A："):
            answer_lines.append(stripped[2:].strip())
            in_answer = True
        elif in_answer:
            answer_lines.append(stripped)

    if not question:
        question = f"{fallback_topic}是什麼？"
    answer = "\n".join(answer_lines).strip() or text.strip()
    return question, answer
