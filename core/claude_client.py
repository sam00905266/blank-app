import anthropic

MODEL = "claude-sonnet-4-6"


def create_client(api_key: str) -> anthropic.Anthropic:
    if not api_key or not api_key.strip():
        raise ValueError("API key 不能為空")
    return anthropic.Anthropic(api_key=api_key.strip())


def chat(client: anthropic.Anthropic, messages: list, system_prompt: str = "") -> str:
    kwargs = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    return response.content[0].text


def classify_answer(client: anthropic.Anthropic, question: str, answer: str) -> str:
    prompt = (
        f"以下是一組問題和答案。請判斷這個答案是「通用知識」（任何人問都適用）"
        f"還是「個人/情境資訊」（只對特定使用者或情境有效）。\n\n"
        f"問題：{question}\n答案：{answer}\n\n"
        f"只回覆一個英文單字：rule（通用知識）或 memory（個人/情境資訊）。"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.content[0].text.strip().lower()
    if result not in ("rule", "memory"):
        return "memory"
    return result
