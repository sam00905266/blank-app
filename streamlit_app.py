import uuid
import streamlit as st

from core import (
    bm25_engine,
    memory_store,
    rule_store,
    validator,
    logger,
    auto_learner,
    pre_generator,
    claude_client,
    tokenizer,
)

# ── Session state init ────────────────────────────────────────────────────────

def _init_session():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        logger.log_event(st.session_state["session_id"], "session_start", {})
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "api_messages" not in st.session_state:
        st.session_state["api_messages"] = []
    if "client" not in st.session_state:
        st.session_state["client"] = None
    if "admin_mode" not in st.session_state:
        st.session_state["admin_mode"] = False
    if "bm25_threshold" not in st.session_state:
        st.session_state["bm25_threshold"] = bm25_engine.HIGH_CONFIDENCE_THRESHOLD
    memory_store.init_memory(st.session_state)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.title("⚙️ 設定")

        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            key="api_key_input",
        )
        if api_key and st.session_state["client"] is None:
            try:
                st.session_state["client"] = claude_client.create_client(api_key)
                st.success("✅ API Key 已設定")
            except ValueError as e:
                st.error(str(e))
        elif st.session_state["client"] is not None:
            st.success("✅ API Key 已設定")

        st.divider()

        st.session_state["admin_mode"] = st.checkbox(
            "🔧 管理員模式", value=st.session_state["admin_mode"]
        )

        if st.session_state["admin_mode"]:
            st.subheader("管理員面板")

            st.session_state["bm25_threshold"] = st.slider(
                "BM25 信心門檻",
                min_value=0.5,
                max_value=10.0,
                value=st.session_state["bm25_threshold"],
                step=0.1,
                help="越高表示更嚴格，越多問題走 Claude",
            )

            st.subheader("預生成規則")
            topics_input = st.text_area(
                "輸入主題（每行一個）",
                placeholder="Python 程式語言\n機器學習\n台灣歷史",
            )
            if st.button("🚀 生成規則", disabled=st.session_state["client"] is None):
                topics = [t for t in topics_input.splitlines() if t.strip()]
                if topics:
                    with st.spinner("正在生成規則..."):
                        saved = pre_generator.generate_rules_from_topics(
                            st.session_state["client"],
                            topics,
                            st.session_state["session_id"],
                        )
                    st.success(f"✅ 已生成 {len(saved)} 條規則")
                else:
                    st.warning("請輸入至少一個主題")

            with st.expander(f"📚 查看規則庫 ({len(rule_store.all_rules())} 條)"):
                rules = rule_store.all_rules()
                if rules:
                    for r in rules:
                        st.markdown(
                            f"**Q:** {r['question']}  \n"
                            f"**A:** {r['answer'][:80]}{'...' if len(r['answer']) > 80 else ''}  \n"
                            f"<small>來源: {r['source']} | {r['created_at'][:10]}</small>",
                            unsafe_allow_html=True,
                        )
                        if st.button("🗑️ 刪除", key=f"del_{r['id']}"):
                            rule_store.delete_rule(r["id"])
                            st.rerun()
                else:
                    st.info("規則庫是空的")

        st.divider()

        mem_count = len(memory_store.get_all(st.session_state))
        rule_count = len(rule_store.all_rules())
        st.caption(f"記憶條目：{mem_count}　｜　規則條目：{rule_count}")

        if st.button("🧹 清除 Session 記憶"):
            memory_store.clear_memory(st.session_state)
            st.success("記憶已清除")
            st.rerun()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _run_pipeline(query: str) -> tuple:
    session_id = st.session_state["session_id"]
    threshold = st.session_state["bm25_threshold"]

    if len(query.strip()) < 2:
        return None, "too_short"

    rules = rule_store.all_rules()
    memories = memory_store.get_all(st.session_state)
    all_entries = rules + memories

    query_tokens = tokenizer.tokenize(query)
    index, entries = bm25_engine.build_index(all_entries)
    results = bm25_engine.search(index, entries, query_tokens, top_k=5)

    top_score = results[0]["score"] if results else 0.0

    logger.log_event(session_id, "bm25_search", {
        "query": query,
        "top_score": top_score,
        "result_count": len(results),
        "top_ids": [r["entry"]["id"] for r in results[:3]],
    })

    if top_score >= threshold:
        for result in results:
            entry = result["entry"]
            val_result = validator.validate(query, entry)
            event = "validation_pass" if val_result.passed else "validation_fail"
            logger.log_event(session_id, event, {
                "candidate_id": entry["id"],
                "confidence": val_result.confidence,
                "failed_checks": val_result.failed_checks,
                "details": val_result.details,
            })
            if val_result.passed:
                return entry["answer"], "cache"

    return None, "need_claude"


def _call_claude(query: str) -> str:
    client = st.session_state["client"]
    session_id = st.session_state["session_id"]
    api_messages = st.session_state["api_messages"] + [
        {"role": "user", "content": query}
    ]
    system = "你是一個有幫助的助理，請用繁體中文回答。"
    answer = claude_client.chat(client, api_messages, system_prompt=system)
    logger.log_event(session_id, "claude_call", {
        "query": query,
        "answer_preview": answer[:100],
    })
    return answer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="智能對話助理", page_icon="🤖", layout="wide")
    _init_session()
    _render_sidebar()

    st.title("🤖 智能對話助理")
    st.caption("分層 AI 架構：BM25 快取 → 多重驗證 → Claude 兜底 → 自動學習")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "source" in msg:
                labels = {"cache": "⚡ 快取", "claude": "🧠 Claude",
                          "too_short": "⚠️ 查詢太短", "error": "❌ 錯誤"}
                st.caption(labels.get(msg["source"], msg["source"]))

    if prompt := st.chat_input("輸入您的問題..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer, source = _run_pipeline(prompt)

                if source == "too_short":
                    answer = "請輸入較完整的問題（至少 2 個字）。"
                elif source == "need_claude":
                    if st.session_state["client"] is None:
                        answer = "⚠️ 請先在側欄輸入 Anthropic API Key。"
                        source = "error"
                    else:
                        answer = _call_claude(prompt)
                        auto_learner.learn(
                            st.session_state["client"],
                            st.session_state,
                            prompt,
                            answer,
                            st.session_state["session_id"],
                        )
                        st.session_state["api_messages"].append(
                            {"role": "user", "content": prompt}
                        )
                        st.session_state["api_messages"].append(
                            {"role": "assistant", "content": answer}
                        )

            st.markdown(answer)
            labels = {"cache": "⚡ 快取", "claude": "🧠 Claude",
                      "too_short": "⚠️ 查詢太短", "error": "❌ 錯誤"}
            st.caption(labels.get(source, source))

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "source": source,
        })


if __name__ == "__main__":
    main()
