from rank_bm25 import BM25Okapi

HIGH_CONFIDENCE_THRESHOLD = 2.5


def build_index(entries: list):
    if not entries:
        return None, []
    corpus = [e["tokens"] for e in entries]
    index = BM25Okapi(corpus)
    return index, entries


def search(index, entries: list, query_tokens: list, top_k: int = 5) -> list:
    if index is None or not entries or not query_tokens:
        return []
    scores = index.get_scores(query_tokens)
    ranked = sorted(
        zip(scores, entries), key=lambda x: x[0], reverse=True
    )
    results = [
        {"entry": entry, "score": float(score)}
        for score, entry in ranked[:top_k]
        if score > 0
    ]
    return results


def get_top_score(index, entries: list, query_tokens: list) -> float:
    results = search(index, entries, query_tokens, top_k=1)
    if not results:
        return 0.0
    return results[0]["score"]
