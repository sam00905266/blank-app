import jieba

_initialized = False


def _ensure_initialized():
    global _initialized
    if not _initialized:
        jieba.initialize()
        _initialized = True


def tokenize(text: str) -> list:
    _ensure_initialized()
    tokens = jieba.lcut(text.strip())
    return [t.lower() for t in tokens if t.strip()]


def tokenize_batch(texts: list) -> list:
    return [tokenize(t) for t in texts]
