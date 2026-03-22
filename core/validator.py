from dataclasses import dataclass, field
from core.tokenizer import tokenize

NEGATION_WORDS = ["不", "沒", "非", "無", "否", "別", "莫", "未", "勿",
                  "不是", "不會", "不能", "沒有", "不行", "不對"]

TIME_WORDS = ["今天", "明天", "昨天", "現在", "最近", "這週", "這個月",
              "今年", "當前", "目前", "暫時", "臨時", "這陣子"]


@dataclass
class ValidationResult:
    passed: bool
    confidence: float
    failed_checks: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


MIN_PASSING_CHECKS = 3


def _negation_check(query: str, candidate_question: str) -> tuple:
    q_has_neg = any(w in query for w in NEGATION_WORDS)
    c_has_neg = any(w in candidate_question for w in NEGATION_WORDS)
    passed = q_has_neg == c_has_neg
    return passed, {"query_negation": q_has_neg, "candidate_negation": c_has_neg}


def _abstraction_check(query_tokens: list, candidate_tokens: list) -> tuple:
    q_set = set(query_tokens)
    c_set = set(candidate_tokens)
    union = q_set | c_set
    if not union:
        return True, {"jaccard": 1.0}
    jaccard = len(q_set & c_set) / len(union)
    passed = jaccard >= 0.15
    return passed, {"jaccard": round(jaccard, 3)}


def _subject_check(query_tokens: list, candidate_tokens: list) -> tuple:
    def first_noun(tokens):
        for t in tokens:
            if len(t) >= 2:
                return t
        return tokens[0] if tokens else ""

    q_subj = first_noun(query_tokens)
    c_subj = first_noun(candidate_tokens)
    passed = q_subj == c_subj or not q_subj or not c_subj
    return passed, {"query_subject": q_subj, "candidate_subject": c_subj}


def _time_check(query: str, candidate_question: str) -> tuple:
    q_has_time = any(w in query for w in TIME_WORDS)
    c_has_time = any(w in candidate_question for w in TIME_WORDS)
    passed = q_has_time == c_has_time
    return passed, {"query_time": q_has_time, "candidate_time": c_has_time}


def validate(query: str, candidate_entry: dict) -> ValidationResult:
    candidate_question = candidate_entry["question"]
    query_tokens = tokenize(query)
    candidate_tokens = candidate_entry.get("tokens") or tokenize(candidate_question)

    checks = {
        "negation": _negation_check(query, candidate_question),
        "abstraction": _abstraction_check(query_tokens, candidate_tokens),
        "subject": _subject_check(query_tokens, candidate_tokens),
        "time": _time_check(query, candidate_question),
    }

    passed_count = sum(1 for passed, _ in checks.values() if passed)
    failed = [name for name, (passed, _) in checks.items() if not passed]
    details = {name: detail for name, (_, detail) in checks.items()}
    confidence = passed_count / 4.0

    return ValidationResult(
        passed=passed_count >= MIN_PASSING_CHECKS,
        confidence=confidence,
        failed_checks=failed,
        details=details,
    )
