"""Metrics for open-ended and multiple-choice audio question answering.

The string-normalisation and token-F1 follow the SQuAD convention so scores are
comparable with the wider QA literature.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = {"a", "an", "the"}
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCT)
    tokens = [t for t in no_punct.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalised strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between prediction and reference."""
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, references: Sequence[str]) -> float:
    """Max exact match over several acceptable references."""
    return max((exact_match(prediction, r) for r in references), default=0.0)


def best_token_f1(prediction: str, references: Sequence[str]) -> float:
    """Max token F1 over several acceptable references."""
    return max((token_f1(prediction, r) for r in references), default=0.0)


# --- Abstention-aware scoring -------------------------------------------------
# Some audio questions are deliberately unanswerable (the answer is absent, the
# options are incompatible, or the question is not grounded in the audio). A good
# model should abstain rather than guess. These helpers, inspired by AQUA-Bench,
# score both answering and abstaining.

ABSTENTION_MARKERS = (
    "none of the above",
    "cannot be answered",
    "can't be answered",
    "unanswerable",
    "no answer",
    "i don't know",
    "i do not know",
    "not possible to answer",
    "not enough information",
)


def is_abstention(text: str, markers: Sequence[str] = ABSTENTION_MARKERS) -> bool:
    """Whether a response declines to answer."""
    low = text.lower()
    return any(marker in low for marker in markers)


def _record_correct(record: dict) -> bool:
    if record["type"] == "unanswerable":
        return is_abstention(record["prediction"])
    return exact_match(record["prediction"], record["answer"]) == 1.0


def _conditional_unanswerable_accuracy(records: Sequence[dict]) -> float | None:
    """Accuracy on unanswerable items, restricted to groups whose solvable
    item was answered correctly -- isolating abstention from basic recognition.
    """
    groups: dict[object, list[dict]] = {}
    for record in records:
        group = record.get("group")
        if group is not None:
            groups.setdefault(group, []).append(record)

    correct = total = 0
    for items in groups.values():
        solvable = [x for x in items if x["type"] == "solvable"]
        if not solvable or not all(_record_correct(x) for x in solvable):
            continue
        for item in items:
            if item["type"] == "unanswerable":
                total += 1
                correct += int(_record_correct(item))
    return correct / total if total else None


def score_abstention(records: Sequence[dict]) -> dict:
    """Score a list of records with ``prediction``, ``answer``, ``type`` keys.

    ``type`` is ``"solvable"`` or ``"unanswerable"``; an optional ``group`` key
    links items derived from the same audio clip for conditional accuracy.
    """
    by_type_correct: dict[str, int] = {}
    by_type_total: dict[str, int] = {}
    overall_correct = 0
    for record in records:
        qtype = record["type"]
        ok = _record_correct(record)
        by_type_correct[qtype] = by_type_correct.get(qtype, 0) + int(ok)
        by_type_total[qtype] = by_type_total.get(qtype, 0) + 1
        overall_correct += int(ok)

    accuracy_by_type = {
        qtype: by_type_correct[qtype] / by_type_total[qtype] for qtype in by_type_total
    }
    return {
        "overall_accuracy": overall_correct / len(records) if records else 0.0,
        "accuracy_by_type": accuracy_by_type,
        "conditional_unanswerable_accuracy": _conditional_unanswerable_accuracy(
            records
        ),
    }
