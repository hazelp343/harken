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
