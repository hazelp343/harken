import pytest
from harken.evaluation import (
    aggregate,
    exact_match,
    is_abstention,
    normalize_answer,
    score_abstention,
    token_f1,
)


def test_normalize_strips_articles_and_punctuation():
    assert normalize_answer("The Dog!") == "dog"
    assert normalize_answer("a CAT.") == "cat"


def test_exact_match():
    assert exact_match("the dog", "Dog") == 1.0
    assert exact_match("dog", "cat") == 0.0


def test_token_f1_partial():
    score = token_f1("a brown dog", "the brown cat")
    assert 0.0 < score < 1.0


def test_token_f1_identical():
    assert token_f1("piano music", "piano music") == 1.0


def test_aggregate():
    out = aggregate(["dog", "cat"], ["dog", "bird"])
    assert out["n"] == 2
    assert out["exact_match"] == 0.5


@pytest.mark.parametrize(
    "text,expected",
    [
        ("none of the above", True),
        ("This is unanswerable.", True),
        ("It is a dog barking.", False),
    ],
)
def test_is_abstention(text, expected):
    assert is_abstention(text) is expected


def test_score_abstention_conditional():
    records = [
        {"prediction": "dog", "answer": "dog", "type": "solvable", "group": 1},
        {
            "prediction": "none of the above",
            "answer": "",
            "type": "unanswerable",
            "group": 1,
        },
        {"prediction": "cat", "answer": "dog", "type": "solvable", "group": 2},
        {"prediction": "it is a cat", "answer": "", "type": "unanswerable", "group": 2},
    ]
    out = score_abstention(records)
    assert out["accuracy_by_type"]["solvable"] == 0.5
    # Group 2's solvable is wrong, so only group 1's unanswerable counts -> 1.0.
    assert out["conditional_unanswerable_accuracy"] == 1.0


def test_score_abstention_without_groups():
    records = [
        {"prediction": "dog", "answer": "dog", "type": "solvable"},
        {"prediction": "unanswerable", "answer": "", "type": "unanswerable"},
    ]
    out = score_abstention(records)
    assert out["overall_accuracy"] == 1.0
    assert out["conditional_unanswerable_accuracy"] is None
