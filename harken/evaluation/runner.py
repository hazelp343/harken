"""Run a model over a dataset and aggregate metrics.

The runner duck-types the examples: each item only needs ``question`` and
``answer`` attributes, with optional ``audio``, ``options``, ``type`` and
``group``. String ``audio`` values are treated as file paths and loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from harken.evaluation.metrics import exact_match, score_abstention, token_f1


def aggregate(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """Mean exact match and token F1 over aligned prediction/reference lists."""
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0, "token_f1": 0.0, "n": 0}
    em = sum(exact_match(p, r) for p, r in zip(predictions, references)) / n
    f1 = sum(token_f1(p, r) for p, r in zip(predictions, references)) / n
    return {"exact_match": em, "token_f1": f1, "n": n}


def evaluate_dataset(
    model,
    processor,
    examples: Iterable,
    *,
    audio_loader: Callable[..., np.ndarray] | None = None,
    max_new_tokens: int = 32,
) -> dict:
    """Generate an answer per example and return predictions plus metrics."""
    predictions: list[str] = []
    references: list[str] = []
    records: list[dict] = []

    for example in examples:
        audio = example.audio
        if isinstance(audio, str):
            if audio_loader is None:
                from harken.audio_io import load_audio

                audio_loader = load_audio
            audio = audio_loader(audio, sr=processor.sample_rate)

        options = getattr(example, "options", None)
        prediction = model.answer(
            processor,
            example.question,
            audio,
            options=options,
            max_new_tokens=max_new_tokens,
        )
        predictions.append(prediction)
        references.append(example.answer)
        records.append(
            {
                "prediction": prediction,
                "answer": example.answer,
                "type": getattr(example, "type", "solvable"),
                "group": getattr(example, "group", None),
            }
        )

    metrics = aggregate(predictions, references)
    metrics["abstention"] = score_abstention(records)
    return {"predictions": predictions, "metrics": metrics}
