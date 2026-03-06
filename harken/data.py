"""Dataset schema and IO for audio question answering.

An :class:`AudioQAExample` is a lightweight record; ``audio`` is usually a path
(loaded lazily during evaluation) but may also be a pre-loaded array.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FIELDS = {"question", "answer", "audio", "options", "type", "group", "id"}


@dataclass
class AudioQAExample:
    question: str
    answer: str = ""
    audio: str | np.ndarray | None = None
    options: list[str] | None = None
    type: str = "solvable"
    group: str | int | None = None
    id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> AudioQAExample:
        return cls(**{k: v for k, v in data.items() if k in _FIELDS})

    def to_dict(self) -> dict:
        out: dict = {"question": self.question, "answer": self.answer}
        if isinstance(self.audio, str):
            out["audio"] = self.audio
        for key in ("options", "group", "id"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out["type"] = self.type
        return out


def load_examples(path: str | Path) -> list[AudioQAExample]:
    """Load examples from a JSON Lines file (one object per line)."""
    examples: list[AudioQAExample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(AudioQAExample.from_dict(json.loads(line)))
    return examples


def save_examples(examples: Iterable[AudioQAExample], path: str | Path) -> None:
    """Write examples to a JSON Lines file."""
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict()) + "\n")
