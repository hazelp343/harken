"""Dataset schema and IO for audio question answering.

An :class:`AudioQAExample` is a lightweight record; ``audio`` is usually a path
(loaded lazily during evaluation) but may also be a pre-loaded array.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harken.audio_io import load_audio
from harken.templates import build_prompt

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


class Collator:
    """Turn a list of examples into batched processor inputs.

    Audio is resolved per example (loading file paths and passing arrays
    through). A batch must be all-audio or no-audio so the tensors stack.
    """

    def __init__(self, processor, audio_loader=None) -> None:
        self.processor = processor
        self.audio_loader = audio_loader or load_audio

    def _resolve_audio(self, example: AudioQAExample) -> np.ndarray | None:
        if example.audio is None:
            return None
        if isinstance(example.audio, str):
            return self.audio_loader(example.audio, sr=self.processor.sample_rate)
        return np.asarray(example.audio, dtype=np.float32)

    def __call__(self, examples: Sequence[AudioQAExample]) -> dict:
        prompts = [build_prompt(e.question, e.options) for e in examples]
        audios = [self._resolve_audio(e) for e in examples]
        if all(a is None for a in audios):
            return self.processor(prompts)
        if any(a is None for a in audios):
            raise ValueError("a batch must be either all-audio or no-audio")
        return self.processor(prompts, audio=audios)
