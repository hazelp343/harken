"""Turn ``(text, audio)`` pairs into model inputs.

The processor pairs a tokenizer with the audio front end: it expands the single
``<audio>`` placeholder in the prompt into ``num_audio_tokens`` copies (so each
maps to one projected audio embedding) and returns padded tensors ready for the
model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch

from harken.constants import AUDIO_TOKEN


class TokenizerLike(Protocol):
    """The minimal tokenizer interface the processor relies on."""

    audio_token_id: int
    pad_token_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class AudioQAProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        num_audio_tokens: int,
        *,
        audio_token: str = AUDIO_TOKEN,
        sample_rate: int = 16_000,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_audio_tokens = num_audio_tokens
        self.audio_token = audio_token
        self.sample_rate = sample_rate

    def expand_audio_tokens(self, text: str) -> str:
        """Expand each placeholder into ``num_audio_tokens`` copies."""
        return text.replace(self.audio_token, self.audio_token * self.num_audio_tokens)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(self.expand_audio_tokens(text))

    def __call__(
        self,
        text: str | Sequence[str],
        audio: np.ndarray | Sequence[np.ndarray] | None = None,
    ) -> dict[str, torch.Tensor]:
        single = isinstance(text, str)
        texts: list[str] = [text] if isinstance(text, str) else list(text)
        if audio is not None and not all(self.audio_token in t for t in texts):
            raise ValueError(
                f"audio was supplied but a prompt has no {self.audio_token!r} "
                "placeholder; add one with harken.templates.build_prompt"
            )
        encoded = [self._encode(t) for t in texts]

        max_len = max(len(e) for e in encoded)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0))
        input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, e in enumerate(encoded):
            input_ids[i, : len(e)] = torch.tensor(e, dtype=torch.long)
            attention_mask[i, : len(e)] = 1

        out: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if audio is not None:
            audios = [audio] if single else list(audio)
            wavs = [torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in audios]
            max_samples = max(w.shape[0] for w in wavs)
            batch = torch.zeros(len(wavs), max_samples, dtype=torch.float32)
            for i, w in enumerate(wavs):
                batch[i, : w.shape[0]] = w
            out["audio_values"] = batch
        return out
