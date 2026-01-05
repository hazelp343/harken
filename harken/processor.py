"""Turn ``(text, audio)`` pairs into model inputs.

The processor pairs a tokenizer with the audio front end: it expands the single
``<audio>`` placeholder in the prompt into ``num_audio_tokens`` copies (so each
maps to one projected audio embedding) and returns padded tensors ready for the
model.
"""

from __future__ import annotations

import numpy as np
import torch

from harken.prompts import AUDIO_TOKEN


class AudioQAProcessor:
    def __init__(
        self,
        tokenizer: object,
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
        text: str,
        audio: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        ids = self._encode(text)
        out: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }
        if audio is not None:
            wav = torch.as_tensor(np.asarray(audio, dtype=np.float32))
            out["audio_values"] = wav.unsqueeze(0)
        return out
