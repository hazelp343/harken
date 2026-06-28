"""Lightweight, download-free stand-ins for a tokenizer and a causal LM.

These let the full audio-QA pipeline be exercised on CPU in milliseconds, with
no model downloads. They implement just enough of the Hugging Face interface
(``get_input_embeddings``, ``forward(inputs_embeds=...)``, ``generate``) for
harken to drive them exactly as it would a real model.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

from harken.prompts import AUDIO_TOKEN

_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class TinyTokenizer:
    """A bounded, hashing word-level tokenizer with a dedicated audio token."""

    def __init__(self, vocab_size: int = 256, audio_token: str = AUDIO_TOKEN) -> None:
        self.audio_token = audio_token
        specials = [*_SPECIALS, audio_token]
        self._itos = dict(enumerate(specials))
        self._stoi = {tok: i for i, tok in self._itos.items()}
        self._n_special = len(specials)
        self.vocab_size = max(vocab_size, self._n_special + 1)

        self.pad_token_id = self._stoi["<pad>"]
        self.bos_token_id = self._stoi["<bos>"]
        self.eos_token_id = self._stoi["<eos>"]
        self.unk_token_id = self._stoi["<unk>"]
        self.audio_token_id = self._stoi[audio_token]

    def _word_id(self, word: str) -> int:
        span = self.vocab_size - self._n_special
        return self._n_special + (zlib.adler32(word.encode("utf-8")) % span)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = [self.bos_token_id] if add_special_tokens else []
        parts = text.split(self.audio_token)
        for i, part in enumerate(parts):
            ids.extend(self._word_id(w) for w in part.split())
            if i < len(parts) - 1:
                ids.append(self.audio_token_id)
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        out: list[str] = []
        specials = set(range(self._n_special))
        for i in ids:
            if i in self._itos:
                if skip_special_tokens and i in specials:
                    continue
                out.append(self._itos[i])
            else:
                out.append(f"w{i}")
        return " ".join(out)
