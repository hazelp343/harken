"""Lightweight, download-free stand-ins for a tokenizer and a causal LM.

These let the full audio-QA pipeline be exercised on CPU in milliseconds, with
no model downloads. They implement just enough of the Hugging Face interface
(``get_input_embeddings``, ``forward(inputs_embeds=...)``, ``generate``) for
harken to drive them exactly as it would a real model.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

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


@dataclass
class TinyCausalLMConfig:
    vocab_size: int = 256
    hidden_size: int = 64


@dataclass
class CausalLMOutput:
    """Minimal stand-in for a Hugging Face causal-LM output."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TinyCausalLM(nn.Module):
    """A tiny GRU-based causal LM exposing the bits harken relies on."""

    def __init__(self, config: TinyCausalLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyCausalLMConfig()
        h = self.config.hidden_size
        self.embed = nn.Embedding(self.config.vocab_size, h)
        self.rnn = nn.GRU(h, h, batch_first=True)
        self.lm_head = nn.Linear(h, self.config.vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def _backbone(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(inputs_embeds)
        return out

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("provide input_ids or inputs_embeds")
            inputs_embeds = self.embed(input_ids)
        logits = self.lm_head(self._backbone(inputs_embeds))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 8,
        eos_token_id: int | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Greedy decode, returning only the newly generated token ids."""
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("provide input_ids or inputs_embeds")
            inputs_embeds = self.embed(input_ids)

        cur = inputs_embeds
        generated: list[torch.Tensor] = []
        for _step in range(max_new_tokens):
            logits = self.lm_head(self._backbone(cur))[:, -1, :]
            nxt = logits.argmax(dim=-1)
            generated.append(nxt)
            cur = torch.cat([cur, self.embed(nxt).unsqueeze(1)], dim=1)
        return torch.stack(generated, dim=1)
