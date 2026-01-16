"""The :class:`AudioQAModel` ties an audio encoder, a projector and a causal LM
together: audio is encoded, projected into the LLM embedding space, and spliced
in wherever the prompt carries an audio token.
"""

from __future__ import annotations

import torch
from torch import nn

from harken.config import AudioQAConfig
from harken.encoders.base import AudioEncoder
from harken.projectors import Projector


class AudioQAModel(nn.Module):
    def __init__(
        self,
        encoder: AudioEncoder,
        projector: Projector,
        language_model: nn.Module,
        config: AudioQAConfig,
        audio_token_id: int,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.language_model = language_model
        self.config = config
        self.audio_token_id = audio_token_id

        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_audio(self, audio_values: torch.Tensor) -> torch.Tensor:
        """Encode and project audio to ``(batch, num_audio_tokens, llm_dim)``."""
        frames = self.encoder(audio_values)
        return self.projector(frames)

    def _merge_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Replace audio-token embeddings with the projected audio embeddings.

        Each row's audio-token positions are filled, in order, from that row's
        projected audio embeddings (LLaVA-style splicing).
        """
        merged = inputs_embeds.clone()
        n_audio = audio_embeds.shape[1]
        is_audio = input_ids == self.audio_token_id
        for i in range(input_ids.shape[0]):
            positions = is_audio[i].nonzero(as_tuple=True)[0]
            k = min(positions.numel(), n_audio)
            if k:
                merged[i, positions[:k]] = audio_embeds[i, :k].to(merged.dtype)
        return merged

    def _prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        audio_values: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if audio_values is not None:
            audio_embeds = self.encode_audio(audio_values)
            inputs_embeds = self._merge_audio_embeddings(
                input_ids, inputs_embeds, audio_embeds
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        inputs_embeds = self._prepare_inputs_embeds(input_ids, audio_values)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
