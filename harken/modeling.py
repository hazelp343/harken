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
