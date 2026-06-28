"""Whisper encoder wrapper.

Uses the encoder half of a frozen `Whisper <https://github.com/openai/whisper>`_
model as an audio feature extractor. Whisper expects 16 kHz audio and returns a
fixed-length sequence of frame embeddings. Requires the optional ``hf`` extra.
"""

from __future__ import annotations

import torch

from harken.encoders import register_encoder
from harken.encoders.base import AudioEncoder
from harken.utils import require


@register_encoder("whisper")
class WhisperAudioEncoder(AudioEncoder):
    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        sampling_rate: int = 16_000,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        transformers = require("transformers", extra="hf")
        self.model_id = model_id
        self.sampling_rate = sampling_rate

        self.feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
            model_id
        )
        self.encoder = transformers.WhisperModel.from_pretrained(model_id).get_encoder()
        self.encoder.eval()
        self.output_dim = output_dim or int(self.encoder.config.d_model)

    @torch.no_grad()
    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        waves = [w.detach().cpu().numpy() for w in audio_values]
        features = self.feature_extractor(
            waves, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features.to(audio_values.device)
        return self.encoder(features).last_hidden_state
