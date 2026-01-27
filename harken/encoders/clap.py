"""CLAP audio encoder wrapper.

Wraps a frozen `CLAP <https://github.com/LAION-AI/CLAP>`_ audio tower from the
Hugging Face hub and exposes its frame embeddings. Requires the optional ``hf``
extra (``pip install harken[hf]``).
"""

from __future__ import annotations

import torch

from harken.encoders import register_encoder
from harken.encoders.base import AudioEncoder
from harken.utils import require


@register_encoder("clap")
class ClapAudioEncoder(AudioEncoder):
    def __init__(
        self,
        model_id: str = "laion/clap-htsat-unfused",
        sampling_rate: int = 48_000,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        transformers = require("transformers", extra="hf")
        self.model_id = model_id
        self.sampling_rate = sampling_rate

        self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            model_id
        )
        self.model = transformers.ClapAudioModel.from_pretrained(model_id)
        self.model.eval()
        self.output_dim = output_dim or int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        waves = [w.detach().cpu().numpy() for w in audio_values]
        inputs = self.feature_extractor(
            waves, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).to(audio_values.device)
        hidden = self.model(**inputs).last_hidden_state
        # CLAP's HTSAT tower returns (B, C, T, F); flatten spatial dims to a
        # sequence of (T*F) tokens with C channels.
        if hidden.dim() == 4:
            b, c, t, f = hidden.shape
            hidden = hidden.permute(0, 2, 3, 1).reshape(b, t * f, c)
        return hidden
