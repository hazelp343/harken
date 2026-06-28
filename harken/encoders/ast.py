"""Audio Spectrogram Transformer (AST) encoder wrapper.

Wraps a frozen `AST <https://github.com/YuanGongND/ast>`_ model from the Hugging
Face hub. AST consumes log-mel patches and returns a sequence of patch
embeddings (including the CLS/distillation tokens). Requires the optional ``hf``
extra.
"""

from __future__ import annotations

import torch

from harken.encoders import register_encoder
from harken.encoders.base import AudioEncoder
from harken.utils import require


@register_encoder("ast")
class ASTAudioEncoder(AudioEncoder):
    def __init__(
        self,
        model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        sampling_rate: int = 16_000,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        transformers = require("transformers", extra="hf")
        self.model_id = model_id
        self.sampling_rate = sampling_rate

        self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            model_id
        )
        self.model = transformers.ASTModel.from_pretrained(model_id)
        self.model.eval()
        self.output_dim = output_dim or int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        waves = [w.detach().cpu().numpy() for w in audio_values]
        inputs = self.feature_extractor(
            waves, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).to(audio_values.device)
        return self.model(**inputs).last_hidden_state
