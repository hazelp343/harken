"""A deterministic, download-free encoder used in tests and examples.

It maps a waveform to a short sequence of frame embeddings with a fixed
(non-trainable) projection, so its output is fully reproducible without any
pretrained weights. It is intentionally *not* a meaningful audio model -- it
exists so the rest of the pipeline can be exercised on CPU in milliseconds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from harken.encoders import register_encoder
from harken.encoders.base import AudioEncoder


@register_encoder("dummy")
class DummyAudioEncoder(AudioEncoder):
    def __init__(
        self,
        output_dim: int = 512,
        num_frames: int = 16,
        sampling_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate

        gen = torch.Generator().manual_seed(0)
        self.register_buffer("_proj", torch.randn(output_dim, generator=gen))
        self.register_buffer(
            "_frame_bias", torch.randn(num_frames, output_dim, generator=gen) * 0.1
        )

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        x = audio_values.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Pool the waveform into a fixed number of frame statistics.
        pooled = F.adaptive_avg_pool1d(x.unsqueeze(1), self.num_frames).squeeze(1)
        emb = pooled.unsqueeze(-1) * self._proj + self._frame_bias
        return emb
