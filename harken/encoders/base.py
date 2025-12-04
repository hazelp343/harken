"""Base class for pretrained audio encoders.

An encoder turns a batch of waveforms into a sequence of frame embeddings that
the projector later maps into the language model's embedding space. Concrete
encoders typically wrap a frozen pretrained model (CLAP, Whisper, AST, ...).
"""

from __future__ import annotations

import abc

import torch
from torch import nn

from harken.constants import DEFAULT_SAMPLE_RATE


class AudioEncoder(nn.Module, abc.ABC):
    #: Hidden size of the produced frame embeddings.
    output_dim: int
    #: Sample rate (Hz) the encoder expects its input waveforms at.
    sampling_rate: int = DEFAULT_SAMPLE_RATE

    @abc.abstractmethod
    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """Encode audio into frame embeddings.

        Parameters
        ----------
        audio_values:
            A batch of waveforms shaped ``(batch, samples)``.

        Returns
        -------
        torch.Tensor
            Frame embeddings shaped ``(batch, frames, output_dim)``.
        """

    @property
    def hidden_size(self) -> int:
        return self.output_dim

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
