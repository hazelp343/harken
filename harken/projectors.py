"""Projectors map encoder frame embeddings into the language model's space.

Every projector takes ``(batch, frames, in_dim)`` and returns a fixed-width
``(batch, num_tokens, out_dim)`` sequence that is spliced into the LLM input
embeddings. Several designs are provided, mirroring common audio-LLM recipes.
"""

from __future__ import annotations

import abc

import torch
import torch.nn.functional as F
from torch import nn

from harken.registry import Registry

projectors: Registry[type["Projector"]] = Registry("projector")


class Projector(nn.Module, abc.ABC):
    def __init__(self, in_dim: int, out_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens = num_tokens

    @abc.abstractmethod
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``(B, frames, in_dim) -> (B, num_tokens, out_dim)``."""

    @staticmethod
    def _pool_time(x: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Adaptive average pool over the time axis to ``n_tokens`` steps."""
        return F.adaptive_avg_pool1d(x.transpose(1, 2), n_tokens).transpose(1, 2)


@projectors.register("linear")
class LinearProjector(Projector):
    """A single affine map per frame, pooled to a fixed token count."""

    def __init__(self, in_dim: int, out_dim: int, num_tokens: int = 8) -> None:
        super().__init__(in_dim, out_dim, num_tokens)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self._pool_time(self.proj(frames), self.num_tokens)
