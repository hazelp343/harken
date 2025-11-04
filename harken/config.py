"""Typed configuration objects for an audio-QA model.

The configuration is deliberately a plain dataclass (not tied to any framework)
so it can be serialised to JSON and round-tripped without importing torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from harken.constants import DEFAULT_SAMPLE_RATE


@dataclass
class AudioQAConfig:
    """Describes how an audio encoder is wired to a language model."""

    encoder_name: str = "dummy"
    llm_name: str = "tiny"
    projector: str = "mlp"
    encoder_dim: int = 512
    llm_dim: int = 768
    num_audio_tokens: int = 8
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AudioQAConfig:
        fields = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in data.items() if k in fields})
