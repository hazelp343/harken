"""Typed configuration objects for an audio-QA model.

The configuration is deliberately a plain dataclass (not tied to any framework)
so it can be serialised to JSON and round-tripped without importing torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from harken.constants import AUDIO_TOKEN, DEFAULT_SAMPLE_RATE
from harken.exceptions import ConfigError

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about audio recordings."
)


@dataclass
class AudioQAConfig:
    """Describes how an audio encoder is wired to a language model."""

    encoder_name: str = "dummy"
    llm_name: str = "tiny"
    projector: str = "mlp"
    encoder_dim: int = 512
    llm_dim: int = 768
    num_audio_tokens: int = 8
    projector_hidden_dim: int | None = None
    stack_factor: int = 4
    sample_rate: int = DEFAULT_SAMPLE_RATE
    audio_token: str = AUDIO_TOKEN
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    encoder_kwargs: dict = field(default_factory=dict)
    projector_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_audio_tokens <= 0:
            raise ConfigError("num_audio_tokens must be positive")
        if self.encoder_dim <= 0 or self.llm_dim <= 0:
            raise ConfigError("encoder_dim and llm_dim must be positive")
        if self.stack_factor <= 0:
            raise ConfigError("stack_factor must be positive")
        if not self.audio_token:
            raise ConfigError("audio_token must be a non-empty string")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AudioQAConfig:
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AudioQAConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
