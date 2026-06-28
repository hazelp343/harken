"""Audio encoder registry and public exports."""

from __future__ import annotations

from collections.abc import Callable

from harken.encoders.base import AudioEncoder
from harken.registry import Registry

audio_encoders: Registry[type[AudioEncoder]] = Registry("audio encoder")


def register_encoder(name: str) -> Callable[[type[AudioEncoder]], type[AudioEncoder]]:
    """Class decorator that registers an :class:`AudioEncoder` under ``name``."""
    return audio_encoders.register(name)


def list_encoders() -> list[str]:
    """Names of all registered encoders."""
    return audio_encoders.keys()


__all__ = [
    "AudioEncoder",
    "audio_encoders",
    "register_encoder",
    "list_encoders",
]
