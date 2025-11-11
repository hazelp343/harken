"""Prompt construction for audio question answering.

A single placeholder token marks where the projected audio embeddings are
spliced into the text stream; the processor later expands it into the right
number of audio tokens.
"""

from __future__ import annotations

# Placeholder inserted into the prompt text where audio should be attended to.
AUDIO_TOKEN = "<audio>"

__all__ = ["AUDIO_TOKEN", "build_prompt"]


def build_prompt(question: str, audio_token: str = AUDIO_TOKEN) -> str:
    """Build a minimal prompt placing the audio before the question."""
    return f"{audio_token}\n{question}"
