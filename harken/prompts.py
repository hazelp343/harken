"""Prompt construction for audio question answering.

A single placeholder token marks where the projected audio embeddings are
spliced into the text stream; the processor later expands it into the right
number of audio tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

# Placeholder inserted into the prompt text where audio should be attended to.
AUDIO_TOKEN = "<audio>"

__all__ = ["AUDIO_TOKEN", "build_prompt", "format_options", "build_chat"]

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def format_options(options: Sequence[str]) -> str:
    """Render multiple-choice options as ``(a) ... (b) ...`` lines."""
    if len(options) > len(_LETTERS):
        raise ValueError("too many options to label with single letters")
    return "\n".join(f"({_LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def build_prompt(
    question: str,
    options: Sequence[str] | None = None,
    audio_token: str = AUDIO_TOKEN,
    instruction: str | None = None,
) -> str:
    """Build a user prompt placing the audio before the question.

    When ``options`` are given the prompt becomes a multiple-choice question and
    a short instruction to answer with a letter is appended.
    """
    parts = [audio_token, question.strip()]
    if options:
        parts.append(format_options(options))
        parts.append(instruction or "Answer with the letter of the correct option.")
    elif instruction:
        parts.append(instruction)
    return "\n".join(parts)


def build_chat(
    question: str,
    options: Sequence[str] | None = None,
    system: str | None = None,
    audio_token: str = AUDIO_TOKEN,
) -> list[dict[str, str]]:
    """Return chat messages (role/content) for a tokenizer chat template."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {"role": "user", "content": build_prompt(question, options, audio_token)}
    )
    return messages
