"""Shared pytest fixtures."""

import numpy as np
import pytest

SAMPLE_RATE = 16_000


@pytest.fixture
def tone():
    """A 1-second 440 Hz sine tone as float32 mono."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def wav_path(tmp_path, tone):
    """Path to a small WAV file written from the ``tone`` fixture."""
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "tone.wav"
    sf.write(path, tone, SAMPLE_RATE)
    return str(path)


@pytest.fixture
def tiny_processor():
    from harken.processor import AudioQAProcessor
    from harken.testing import TinyTokenizer

    return AudioQAProcessor(TinyTokenizer(), num_audio_tokens=4)


@pytest.fixture
def tiny_model():
    from harken.config import AudioQAConfig
    from harken.modeling import build_model
    from harken.testing import TinyTokenizer

    tok = TinyTokenizer()
    config = AudioQAConfig(
        encoder_dim=32,
        llm_dim=64,
        num_audio_tokens=4,
        encoder_kwargs={"num_frames": 8},
    )
    return build_model(config, audio_token_id=tok.audio_token_id)
