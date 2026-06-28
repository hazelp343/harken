import numpy as np
from harken.features import (
    hz_to_mel,
    log_mel_spectrogram,
    mel_filterbank,
    mel_to_hz,
    power_spectrogram,
)


def test_mel_hz_roundtrip():
    hz = np.array([0.0, 440.0, 1000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-3)


def test_filterbank_shape():
    fb = mel_filterbank(sample_rate=16000, n_fft=400, n_mels=64)
    assert fb.shape == (64, 201)
    assert np.all(fb >= 0.0)


def test_power_spectrogram_shape():
    sig = np.random.randn(16000).astype(np.float32)
    spec = power_spectrogram(sig, n_fft=400, hop_length=160)
    assert spec.shape[0] == 201
    assert spec.shape[1] > 0
    assert np.all(spec >= 0.0)


def test_log_mel_shape_and_finite():
    sig = np.sin(np.linspace(0, 50 * np.pi, 16000)).astype(np.float32)
    feats = log_mel_spectrogram(sig, sample_rate=16000, n_mels=64)
    assert feats.shape[0] == 64
    assert np.all(np.isfinite(feats))


def test_log_mel_silence_is_floored():
    feats = log_mel_spectrogram(np.zeros(16000, dtype=np.float32))
    assert np.all(np.isfinite(feats))
    assert feats.max() <= np.log(1e-10) + 1e-3
