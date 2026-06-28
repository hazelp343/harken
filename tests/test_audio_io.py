import numpy as np
import pytest

from harken.audio_io import load_audio, peak_normalize, resample, to_mono
from harken.exceptions import AudioLoadError


def test_to_mono_passthrough_1d():
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.array_equal(to_mono(x), x)


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [2.0, 3.0])


def test_to_mono_rejects_3d():
    with pytest.raises(ValueError):
        to_mono(np.zeros((2, 2, 2)))


def test_peak_normalize_scales_to_peak():
    x = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    out = peak_normalize(x)
    assert np.isclose(np.abs(out).max(), 1.0)


def test_peak_normalize_silence_is_safe():
    x = np.zeros(8, dtype=np.float32)
    assert np.array_equal(peak_normalize(x), x)


def test_resample_changes_length():
    x = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
    out = resample(x, 1000, 500)
    assert abs(out.shape[0] - 500) <= 1


def test_resample_noop_when_equal():
    x = np.ones(10, dtype=np.float32)
    assert np.array_equal(resample(x, 16000, 16000), x)


def test_load_audio_roundtrip(tmp_path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "tone.wav"
    tone = np.sin(np.linspace(0, 8 * np.pi, 16000)).astype(np.float32)
    sf.write(path, tone, 16000)

    loaded = load_audio(str(path), sr=16000)
    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert loaded.shape[0] == 16000


def test_load_audio_missing_file_raises():
    with pytest.raises(AudioLoadError):
        load_audio("/no/such/file.wav")
