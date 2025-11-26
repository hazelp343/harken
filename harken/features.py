"""A small, pure-numpy log-mel spectrogram front end.

Keeping this dependency-free (no torchaudio/librosa required) means the audio
preprocessing path is import-light and trivially testable in CI. The optional
``audio`` extra can still be used by encoder wrappers that prefer librosa.
"""

from __future__ import annotations

import numpy as np

from harken.constants import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
)

__all__ = [
    "hz_to_mel",
    "mel_to_hz",
    "mel_filterbank",
    "power_spectrogram",
    "log_mel_spectrogram",
]


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """Convert frequency in Hz to the HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Build a triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``."""
    if fmax is None:
        fmax = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        fb[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def power_spectrogram(
    signal: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Short-time Fourier power spectrogram, shape ``(n_fft // 2 + 1, frames)``.

    A periodic Hann window is applied per frame; the signal is centre-padded so
    the first frame is centred on sample zero (matching librosa's default).
    """
    signal = np.asarray(signal, dtype=np.float32)
    pad = n_fft // 2
    padded = np.pad(signal, pad, mode="reflect" if signal.size > pad else "constant")

    n_frames = 1 + (len(padded) - n_fft) // hop_length
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    frames = np.stack(
        [padded[i * hop_length : i * hop_length + n_fft] for i in range(n_frames)],
        axis=0,
    )
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    return (np.abs(spectrum) ** 2).T.astype(np.float32)


def log_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    log_floor: float = 1e-10,
) -> np.ndarray:
    """Log-mel spectrogram of shape ``(n_mels, frames)``.

    This is the default front end for encoders that consume mel features. The
    output is natural-log magnitude, floored to avoid ``-inf`` on silence.
    """
    power = power_spectrogram(signal, n_fft=n_fft, hop_length=hop_length)
    fb = mel_filterbank(sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = fb @ power
    return np.log(np.maximum(mel, log_floor)).astype(np.float32)
