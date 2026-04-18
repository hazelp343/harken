"""Audio input/output and basic signal conditioning.

Everything here works on plain ``float32`` numpy arrays so the rest of the
package never has to care about the on-disk audio format.
"""

from __future__ import annotations

import numpy as np

from harken.constants import DEFAULT_SAMPLE_RATE
from harken.exceptions import AudioLoadError

__all__ = ["load_audio", "to_mono", "peak_normalize", "resample"]


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Average a multi-channel signal down to a single channel.

    Accepts ``(samples,)`` or ``(samples, channels)`` and always returns a
    1-D array.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        return signal
    if signal.ndim == 2:
        return signal.mean(axis=1).astype(np.float32)
    raise ValueError(f"expected a 1-D or 2-D signal, got shape {signal.shape}")


def peak_normalize(signal: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """Scale a signal so its largest absolute sample equals ``peak``.

    Silent signals are returned unchanged to avoid dividing by zero.
    """
    signal = np.asarray(signal, dtype=np.float32)
    max_abs = float(np.abs(signal).max()) if signal.size else 0.0
    if max_abs == 0.0:
        return signal
    return (signal * (peak / max_abs)).astype(np.float32)


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a signal from ``orig_sr`` to ``target_sr``.

    Handles mono ``(samples,)`` and multi-channel ``(samples, channels)`` input
    (resampled per channel). Uses ``resampy`` when installed (band-limited, high
    quality) and falls back to linear interpolation otherwise so the core stays
    usable with no extra dependencies.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if orig_sr == target_sr or signal.size == 0:
        return signal

    if signal.ndim == 2:
        channels = [
            resample(signal[:, c], orig_sr, target_sr) for c in range(signal.shape[1])
        ]
        return np.stack(channels, axis=1)

    try:
        import resampy

        return resampy.resample(signal, orig_sr, target_sr).astype(np.float32)
    except ImportError:
        duration = signal.shape[0] / float(orig_sr)
        n_target = int(round(duration * target_sr))
        if n_target <= 0:
            return signal[:0]
        src_idx = np.linspace(0.0, signal.shape[0] - 1, num=n_target)
        xp = np.arange(signal.shape[0])
        return np.interp(src_idx, xp, signal).astype(np.float32)


def load_audio(
    path: str,
    sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> np.ndarray:
    """Load an audio file as a 1-D (or 2-D) float32 array at ``sr`` Hz.

    Parameters
    ----------
    path:
        Path to any libsndfile-readable file (WAV, FLAC, OGG, ...).
    sr:
        Target sample rate. The file is resampled if it differs.
    mono:
        Downmix to a single channel when ``True``.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise AudioLoadError("soundfile is required to read audio files") from exc

    try:
        signal, file_sr = sf.read(path, dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as exc:
        raise AudioLoadError(f"could not read audio file: {path}") from exc

    if mono:
        signal = to_mono(signal)
    if file_sr != sr:
        signal = resample(signal, file_sr, sr)
    return np.ascontiguousarray(signal, dtype=np.float32)
