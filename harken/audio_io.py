"""Audio input/output and basic signal conditioning.

Everything here works on plain ``float32`` numpy arrays so the rest of the
package never has to care about the on-disk audio format.
"""

from __future__ import annotations

import numpy as np

from harken.constants import DEFAULT_SAMPLE_RATE

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


def load_audio(
    path: str,
    sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> np.ndarray:
    """Load an audio file as a 1-D (or 2-D) float32 array at ``sr`` Hz.

    Implementation lands in a later change.
    """
    raise NotImplementedError
