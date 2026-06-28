"""Split long audio into fixed-length windows for encoders with a bounded
receptive field.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frame_signal", "num_frames"]


def num_frames(n_samples: int, frame_length: int, hop_length: int) -> int:
    """Number of frames :func:`frame_signal` produces for ``n_samples``."""
    if n_samples < frame_length:
        return 1
    return 1 + (n_samples - frame_length) // hop_length


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int | None = None,
    pad: bool = True,
) -> np.ndarray:
    """Slice ``signal`` into frames of ``frame_length`` samples.

    Returns an array of shape ``(n_frames, frame_length)``. When ``pad`` is
    ``True`` the final frame is zero-padded so no samples are dropped; when
    ``False`` a trailing partial frame is discarded.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    hop = hop_length or frame_length
    if hop <= 0:
        raise ValueError("hop_length must be positive")

    if signal.shape[0] <= frame_length:
        if not pad:
            return signal[np.newaxis, :].copy()
        out = np.zeros((1, frame_length), dtype=np.float32)
        out[0, : signal.shape[0]] = signal
        return out

    n = num_frames(signal.shape[0], frame_length, hop)
    needed = (n - 1) * hop + frame_length
    if pad and needed > signal.shape[0]:
        signal = np.pad(signal, (0, needed - signal.shape[0]))

    return np.stack(
        [signal[i * hop : i * hop + frame_length] for i in range(n)], axis=0
    )
