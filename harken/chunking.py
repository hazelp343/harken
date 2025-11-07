"""Split long audio into fixed-length windows for encoders with a bounded
receptive field. Implementation lands in a later change.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frame_signal"]


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int | None = None,
    pad: bool = True,
) -> np.ndarray:
    """Slice ``signal`` into overlapping frames of ``frame_length`` samples."""
    raise NotImplementedError
