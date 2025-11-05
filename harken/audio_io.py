"""Audio input/output and basic signal conditioning.

Everything here works on plain ``float32`` numpy arrays so the rest of the
package never has to care about the on-disk audio format.
"""

from __future__ import annotations

import numpy as np

from harken.constants import DEFAULT_SAMPLE_RATE

__all__ = ["load_audio", "to_mono", "peak_normalize", "resample"]


def load_audio(
    path: str,
    sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> np.ndarray:
    """Load an audio file as a 1-D (or 2-D) float32 array at ``sr`` Hz.

    Implementation lands in a later change.
    """
    raise NotImplementedError
