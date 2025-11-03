"""Project-wide default constants for audio processing."""

from __future__ import annotations

# Most pretrained audio encoders (CLAP, AST, BEATs, Whisper) operate at 16 kHz.
DEFAULT_SAMPLE_RATE: int = 16_000

# Defaults for the log-mel front end.
DEFAULT_N_FFT: int = 400
DEFAULT_HOP_LENGTH: int = 160
DEFAULT_N_MELS: int = 64

# Hard ceiling on audio length (seconds) accepted by the default front end,
# guarding against accidentally feeding multi-hour files into memory.
MAX_AUDIO_SECONDS: float = 3600.0
