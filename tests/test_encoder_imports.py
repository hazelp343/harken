"""The HF-backed encoders should register without importing transformers, and
fail with a clear DependencyError only when actually instantiated without it.
"""

import pytest
from harken.encoders import get_encoder, list_encoders
from harken.exceptions import DependencyError

HF_ENCODERS = ["clap", "whisper", "ast"]


def _transformers_installed() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.parametrize("name", HF_ENCODERS)
def test_hf_encoder_is_registered(name):
    assert name in list_encoders()


@pytest.mark.parametrize("name", HF_ENCODERS)
def test_hf_encoder_requires_transformers(name):
    if _transformers_installed():
        pytest.skip("transformers is installed; dependency guard not exercised")
    with pytest.raises(DependencyError):
        get_encoder(name)
