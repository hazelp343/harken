import torch
from harken.encoders import get_encoder, list_encoders
from harken.encoders.dummy import DummyAudioEncoder


def test_dummy_is_registered():
    assert "dummy" in list_encoders()


def test_get_encoder_builds_instance():
    enc = get_encoder("dummy", output_dim=32, num_frames=4)
    assert isinstance(enc, DummyAudioEncoder)
    assert enc.hidden_size == 32


def test_dummy_output_shape():
    enc = get_encoder("dummy", output_dim=32, num_frames=8)
    wav = torch.randn(2, 16000)
    out = enc(wav)
    assert out.shape == (2, 8, 32)


def test_dummy_accepts_unbatched():
    enc = DummyAudioEncoder(output_dim=16, num_frames=4)
    out = enc(torch.randn(16000))
    assert out.shape == (1, 4, 16)


def test_dummy_is_deterministic():
    enc = DummyAudioEncoder(output_dim=16, num_frames=4)
    wav = torch.randn(1, 8000)
    a = enc(wav)
    b = enc(wav)
    assert torch.allclose(a, b)
