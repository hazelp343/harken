import pytest

from harken.config import AudioQAConfig
from harken.exceptions import ConfigError


def test_defaults_are_valid():
    cfg = AudioQAConfig()
    assert cfg.encoder_name == "dummy"
    assert cfg.num_audio_tokens > 0
    assert cfg.audio_token


def test_dict_roundtrip():
    cfg = AudioQAConfig(encoder_dim=128, llm_dim=256, num_audio_tokens=4)
    again = AudioQAConfig.from_dict(cfg.to_dict())
    assert again == cfg


def test_from_dict_ignores_unknown_keys():
    cfg = AudioQAConfig.from_dict({"encoder_dim": 64, "bogus": 1})
    assert cfg.encoder_dim == 64


def test_json_save_load(tmp_path):
    cfg = AudioQAConfig(projector="stack", stack_factor=2)
    path = tmp_path / "config.json"
    cfg.save(path)
    assert AudioQAConfig.load(path) == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_audio_tokens": 0},
        {"encoder_dim": 0},
        {"llm_dim": -1},
        {"stack_factor": 0},
        {"audio_token": ""},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ConfigError):
        AudioQAConfig(**kwargs)
