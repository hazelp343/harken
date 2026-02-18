import numpy as np
from harken.processor import AudioQAProcessor
from harken.prompts import build_prompt
from harken.testing import TinyTokenizer


def make_processor(num_audio_tokens=4):
    tok = TinyTokenizer(vocab_size=128)
    return AudioQAProcessor(tok, num_audio_tokens=num_audio_tokens)


def test_expand_audio_tokens_count():
    proc = make_processor(num_audio_tokens=5)
    expanded = proc.expand_audio_tokens("<audio>\nwhat is this?")
    assert expanded.count("<audio>") == 5


def test_single_example_has_right_audio_token_count():
    proc = make_processor(num_audio_tokens=4)
    prompt = build_prompt("what do you hear?")
    out = proc(prompt, audio=np.zeros(8000, dtype=np.float32))
    n_audio = int((out["input_ids"] == proc.tokenizer.audio_token_id).sum())
    assert n_audio == 4
    assert out["audio_values"].shape == (1, 8000)


def test_batch_padding():
    proc = make_processor()
    out = proc(
        [build_prompt("short?"), build_prompt("a much longer question indeed?")],
        audio=[np.zeros(4000, dtype=np.float32), np.zeros(8000, dtype=np.float32)],
    )
    assert out["input_ids"].shape[0] == 2
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["audio_values"].shape == (2, 8000)
    # Shorter example is right-padded.
    assert out["attention_mask"][0].sum() < out["attention_mask"][1].sum()


def test_no_audio_omits_audio_values():
    proc = make_processor()
    out = proc(build_prompt("q?"))
    assert "audio_values" not in out
