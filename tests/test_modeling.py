import torch
from harken.config import AudioQAConfig
from harken.modeling import build_model
from harken.processor import AudioQAProcessor
from harken.templates import build_prompt
from harken.testing import TinyTokenizer


def make_setup(num_audio_tokens=4, llm_dim=32, encoder_dim=24):
    tok = TinyTokenizer(vocab_size=128)
    cfg = AudioQAConfig(
        encoder_dim=encoder_dim,
        llm_dim=llm_dim,
        num_audio_tokens=num_audio_tokens,
        encoder_kwargs={"num_frames": 10},
    )
    model = build_model(cfg, audio_token_id=tok.audio_token_id)
    proc = AudioQAProcessor(tok, num_audio_tokens=num_audio_tokens)
    return model, proc, cfg


def test_encode_audio_shape():
    model, _, cfg = make_setup()
    audio = torch.randn(2, 16000)
    emb = model.encode_audio(audio)
    assert emb.shape == (2, cfg.num_audio_tokens, cfg.llm_dim)


def test_forward_logits_shape():
    model, proc, cfg = make_setup()
    inputs = proc(build_prompt("what is this?"), audio=torch.zeros(16000).numpy())
    out = model(**inputs)
    seq_len = inputs["input_ids"].shape[1]
    assert out.logits.shape[0] == 1
    assert out.logits.shape[1] == seq_len


def test_forward_merges_audio_positions():
    model, proc, cfg = make_setup(num_audio_tokens=4)
    inputs = proc(build_prompt("hi?"), audio=torch.zeros(8000).numpy())
    n_audio = int((inputs["input_ids"] == proc.tokenizer.audio_token_id).sum())
    assert n_audio == cfg.num_audio_tokens
    # forward should run without shape errors when audio tokens are present.
    out = model(**inputs)
    assert out.logits is not None


def test_encoder_is_frozen_by_default():
    model, _, _ = make_setup()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    # projector stays trainable.
    assert any(p.requires_grad for p in model.projector.parameters())


def test_generate_returns_new_tokens():
    model, proc, _ = make_setup()
    inputs = proc(build_prompt("what is this?"), audio=torch.zeros(16000).numpy())
    out = model.generate(**inputs, max_new_tokens=5)
    assert out.shape == (1, 5)
    assert out.dtype == torch.long


def test_answer_returns_string():
    model, proc, _ = make_setup()
    text = model.answer(proc, "what do you hear?", torch.zeros(8000).numpy())
    assert isinstance(text, str)


def test_backward_updates_projector():
    model, proc, _ = make_setup()
    inputs = proc(build_prompt("q?"), audio=torch.zeros(8000).numpy())
    labels = inputs["input_ids"].clone()
    out = model(**inputs, labels=labels)
    out.loss.backward()
    grads = [p.grad for p in model.projector.parameters() if p.grad is not None]
    assert grads
