import numpy as np
import torch
from harken.config import AudioQAConfig
from harken.data import AudioQAExample
from harken.evaluation import evaluate_dataset
from harken.modeling import build_model
from harken.processor import AudioQAProcessor
from harken.prompts import build_prompt
from harken.testing import TinyTokenizer


def _setup():
    tok = TinyTokenizer()
    cfg = AudioQAConfig(
        encoder_dim=32,
        llm_dim=64,
        num_audio_tokens=4,
        encoder_kwargs={"num_frames": 8},
    )
    model = build_model(cfg, audio_token_id=tok.audio_token_id)
    proc = AudioQAProcessor(tok, cfg.num_audio_tokens)
    return model, proc


def test_end_to_end_evaluation():
    model, proc = _setup()
    examples = [
        AudioQAExample(
            question="what animal?",
            answer="dog",
            audio=np.zeros(8000, dtype=np.float32),
            type="solvable",
            group=1,
        ),
        AudioQAExample(
            question="which note is it?",
            answer="",
            audio=np.zeros(8000, dtype=np.float32),
            type="unanswerable",
            group=1,
        ),
    ]
    result = evaluate_dataset(model, proc, examples, max_new_tokens=3)
    assert len(result["predictions"]) == 2
    assert "exact_match" in result["metrics"]
    assert "abstention" in result["metrics"]


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model, proc = _setup()
    inputs = proc(build_prompt("what is this?"), audio=np.zeros(8000, dtype=np.float32))
    labels = inputs["input_ids"].clone()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-2)

    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        out = model(**inputs, labels=labels)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())

    assert losses[-1] < losses[0]
