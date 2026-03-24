import json

import numpy as np
import pytest
from harken.cli import main


def test_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "harken" in out
    assert "dummy" in out
    assert "mlp" in out


def test_no_command_prints_help():
    assert main([]) == 1


def test_answer(tmp_path, capsys):
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)

    rc = main(
        [
            "answer",
            "--audio",
            str(wav),
            "--question",
            "what is this?",
            "--num-audio-tokens",
            "4",
            "--max-new-tokens",
            "3",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() != ""


def test_evaluate(tmp_path, capsys):
    from harken.data import AudioQAExample, save_examples

    dataset = tmp_path / "data.jsonl"
    save_examples(
        [
            AudioQAExample(question="q1?", answer="dog"),
            AudioQAExample(question="q2?", answer="cat"),
        ],
        dataset,
    )
    rc = main(
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--num-audio-tokens",
            "4",
            "--max-new-tokens",
            "2",
        ]
    )
    assert rc == 0
    metrics = json.loads(capsys.readouterr().out)
    assert "exact_match" in metrics
    assert "abstention" in metrics
