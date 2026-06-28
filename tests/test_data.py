import numpy as np
from harken.data import AudioQAExample, Collator, load_examples, save_examples


def test_example_from_dict_filters_unknown():
    ex = AudioQAExample.from_dict(
        {"question": "q?", "answer": "a", "type": "solvable", "junk": 1}
    )
    assert ex.question == "q?"
    assert ex.answer == "a"


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "d.jsonl"
    examples = [
        AudioQAExample(question="q1?", answer="dog", audio="a.wav"),
        AudioQAExample(
            question="q2?", answer="b", options=["a", "b"], type="unanswerable", group=1
        ),
    ]
    save_examples(examples, path)
    loaded = load_examples(path)
    assert len(loaded) == 2
    assert loaded[0].audio == "a.wav"
    assert loaded[1].options == ["a", "b"]
    assert loaded[1].type == "unanswerable"


def test_collator_without_audio():
    from harken.processor import AudioQAProcessor
    from harken.testing import TinyTokenizer

    proc = AudioQAProcessor(TinyTokenizer(), num_audio_tokens=4)
    collator = Collator(proc)
    batch = collator(
        [AudioQAExample(question="q1?"), AudioQAExample(question="q2 longer?")]
    )
    assert batch["input_ids"].shape[0] == 2
    assert "audio_values" not in batch


def test_collator_with_audio_arrays():
    from harken.processor import AudioQAProcessor
    from harken.testing import TinyTokenizer

    proc = AudioQAProcessor(TinyTokenizer(), num_audio_tokens=4)
    collator = Collator(proc)
    batch = collator(
        [
            AudioQAExample(question="q1?", audio=np.zeros(4000, dtype=np.float32)),
            AudioQAExample(question="q2?", audio=np.zeros(8000, dtype=np.float32)),
        ]
    )
    assert batch["audio_values"].shape == (2, 8000)


def test_collator_mixed_audio_raises():
    import pytest
    from harken.processor import AudioQAProcessor
    from harken.testing import TinyTokenizer

    proc = AudioQAProcessor(TinyTokenizer(), num_audio_tokens=4)
    collator = Collator(proc)
    with pytest.raises(ValueError):
        collator(
            [
                AudioQAExample(question="q1?", audio=np.zeros(10, dtype=np.float32)),
                AudioQAExample(question="q2?"),
            ]
        )
