import numpy as np
import pytest
from harken.chunking import frame_signal, num_frames


def test_num_frames_short_signal():
    assert num_frames(100, 400, 160) == 1


def test_num_frames_basic():
    assert num_frames(1000, 400, 200) == 4


def test_frame_signal_pads_short():
    x = np.ones(100, dtype=np.float32)
    frames = frame_signal(x, frame_length=400)
    assert frames.shape == (1, 400)
    assert frames[0, :100].sum() == 100
    assert frames[0, 100:].sum() == 0


def test_frame_signal_no_pad_truncates():
    x = np.arange(1000, dtype=np.float32)
    frames = frame_signal(x, frame_length=400, hop_length=400, pad=False)
    assert frames.shape == (2, 400)


def test_frame_signal_covers_all_with_pad():
    x = np.arange(1000, dtype=np.float32)
    frames = frame_signal(x, frame_length=400, hop_length=400, pad=True)
    assert frames.shape[0] == 3


def test_frame_signal_rejects_bad_args():
    with pytest.raises(ValueError):
        frame_signal(np.zeros(10), frame_length=0)
    with pytest.raises(ValueError):
        frame_signal(np.zeros(10), frame_length=4, hop_length=0)
