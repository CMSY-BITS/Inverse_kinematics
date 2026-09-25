"""VJEPA2Encoder itself needs torch+transformers (a GPU workstation), but
its frame-normalization helper is pure and static, so it's tested here
without either dependency — see models/_torch_optional.py's pattern."""
import numpy as np
import pytest

from models.jepa_wrapper import VJEPA2Encoder


def test_single_frame_is_duplicated_into_a_pair():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    clip = VJEPA2Encoder._as_frame_list(frame)
    assert len(clip) == 2
    np.testing.assert_array_equal(clip[0], clip[1])


def test_single_element_list_is_also_duplicated():
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    clip = VJEPA2Encoder._as_frame_list([frame])
    assert len(clip) == 2


def test_multi_frame_clip_passes_through_unchanged():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]
    clip = VJEPA2Encoder._as_frame_list(frames)
    assert len(clip) == 5
    for i, f in enumerate(clip):
        np.testing.assert_array_equal(f, frames[i])


def test_construction_without_torch_or_transformers_raises_clear_error():
    with pytest.raises(ImportError, match="VJEPA2Encoder"):
        VJEPA2Encoder()
