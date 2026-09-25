import numpy as np
import pytest

from data.collect_transitions import EpisodeWriter, iter_transitions, load_episode


def _random_image(rng):
    return rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)


def test_episode_roundtrip(tmp_path):
    writer = EpisodeWriter(tmp_path, task_name="NeedleReach", meta={"seed": 3, "split": "ID"})
    rng = np.random.default_rng(0)
    q = np.zeros(6, dtype=np.float32)
    for t in range(5):
        action = None if t == 4 else rng.normal(size=6).astype(np.float32)
        writer.add_step(_random_image(rng), q, action)
        if action is not None:
            q = q + action

    path = writer.close()
    assert path.exists()

    episode = load_episode(path)
    assert episode["images"].shape == (5, 8, 8, 3)
    assert episode["q"].shape == (5, 6)
    assert episode["actions"].shape == (4, 6)
    assert episode["meta"]["seed"] == 3
    assert episode["meta"]["task"] == "NeedleReach"
    assert episode["meta"]["split"] == "ID"

    transitions = list(iter_transitions(episode))
    assert len(transitions) == 4
    img_t, q_t, a_t, img_t1 = transitions[0]
    np.testing.assert_array_equal(img_t, episode["images"][0])
    np.testing.assert_array_equal(img_t1, episode["images"][1])
    np.testing.assert_array_equal(q_t, episode["q"][0])
    np.testing.assert_array_equal(a_t, episode["actions"][0])


def test_add_step_after_close_raises(tmp_path):
    writer = EpisodeWriter(tmp_path, task_name="X")
    rng = np.random.default_rng(0)
    writer.add_step(_random_image(rng), np.zeros(6), np.zeros(6))
    writer.add_step(_random_image(rng), np.zeros(6), None)
    writer.close()
    with pytest.raises(RuntimeError):
        writer.add_step(_random_image(rng), np.zeros(6), None)


def test_close_requires_matching_action_count(tmp_path):
    writer = EpisodeWriter(tmp_path, task_name="X")
    rng = np.random.default_rng(0)
    # two frames, but both carry an action -> 2 actions for 2 frames,
    # while close() expects exactly len(frames)-1 = 1
    writer.add_step(_random_image(rng), np.zeros(6), np.zeros(6))
    writer.add_step(_random_image(rng), np.zeros(6), np.zeros(6))
    with pytest.raises(ValueError):
        writer.close()


def test_close_requires_at_least_two_frames(tmp_path):
    writer = EpisodeWriter(tmp_path, task_name="X")
    rng = np.random.default_rng(0)
    writer.add_step(_random_image(rng), np.zeros(6), None)
    with pytest.raises(ValueError):
        writer.close()
