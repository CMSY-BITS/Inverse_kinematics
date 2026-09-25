"""Episode logger for (image, q, action, next_image) transitions.

Used both by the scripted/noisy-expert data collection pass in Gazebo
(build pipeline §4 step 3) and, unmodified, when logging closed-loop
evaluation rollouts — same on-disk format either way, so `eval/` and the
training data loader share one reader (`iter_transitions`).

An episode of length T is stored as one `.npz` per episode with T+1 image
frames and joint states but only T actions (there's no action taken after
the last frame); `iter_transitions` turns that into the explicit 4-tuples
the evaluation plan calls out, without duplicating every image on disk.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class EpisodeWriter:
    """Accumulates one episode in memory, then writes it as a single
    compressed `.npz`. Call `add_step` once per environment step, in order,
    then `close()`.
    """

    out_dir: str | Path
    task_name: str
    episode_id: str | None = None
    meta: dict = field(default_factory=dict)

    _images: list = field(default_factory=list, init=False, repr=False)
    _q: list = field(default_factory=list, init=False, repr=False)
    _actions: list = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.episode_id = self.episode_id or f"{self.task_name}_{int(time.time() * 1000)}"

    def add_step(self, image: np.ndarray, q: np.ndarray, action: np.ndarray | None) -> None:
        """`image`/`q` are the state *before* `action` is applied. Call once
        more at episode end with `action=None` to record the final frame."""
        if self._closed:
            raise RuntimeError("episode already closed")
        self._images.append(np.asarray(image))
        self._q.append(np.asarray(q, dtype=np.float32))
        if action is not None:
            self._actions.append(np.asarray(action, dtype=np.float32))

    def close(self) -> Path:
        if self._closed:
            raise RuntimeError("episode already closed")
        if len(self._images) < 2:
            raise ValueError(
                "an episode needs at least 2 frames (one before the final "
                "action=None call) to contain a transition"
            )
        if len(self._actions) != len(self._images) - 1:
            raise ValueError(
                f"expected {len(self._images) - 1} actions for {len(self._images)} "
                f"frames, got {len(self._actions)} — call add_step(..., action=None) "
                "exactly once, at the end"
            )
        self._closed = True

        task_dir = self.out_dir / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / f"{self.episode_id}.npz"
        np.savez_compressed(
            path,
            images=np.stack(self._images),
            q=np.stack(self._q),
            actions=np.stack(self._actions),
            meta=json.dumps({**self.meta, "task": self.task_name, "episode_id": self.episode_id}),
        )
        return path


def load_episode(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {
            "images": f["images"],
            "q": f["q"],
            "actions": f["actions"],
            "meta": json.loads(str(f["meta"])),
        }


def iter_transitions(episode: dict) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (image_t, q_t, action_t, image_{t+1}) for every step of a
    loaded episode (as returned by `load_episode`)."""
    images, q, actions = episode["images"], episode["q"], episode["actions"]
    for t in range(len(actions)):
        yield images[t], q[t], actions[t], images[t + 1]


if __name__ == "__main__":
    # Smoke-test the format with a synthetic scripted episode — no Gazebo
    # needed. Run: `python -m data.collect_transitions /tmp/demo_episodes`
    import sys

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo_episodes"
    rng = np.random.default_rng(0)
    writer = EpisodeWriter(out_dir, task_name="NeedleReach", meta={"seed": 0, "split": "ID"})
    q = np.zeros(6, dtype=np.float32)
    for t in range(10):
        image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        action = None if t == 9 else rng.normal(scale=0.05, size=6).astype(np.float32)
        writer.add_step(image, q, action)
        if action is not None:
            q = q + action
    path = writer.close()
    print(f"wrote {path}")
    n = sum(1 for _ in iter_transitions(load_episode(path)))
    print(f"read back {n} transitions")
