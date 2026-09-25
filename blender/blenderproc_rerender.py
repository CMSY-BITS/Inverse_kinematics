"""Photoreal re-render pass — evaluation plan §4 step 4.

Takes episodes collected in Gazebo (`data/collect_transitions.py`, plain
rasterized frames) and re-renders the same trajectories with BlenderProc's
physically-based renderer and HDRI lighting, producing paired
(sim_frame, photoreal_frame) data for the domain-gap study and for
training the MMD adapter (`models/mmd_adapter.py`), which is trained
precisely to make V-JEPA features agree between these two versions of the
same frame.

Run with BlenderProc's own launcher, not plain python:
    blenderproc run blender/blenderproc_rerender.py \\
        --episodes-dir data/episodes/NeedleReach \\
        --assets-dir blender/assets/export \\
        --out-dir data/episodes_photoreal/NeedleReach \\
        --hdri path/to/operating_room.hdr

Each output episode directory mirrors the input's frame count and gets a
`pairing.json` mapping frame index -> {sim_frame, photoreal_frame} so a
data loader can pull matched pairs without re-deriving the correspondence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import blenderproc as bproc

    BLENDERPROC_AVAILABLE = True
except ImportError:  # running outside `blenderproc run`, e.g. for linting/tests
    bproc = None
    BLENDERPROC_AVAILABLE = False

from blender.export_sdf import PSM_LINKS
from data.collect_transitions import load_episode
from kinematics.psm_kinematics import PSMKinematics, _dh_transform  # noqa: F401 (see comment below)

# `_dh_transform` is psm_kinematics.py's package-private per-joint DH
# transform builder. Reused directly here (rather than reimplemented) so
# the per-link poses this script sets in Blender and the tip pose
# `PSMKinematics.forward` computes for IK/planning can never drift apart.


def _require_blenderproc():
    if not BLENDERPROC_AVAILABLE:
        raise RuntimeError(
            "blenderproc_rerender.py must be run with `blenderproc run "
            "blender/blenderproc_rerender.py -- <args>`, not plain python."
        )


def psm_link_transforms(q: np.ndarray, params=None) -> list[np.ndarray]:
    """Cumulative 4x4 transforms for each of the 6 PSM links at joint
    vector `q`, in the same order as `export_sdf.PSM_LINKS[1:]` (the base
    link is the identity/world-mounted frame and isn't included)."""
    params = params or PSMKinematics().params
    q1, q2, q3, q4, q5, q6 = q
    dh = [
        (np.pi / 2, 0.0, 0.0, q1 + np.pi / 2),
        (-np.pi / 2, 0.0, 0.0, q2 - np.pi / 2),
        (np.pi / 2, 0.0, q3 - params.l_rcc, 0.0),
        (0.0, 0.0, params.l_tool, q4),
        (-np.pi / 2, 0.0, 0.0, q5 - np.pi / 2),
        (-np.pi / 2, params.l_pitch2yaw, 0.0, q6 - np.pi / 2),
    ]
    transforms = []
    T = np.eye(4)
    for alpha, a, d, theta in dh:
        T = T @ _dh_transform(alpha, a, d, theta)
        transforms.append(T.copy())
    return transforms


def _load_scene(assets_dir: Path, hdri_path: str | None):
    _require_blenderproc()
    bproc.init()
    for model_dir in sorted(assets_dir.iterdir()):
        glb_files = list((model_dir / "meshes").glob("*.glb"))
        for glb in glb_files:
            bproc.loader.load_obj(str(glb))
    if hdri_path:
        bproc.world.set_world_background_hdr_img(hdri_path)


def _set_psm_pose(q: np.ndarray) -> None:
    """Pose each PSM link object in the loaded scene to match joint vector
    `q`. Assumes link objects were loaded (via `_load_scene`) with the
    names from `export_sdf.PSM_LINKS`."""
    transforms = psm_link_transforms(q)
    link_names = [link[0] for link in PSM_LINKS[1:]]
    all_objs = {obj.get_name(): obj for obj in bproc.object.get_all_mesh_objects()}
    for link_name, T in zip(link_names, transforms):
        obj = all_objs.get(link_name)
        if obj is not None:
            obj.set_local2world_mat(T)


def rerender_episode(episode_path: Path, out_dir: Path) -> None:
    episode = load_episode(episode_path)
    images, q_seq = episode["images"], episode["q"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pairing = {}
    photoreal_frames = []
    for t, q in enumerate(q_seq):
        _set_psm_pose(q)
        data = bproc.renderer.render()
        frame = np.asarray(data["colors"][0])
        photoreal_frames.append(frame)
        pairing[t] = {"sim_frame": t, "photoreal_frame": t}

    np.savez_compressed(
        out_dir / f"{episode_path.stem}_photoreal.npz",
        images=np.stack(photoreal_frames),
        q=q_seq,
        actions=episode["actions"],
        meta=json.dumps({**episode["meta"], "rerender": "blenderproc"}),
    )
    with open(out_dir / f"{episode_path.stem}_pairing.json", "w") as f:
        json.dump(pairing, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    _require_blenderproc()
    argv = argv if argv is not None else sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hdri", default=None)
    args = parser.parse_args(argv)

    _load_scene(args.assets_dir, args.hdri)
    for episode_path in sorted(args.episodes_dir.glob("*.npz")):
        print(f"re-rendering {episode_path.name}")
        rerender_episode(episode_path, args.out_dir)


if __name__ == "__main__":
    main()
