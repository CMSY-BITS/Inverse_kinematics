"""scene_gen.py itself needs Blender (bpy) to run, but the parts that
matter for correctness — that every name export_sdf.py expects is
actually covered, and that the placeholder PSM poses agree with forward
kinematics — are pure and tested here without it."""
import numpy as np
import pytest

import blender.scene_gen as scene_gen
from blender.blenderproc_rerender import psm_link_transforms
from blender.export_sdf import PSM_LINKS, STATIC_MODELS
from kinematics.psm_kinematics import PSMKinematics


def test_every_static_model_has_a_placeholder_spec():
    for model_key, blender_name in STATIC_MODELS.items():
        assert blender_name in scene_gen.STATIC_PROP_SPECS, f"no placeholder spec for {blender_name}"


def test_no_orphan_placeholder_specs():
    # every spec should correspond to a real STATIC_MODELS entry, so a
    # renamed/removed model in export_sdf.py doesn't leave a silently
    # unused spec behind here
    known_names = set(STATIC_MODELS.values())
    for blender_name in scene_gen.STATIC_PROP_SPECS:
        assert blender_name in known_names, f"{blender_name} has no matching STATIC_MODELS entry"


def test_psm_link_transform_count_matches_link_list():
    transforms = psm_link_transforms(np.zeros(6))
    assert len(transforms) == len(PSM_LINKS) - 1  # base link has no joint transform


def test_psm_placeholder_positions_are_finite_and_near_the_robot():
    transforms = psm_link_transforms(np.zeros(6))
    for T in transforms:
        pos = T[:3, 3]
        assert np.all(np.isfinite(pos))
        # PSM link lengths are tens of cm; a placeholder wildly outside
        # that would signal a units or transform-order mistake.
        assert np.linalg.norm(pos) < 1.0

    # and the last link's pose, offset by the tool-tip control point,
    # should reproduce PSMKinematics.forward's own tip position exactly
    # (same consistency check as blender/blenderproc_rerender.py relies on).
    kin = PSMKinematics()
    last = transforms[-1]
    tip_from_link_transform = last[:3, 3] + last[:3, :3] @ np.array([0.0, kin.params.l_yaw2ctrlpnt, 0.0])
    tip_from_forward = kin.forward(np.zeros(6))[:3, 3]
    np.testing.assert_allclose(tip_from_link_transform, tip_from_forward, atol=1e-9)


def test_main_without_bpy_raises_clear_error():
    with pytest.raises(RuntimeError, match="scene_gen.py must be run inside Blender"):
        scene_gen.main(["--out", "/tmp/unused.blend"])
