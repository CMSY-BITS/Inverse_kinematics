"""Placeholder scene generator.

Builds a minimal `.blend` file with objects named exactly as
`export_sdf.py` expects (`STATIC_MODELS`, `PSM_LINKS`), so the export ->
Gazebo pipeline has something real to run end-to-end before actual
modeled assets exist. Every object is a plain box or cylinder, sized
roughly to real dimensions and, for the PSM, posed at the DH chain's q=0
configuration via `kinematics.psm_kinematics` — not meant to look right,
only to exist at about the right place with the right name. Replace
object-by-object with real geometry later; nothing downstream cares what
an object is made of, only that its name and rough pose match.

Run:
    blender --background --python blender/scene_gen.py -- \\
        --out blender/assets/surgical_scene.blend
Then export it to Gazebo models with export_sdf.py (see README.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import bpy

    BPY_AVAILABLE = True
except ImportError:  # running outside Blender, e.g. for linting/tests
    bpy = None
    BPY_AVAILABLE = False

# Blender's `--python <path>` puts this script's own directory (blender/)
# on sys.path, not the repo root — so the repo-root-relative imports below
# (`blender.X`, reaching into this script's own containing package) would
# otherwise fail with "No module named 'blender'" regardless of the
# invoking shell's cwd or PYTHONPATH. Insert the repo root explicitly so
# this runs the same way from anywhere.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from blender.blenderproc_rerender import psm_link_transforms
from blender.export_sdf import PSM_LINKS, STATIC_MODELS

# name -> ("box", (x,y,z) size) | ("cylinder", radius, depth), plus a
# placement roughly matching worlds/needle_reach.sdf's own poses.
STATIC_PROP_SPECS = {
    "TissuePad": ("box", (0.15, 0.10, 0.01), (0, 0, -0.005)),
    "Needle": ("cylinder", (0.0005, 0.02), (0.04, 0.01, 0.005)),
    "GauzePatch": ("box", (0.03, 0.03, 0.002), (-0.03, 0.04, 0.001)),
    "PegBoard": ("box", (0.10, 0.10, 0.01), (0, 0, -0.005)),
    "TransferRing": ("cylinder", (0.006, 0.004), (0.02, -0.03, 0.02)),
    "SutureTarget": ("box", (0.01, 0.01, 0.001), (-0.02, -0.02, 0.001)),
    "EndoscopeCameraHousing": ("cylinder", (0.006, 0.03), (0, 0, 0.35)),
}


def _require_bpy() -> None:
    if not BPY_AVAILABLE:
        raise RuntimeError(
            "scene_gen.py must be run inside Blender: "
            "`blender --background --python blender/scene_gen.py -- --out <file.blend>`"
        )


def _clear_default_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _add_box(name: str, size_xyz: tuple, location: tuple) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)  # unit cube; scale sets final size
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size_xyz


def _add_cylinder(name: str, radius: float, depth: float, location: tuple) -> None:
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location)
    bpy.context.active_object.name = name


def build_psm_links() -> None:
    """One small placeholder cylinder per PSM link, positioned at the DH
    chain's q=0 pose (the same `psm_link_transforms` used at inference
    time in `blender/blenderproc_rerender.py`, so this placeholder starts
    from the same zero-pose the rest of the code assumes)."""
    base_link_name = PSM_LINKS[0][1]
    _add_cylinder(base_link_name, radius=0.008, depth=0.03, location=(0, 0, -0.015))

    transforms = psm_link_transforms(np.zeros(6))
    for (_, blender_name, *_rest), T in zip(PSM_LINKS[1:], transforms):
        # A stub marker at each joint's zero-pose origin, not an oriented
        # connecting rod — good enough for export_sdf.py's link poses and
        # for a crude visual sanity check in Gazebo, nothing more.
        _add_cylinder(blender_name, radius=0.004, depth=0.02, location=tuple(T[:3, 3]))


def build_static_props() -> None:
    for model_key, blender_name in STATIC_MODELS.items():
        kind, dims, location = STATIC_PROP_SPECS[blender_name]
        if kind == "box":
            _add_box(blender_name, dims, location)
        else:
            radius, depth = dims
            _add_cylinder(blender_name, radius, depth, location)
        print(f"placed {model_key} ({blender_name})")


def main(argv: list[str] | None = None) -> None:
    _require_bpy()
    argv = argv if argv is not None else sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output .blend file path")
    args = parser.parse_args(argv)

    _clear_default_scene()
    build_psm_links()
    build_static_props()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    print(f"wrote placeholder scene -> {out_path}")


if __name__ == "__main__":
    main()
