"""Export Blender-authored assets to Gazebo SDF models.

Run inside Blender, not as a plain Python script:
    blender --background assets/surgical_scene.blend --python export_sdf.py -- \\
        --out sim/ros2_ws/src/surg_sim/../../../blender/assets/export

Produces, per exported object/collection, a Gazebo model directory:
    <out>/<name>/model.config
    <out>/<name>/model.sdf
    <out>/<name>/meshes/<name>.glb

`worlds/*.sdf` in the ROS 2 package reference these by `model://<name>`, so
point `GZ_SIM_RESOURCE_PATH` at `<out>` before launching a world (see
README.md). The single-mesh export path below covers the static/rigid
props (tissue_pad, needle, gauze_patch, peg_board, transfer_ring,
suture_target, endoscope_camera housing); `export_psm` is separate because
the PSM is an articulated multi-link model whose joints must match
`kinematics/psm_kinematics.py`'s DH chain and
`sim/.../config/psm_controllers.yaml`'s joint names exactly.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import bpy

    BPY_AVAILABLE = True
except ImportError:  # running outside Blender, e.g. for linting/tests
    bpy = None
    BPY_AVAILABLE = False


# Static props exported as single-mesh models. Maps the model name (used in
# `model://<name>` URIs) to the Blender object or collection name in the
# authored .blend file.
STATIC_MODELS = {
    "tissue_pad": "TissuePad",
    "needle": "Needle",
    "gauze_patch": "GauzePatch",
    "peg_board": "PegBoard",
    "transfer_ring": "TransferRing",
    "suture_target": "SutureTarget",
    "endoscope_camera": "EndoscopeCameraHousing",
}

# PSM links in kinematic order, matching kinematics/psm_kinematics.py's DH
# chain and config/psm_controllers.yaml's joint names. `axis` is the joint's
# rotation/translation axis in the parent link's frame; `joint_type` is
# "revolute" for every joint except the insertion stage.
PSM_LINKS = [
    # (link_name, blender_object_name, joint_name, joint_type, axis)
    ("psm_base", "PSM_Base", None, None, None),
    ("psm_outer_yaw_link", "PSM_OuterYaw", "psm_outer_yaw_joint", "revolute", (0, 0, 1)),
    ("psm_outer_pitch_link", "PSM_OuterPitch", "psm_outer_pitch_joint", "revolute", (1, 0, 0)),
    ("psm_shaft_link", "PSM_Shaft", "psm_insertion_joint", "prismatic", (0, 0, 1)),
    ("psm_tool_roll_link", "PSM_ToolRoll", "psm_tool_roll_joint", "revolute", (0, 0, 1)),
    ("psm_wrist_pitch_link", "PSM_WristPitch", "psm_wrist_pitch_joint", "revolute", (1, 0, 0)),
    ("psm_wrist_yaw_link", "PSM_WristYaw", "psm_wrist_yaw_joint", "revolute", (0, 1, 0)),
]


def _require_bpy():
    if not BPY_AVAILABLE:
        raise RuntimeError(
            "export_sdf.py must be run inside Blender: "
            "`blender --background <file.blend> --python export_sdf.py -- --out <dir>`"
        )


def _export_mesh_glb(object_name: str, out_path: Path) -> None:
    _require_bpy()
    bpy.ops.object.select_all(action="DESELECT")
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise KeyError(f"Blender object '{object_name}' not found in the current file")
    obj.select_set(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out_path), use_selection=True, export_format="GLB")


def _write_model_config(model_dir: Path, name: str) -> None:
    config = ET.Element("model")
    ET.SubElement(config, "name").text = name
    ET.SubElement(config, "version").text = "1.0"
    sdf_el = ET.SubElement(config, "sdf", version="1.9")
    sdf_el.text = "model.sdf"
    ET.SubElement(config, "description").text = (
        f"Auto-exported from Blender by export_sdf.py for the JEPA-IK surgical sim. "
        f"See docs/evaluation_plan.md."
    )
    ET.ElementTree(config).write(model_dir / "model.config", encoding="utf-8", xml_declaration=True)


def _write_static_model_sdf(model_dir: Path, name: str, static: bool = True) -> None:
    sdf = ET.Element("sdf", version="1.9")
    model = ET.SubElement(sdf, "model", name=name)
    ET.SubElement(model, "static").text = "true" if static else "false"
    link = ET.SubElement(model, "link", name=f"{name}_link")

    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag, name=f"{name}_{tag}")
        geometry = ET.SubElement(el, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        ET.SubElement(mesh, "uri").text = f"meshes/{name}.glb"

    if not static:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass").text = "0.01"  # placeholder; recompute from the mesh's real material

    ET.ElementTree(sdf).write(model_dir / "model.sdf", encoding="utf-8", xml_declaration=True)


def export_static_models(out_dir: Path, models: dict[str, str] = STATIC_MODELS) -> None:
    for model_name, blender_object_name in models.items():
        model_dir = out_dir / model_name
        _export_mesh_glb(blender_object_name, model_dir / "meshes" / f"{model_name}.glb")
        _write_model_config(model_dir, model_name)
        _write_static_model_sdf(model_dir, model_name, static=(model_name != "endoscope_camera"))
        print(f"exported {model_name} -> {model_dir}")


def export_psm(out_dir: Path, links: list = PSM_LINKS) -> None:
    """Multi-link, multi-joint SDF model for the PSM. Requires each link
    listed in `PSM_LINKS` to exist as a separate Blender object, already
    posed at the DH chain's zero configuration (q = 0) in the .blend file —
    `kinematics/psm_kinematics.py:PSMKinematics.forward([0]*6)` gives the
    reference pose to check against.
    """
    model_dir = out_dir / "psm"
    meshes_dir = model_dir / "meshes"

    sdf = ET.Element("sdf", version="1.9")
    model = ET.SubElement(sdf, "model", name="psm")
    ET.SubElement(model, "static").text = "false"

    for link_name, blender_name, joint_name, joint_type, axis in links:
        _export_mesh_glb(blender_name, meshes_dir / f"{link_name}.glb")

        link = ET.SubElement(model, "link", name=link_name)
        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag, name=f"{link_name}_{tag}")
            geometry = ET.SubElement(el, "geometry")
            mesh = ET.SubElement(geometry, "mesh")
            ET.SubElement(mesh, "uri").text = f"meshes/{link_name}.glb"
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass").text = "0.05"  # placeholder per link; recompute from calibrated PSM data

        if joint_name is not None:
            parent_name = links[links.index((link_name, blender_name, joint_name, joint_type, axis)) - 1][0]
            joint = ET.SubElement(model, "joint", name=joint_name, type=joint_type)
            ET.SubElement(joint, "parent").text = parent_name
            ET.SubElement(joint, "child").text = link_name
            axis_el = ET.SubElement(joint, "axis")
            ET.SubElement(axis_el, "xyz").text = " ".join(str(a) for a in axis)

    _write_model_config(model_dir, "psm")
    ET.ElementTree(sdf).write(model_dir / "model.sdf", encoding="utf-8", xml_declaration=True)
    print(f"exported psm ({len(links)} links) -> {model_dir}")


def main(argv: list[str] | None = None) -> None:
    _require_bpy()
    # Blender passes its own args before `--`; only parse what follows it.
    argv = argv if argv is not None else sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output directory for exported Gazebo models")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_static_models(out_dir)
    export_psm(out_dir)


if __name__ == "__main__":
    main()
