"""Regression test for a real bug: `blender --python blender/scene_gen.py`
(and `blenderproc run blender/blenderproc_rerender.py`) put the script's
OWN directory on sys.path, not the repo root, so their `from blender.X
import ...` self-referential imports failed with `ModuleNotFoundError: No
module named 'blender'` even though `python3 -c "import blender.scene_gen"`
from the repo root (what the other tests do) worked fine — that only
worked because pytest had already put the repo root on sys.path itself,
masking the bug.

This runs each script the same way Blender/BlenderProc actually would —
as a direct script invocation, with the repo root nowhere on `PYTHONPATH`
— to make sure the sys.path bootstrap in each file actually fixes it.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_standalone(relative_script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, relative_script_path, "--out", "/tmp/unused_test_output"],
        cwd=REPO_ROOT,
        env={},  # deliberately empty: no inherited PYTHONPATH to mask the bug
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_scene_gen_self_locates_repo_root_without_pythonpath():
    result = _run_standalone("blender/scene_gen.py")
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    # bpy isn't installed here, so it should get past the imports and fail
    # with our own clear error instead.
    assert "must be run inside Blender" in result.stderr


def test_blenderproc_rerender_self_locates_repo_root_without_pythonpath():
    result = _run_standalone("blender/blenderproc_rerender.py")
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert "must be run with" in result.stderr
