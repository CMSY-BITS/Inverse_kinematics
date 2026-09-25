# JEPA Visual Inverse Kinematics — dVRK PSM

V-JEPA 2 world model for image-conditioned inverse kinematics on the da
Vinci Research Kit patient-side manipulator (PSM). Full evaluation
protocol: [`docs/evaluation_plan.md`](docs/evaluation_plan.md).

Robot: dVRK PSM (7-DoF incl. jaw) · Sim: ROS 2 Jazzy + Gazebo Harmonic ·
Training GPU: 16 GB (e.g. RTX A1000) · Backbone: V-JEPA 2-AC (frozen)

## What runs where

This repo was scaffolded on a GPU-less cloud container, so it's split
along a hard line: **pure math/logic**, which needs nothing but numpy/scipy
and is fully unit-tested right here, and **hardware-dependent** code
(GPU training, ROS 2/Gazebo, Blender), which is correct, ready-to-run
source but needs the machine it targets.

| Runs anywhere (this container included) | Needs the GPU workstation | Needs a ROS 2 Jazzy + Gazebo machine | Needs Blender |
|---|---|---|---|
| `kinematics/` — PSM forward/inverse kinematics, RCM constraint | `models/jepa_wrapper.py`, `ac_head.py`, `inverse_model.py`, `mmd_adapter.py` (torch) | `sim/ros2_ws/src/surg_sim/` | `blender/export_sdf.py` |
| `models/cem_planner.py` (pure numpy) | UDE residual's learned half (`models/ude_residual.py`, physics half runs anywhere) | | `blender/blenderproc_rerender.py` |
| `supervisor/jev_gate.py` (stdlib only) | | | |
| `data/collect_transitions.py` | | | |
| `eval/metrics.py`, `eval/closed_loop_runner.py` | | | |

Every `models/` class that needs torch (`ACHead`, `InverseModel`,
`MMDAdapter`, `VJEPA2Encoder`) raises a clear `ImportError` from its own
`__init__` if torch isn't installed, rather than failing on `import
models` — so the rest of the package stays usable without a GPU. See
`models/_torch_optional.py`.

## Setup

```bash
pip install -r requirements.txt      # numpy, scipy, pytest — runs anywhere
pip install -e .                     # makes kinematics/models/supervisor/data/eval importable as packages
```

On the GPU workstation, also: `pip install torch` (see the comment at the
top of `requirements.txt` for the CUDA-specific index URL) and download a
V-JEPA 2 checkpoint for `models/jepa_wrapper.py`.

On the ROS 2 + Gazebo machine (Ubuntu, ROS 2 Jazzy, Gazebo Harmonic — the
evaluation plan assumes this under WSL2 on Windows):

```bash
pip install -e .                                    # this repo's packages, on the PYTHONPATH ros2 run picks up
cd sim/ros2_ws && colcon build --packages-select surg_sim
source install/setup.bash
ros2 launch surg_sim needle_reach.launch.py          # (also: gauze_retrieve / peg_transfer / needle_pick)
```

Gazebo needs the exported models first — see the Blender section below;
point `GZ_SIM_RESOURCE_PATH` at `blender/assets/export`.

## Tests

```bash
python3 -m pytest
```

Covers PSM forward/inverse kinematics, the RCM safety clamp, the CEM
planner (against a known quadratic optimum), all of `eval/metrics.py`
(bootstrap CIs, Wilcoxon, ECE, latency/RCM stats), the UDE physics term,
the MMD estimator, the episode logger round-trip, the Jev gate's rate
limiting/timeout/fallback behavior and its data-sanitization whitelist,
and the closed-loop runner against a toy env. None of it needs a GPU, ROS
2, Gazebo, or Blender.

## Repo layout

```
docs/                       evaluation plan (this is the plan's canonical copy)
kinematics/                 PSM forward/inverse kinematics, RCM constraint clamp
models/                     CEM planner, V-JEPA wrapper, AC head, inverse model,
                             UDE residual, MMD adapter — the A0-A3 ablation ladder
supervisor/jev_gate.py      async Jev (TypeSafe System One) client, <=2 Hz, local fallback
data/collect_transitions.py (image, q, action, next_image) episode logger
eval/                       metrics, closed-loop runner
sim/ros2_ws/src/surg_sim/   Gazebo worlds, launch files, controllers, task nodes
blender/                    asset export to Gazebo SDF, BlenderProc photoreal re-render
tests/                      unit tests for everything in the left three columns above
```

## Blender → Gazebo asset pipeline

```bash
blender --background assets/surgical_scene.blend --python blender/export_sdf.py -- \
    --out blender/assets/export
```

Exports the PSM (as a multi-link, multi-joint SDF model matching
`kinematics/psm_kinematics.py`'s DH chain and
`sim/ros2_ws/src/surg_sim/config/psm_controllers.yaml`'s joint names) and
the static task props (tissue pad, needle, gauze, peg board, etc.) as
Gazebo models. `sim/ros2_ws/src/surg_sim/worlds/*.sdf` reference them by
`model://<name>`.

After collecting episodes in Gazebo (`data/collect_transitions.py`),
re-render them photoreal for the domain-gap study:

```bash
blenderproc run blender/blenderproc_rerender.py \
    --episodes-dir data/episodes/NeedleReach \
    --assets-dir blender/assets/export \
    --out-dir data/episodes_photoreal/NeedleReach \
    --hdri path/to/operating_room.hdr
```

Run both from the repo root (or with the repo root on `PYTHONPATH`) — they
import this repo's `kinematics`/`data`/`blender` packages directly.

## Attribution

Rebuilt into this repo from an earlier local-machine session; see
`docs/evaluation_plan.md` for the full plan this scaffold implements.
