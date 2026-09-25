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
`models/_torch_optional.py` (and `models/_transformers_optional.py` for
`VJEPA2Encoder` specifically, which also needs `transformers`).

## Setup

```bash
pip install -r requirements.txt      # numpy, scipy, pytest — runs anywhere
pip install -e .                     # makes kinematics/models/supervisor/data/eval importable as packages
```

**On the GPU workstation** (the RTX A1000 PC, under WSL2 or native Linux):

```bash
bash scripts/setup_gpu_workstation.sh                  # venv + CUDA-matched torch + transformers + this repo
bash scripts/setup_gpu_workstation.sh --fetch-checkpoint  # also downloads the V-JEPA 2 checkpoint (several GB)
```

Detects the driver's CUDA version from `nvidia-smi` and installs a
matching `torch` build automatically (override with `TORCH_CUDA_INDEX=...`
if it guesses wrong), then `transformers` — V-JEPA 2 ships as
`transformers.VJEPA2Model` — and verifies `torch.cuda.is_available()`.
`models/jepa_wrapper.py:VJEPA2Encoder` defaults to
`facebook/vjepa2-vitl-fpc64-256` (hidden_size=1024, matching
`ac_head.py`/`inverse_model.py`'s defaults); override with
`VJEPA2_CHECKPOINT=...` for a different size or a local checkpoint dir.

**On the ROS 2 + Gazebo machine** (Ubuntu 24.04 under WSL2, per the
evaluation plan's sim stack):

```bash
bash scripts/setup_ros2_gazebo.sh
```

Installs ROS 2 Jazzy, Gazebo Harmonic, `ros_gz`, `ros2_control`, and
`cv_bridge` via apt; installs this repo with `pip install -e .` so
`surg_sim`'s nodes can `import kinematics`/`models`/`supervisor`; then
`colcon build`s `sim/ros2_ws/src/surg_sim` and adds `GZ_SIM_RESOURCE_PATH`
(pointing at `blender/assets/export`) to `~/.bashrc`. Idempotent — safe to
re-run. Afterward:

```bash
ros2 launch surg_sim needle_reach.launch.py   # (also: gauze_retrieve / peg_transfer / needle_pick)
```

Gazebo needs the exported models first — see the Blender section below.

## Tests

```bash
python3 -m pytest
```

Covers PSM forward/inverse kinematics, the RCM safety clamp, the CEM
planner (against a known quadratic optimum), all of `eval/metrics.py`
(bootstrap CIs, Wilcoxon, ECE, latency/RCM stats), the UDE physics term,
the MMD estimator, the episode logger round-trip, the Jev gate's rate
limiting/timeout/fallback behavior and its data-sanitization whitelist,
`VJEPA2Encoder`'s frame-clip normalization, and the closed-loop runner
against a toy env. None of it needs a GPU, ROS 2, Gazebo, or Blender.

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
scripts/                    setup_gpu_workstation.sh, setup_ros2_gazebo.sh
tests/                      unit tests for everything in the left three columns above
```

## Blender → Gazebo asset pipeline

There's no modeled `.blend` scene in this repo yet — `blender/scene_gen.py`
builds a placeholder one from primitives (boxes/cylinders, correctly named
and roughly posed) so the rest of the pipeline runs end-to-end before real
art exists:

```bash
blender --background --python blender/scene_gen.py -- \
    --out blender/assets/surgical_scene.blend

blender --background blender/assets/surgical_scene.blend --python blender/export_sdf.py -- \
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
