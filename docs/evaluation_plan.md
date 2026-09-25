# JEPA Visual Inverse Kinematics for Surgical Robotics — Evaluation Plan

**Robot:** dVRK PSM (7-DoF including jaw) &nbsp;·&nbsp; **GPU (training machine):** RTX A1000 16 GB
**Sim stack:** ROS 2 Jazzy · Gazebo Harmonic · WSL2 &nbsp;·&nbsp; **Backbone:** V-JEPA 2-AC (frozen)

How we benchmark a V-JEPA 2 world model for image-conditioned inverse kinematics on
the da Vinci Research Kit patient-side manipulator: datasets, baselines, the
UDE/MMD/Jev extensions, and the Gazebo + Blender simulation pipeline with a
real-time control budget.

---

## 1. Benchmark datasets

| Dataset | Role | Split / protocol |
|---|---|---|
| **Our Gazebo dVRK tasks** — `NeedleReach, GauzeRetrieve, PegTransfer, NeedlePick` | Primary closed-loop benchmark with exact joint + tip ground truth | ID split plus OOD split (new lighting, tissue textures, camera pose ±10°, tool wear) |
| SurRoL | Cross-sim check and RL baseline source | Same four task families where available |
| JIGSAWS | Real kinematics + video, offline IK regression | LOSO and LOUO |
| ROSMA | Real dVRK kinematics with video | Held-out users |
| SAR-RARP50 | Real clinical video, perception robustness | Official split; perception metrics only |
| EndoVis 17 / 18 | Instrument segmentation and pose, feature probing | Official split |
| Blender / BlenderProc re-renders | Photoreal domain gap study | Paired sim-raw vs re-rendered frames of the same trajectories |

## 2. Baselines and state of the art

| Family | Methods | What it tests |
|---|---|---|
| Classical oracle | Damped-least-squares IK / PBVS, IBVS | Upper bound with known pose; image-space servoing without learning |
| Learned IK | MLP-IK, IKFlow | Pose-to-joints regression without vision |
| Imitation | BC, ACT, Diffusion Policy | End-to-end visuomotor policies |
| World models / RL | DINO-WM, TD-MPC2, DreamerV3, SurRoL RL | Planning with learned latent dynamics |

### Ours as an ablation ladder

- **A0 · V-JEPA 2-AC + CEM** — latent planning toward a goal image
- **A1 · + distilled inverse model** — one-shot action at ≥30 Hz, CEM as fallback
- **A2 · + UDE residual** — physics model plus a neural residual for cable hysteresis
- **A3 · + MMD adapter** — domain alignment on frozen V-JEPA features
- **A4 · + Jev supervisor** — async plan checking at ≤2 Hz

### Metrics

| Axis | Metrics |
|---|---|
| Precision | Tip position error (mm), orientation error (deg), success = tip error ≤ 3 mm, RCM deviation (mm) |
| Latency | p50 / p95 / p99 per stage, achieved control rate, deadline misses, sim real-time factor |
| Efficiency | Peak GPU memory, energy per episode |
| Safety | Collisions, RCM violations, peak contact force |
| Supervisor | Jev flag accuracy, expected calibration error (ECE) |
| Robustness | Relative drop from ID to OOD |
| Statistics | 3 seeds × 100 episodes per task, bootstrap 95% CIs, Wilcoxon signed-rank vs strongest baseline |

## 3. Universal differential equations, MMD and Jev

**UDE residual.** Known PSM kinematics and a cable-drive model form the ODE; a
small network learns the residual that the physics misses, mainly hysteresis
and backlash. Trained on sim transitions first, then fine-tuned on dVRK logs.

**MMD adapter.** A light adapter on top of the frozen V-JEPA encoder, trained
with a maximum mean discrepancy loss between sim, re-rendered and real
feature distributions. The backbone stays frozen.

**Jev stays out of the control loop.** The Jev cloud API (TypeSafe System
One) answers in about 230–350 ms. It runs as an asynchronous supervisor at
≤2 Hz that checks plans and can veto or slow the planner. A locally distilled
supervisor is the fallback when the network is slow or down. No patient data
leaves the machine.

## 4. Simulation pipeline

**Build pipeline:** Blender assets → Gazebo worlds → transitions (+ BlenderProc
re-render) → training → closed-loop eval, with failure cases fed back into new
data collection.

**Real-time loop and node rates:**

| Node | Rate / budget |
|---|---|
| `gz_sim` + `ros2_control` (joint servo) | 1 kHz |
| Perception (V-JEPA encode) | 30 Hz |
| Planner — CEM, or distilled inverse model | 5–10 Hz (CEM) or ≥30 Hz (inverse model) |
| IK + RCM filter (DLS IK, RCM clamp) | ≤ 2 ms |
| Jev supervisor (async, local fallback) | ≤ 2 Hz, never blocks the planner |

**Planner budget: 100 ms per CEM step**

- Encode: 30–45 ms, FP16 TensorRT
- CEM: 30–40 ms, N=128, H=3, 3 iterations
- Slack: IK, RCM, ROS messaging

### Steps

1. Model PSM, tissue pad, needle and gauze in Blender; export SDF/URDF with collision meshes and matching PBR materials.
2. Build the four task worlds in Gazebo Harmonic, driven by ROS 2 Jazzy `ros2_control` at 1 kHz under WSL2.
3. Collect transitions `(image, q, action, next image)` with scripted and noisy-expert policies, logged to disk as episodes.
4. Re-render the same trajectories photoreal with BlenderProc to create paired sim/photoreal frames.
5. Train the action-conditioned head, inverse model, UDE residual and MMD adapter on the 16 GB A1000.
6. Run closed-loop evaluation on ID and OOD splits and log every metric in §2.

## 5. Code scaffold

| Path | Contents |
|---|---|
| `docs/` | This plan, protocols, metric definitions |
| `sim/ros2_ws/src/surg_sim` | Gazebo worlds, launch files, controllers, task nodes |
| `blender/` | Asset sources, export and BlenderProc re-render scripts |
| `models/` | V-JEPA wrapper, AC head, inverse model, UDE, MMD adapter |
| `kinematics/` | PSM forward/inverse kinematics, RCM constraint |
| `supervisor/jev_gate.py` | Async Jev client with timeout and local fallback |
| `data/collect_transitions.py` | Episode logger |
| `eval/` | Closed-loop runner, metrics, bootstrap CIs, Wilcoxon |
| `tests/` | Unit tests for IK, RCM clamp, metrics and the gate |

---

*Live page with diagrams: https://claude.ai/artifact/AMo9Bq3UBn3Nb6rW6uc8RV*
*This copy was saved locally so it travels with the code scaffold rather than staying artifact-only.*
