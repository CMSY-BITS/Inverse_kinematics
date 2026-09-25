#!/usr/bin/env bash
# Sets up ROS 2 Jazzy + Gazebo Harmonic + ros2_control under WSL2 Ubuntu
# 24.04 (Noble) — matches docs/evaluation_plan.md's sim stack — then
# colcon-builds sim/ros2_ws/src/surg_sim.
#
# Run ON the WSL2 Ubuntu instance, from the repo root:
#   bash scripts/setup_ros2_gazebo.sh
#
# Idempotent: safe to re-run (skips steps that are already done).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

# ROS 2's own setup.bash files reference internal variables (e.g.
# AMENT_TRACE_SETUP_FILES) without guarding them first, which trips this
# script's `set -u` (nounset). That's expected of ROS 2's scripts, not a
# sign anything is broken — disable nounset just around sourcing them.
source_ros_setup() {
    set +u
    # shellcheck source=/dev/null
    source "$1"
    set -u
}

log "Checking environment"
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Warning: /proc/version doesn't mention 'microsoft' — this doesn't" \
         "look like WSL2. Continuing anyway (this script also works on a" \
         "plain Ubuntu 24.04 machine)." >&2
fi
. /etc/os-release
if [ "${VERSION_CODENAME:-}" != "noble" ]; then
    echo "This targets Ubuntu 24.04 (noble) for ROS 2 Jazzy; found" \
         "'${VERSION_CODENAME:-unknown}'. Aborting rather than guess at a" \
         "different ROS distro's package names." >&2
    exit 1
fi

log "Setting a UTF-8 locale (ROS 2 requirement)"
sudo apt update -qq
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

log "Enabling the universe repo + adding the ROS 2 apt source"
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
    ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -oP '"tag_name":\s*"\K[^"]+')
    curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${VERSION_CODENAME}_all.deb"
    sudo apt install -y /tmp/ros2-apt-source.deb
else
    log "ros2-apt-source already installed, skipping"
fi
sudo apt update -qq
sudo apt upgrade -y -qq

log "Installing ROS 2 Jazzy (desktop) + dev tools"
sudo apt install -y ros-jazzy-desktop ros-dev-tools

log "Installing Gazebo Harmonic + ros_gz bridge + ros2_control stack"
sudo apt install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-cv-bridge

log "Initializing rosdep"
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

log "Adding ROS 2 sourcing to ~/.bashrc (idempotent)"
grep -qxF "source /opt/ros/jazzy/setup.bash" ~/.bashrc || \
    echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source_ros_setup /opt/ros/jazzy/setup.bash

log "Installing pip (Ubuntu 24.04's base image doesn't ship it for python3)"
sudo apt install -y python3-pip

log "Installing this repo's own Python packages (surg_sim imports kinematics/models/...)"
python3 -m pip install --break-system-packages -e "$REPO_ROOT" || \
    python3 -m pip install -e "$REPO_ROOT"

log "Resolving surg_sim's rosdep dependencies"
cd "$REPO_ROOT/sim/ros2_ws"
rosdep install --from-paths src --ignore-src -r -y

log "colcon build"
# --symlink-install links install/share/surg_sim/{worlds,launch,config}
# back to the source tree instead of copying it, so editing a world/
# launch/config file (or reinstalling this repo's Python packages) takes
# effect immediately -- no rebuild needed except after adding/removing a
# file or changing setup.py's entry_points.
colcon build --packages-select surg_sim --symlink-install
source_ros_setup install/setup.bash

log "Adding the surg_sim workspace overlay to ~/.bashrc (idempotent)"
WORKSPACE_SOURCE_LINE="source \"$REPO_ROOT/sim/ros2_ws/install/setup.bash\""
grep -qxF "$WORKSPACE_SOURCE_LINE" ~/.bashrc || echo "$WORKSPACE_SOURCE_LINE" >> ~/.bashrc

log "Pointing Gazebo at the Blender-exported models (once you've run blender/export_sdf.py)"
GZ_RESOURCE_LINE="export GZ_SIM_RESOURCE_PATH=\"$REPO_ROOT/blender/assets/export:\${GZ_SIM_RESOURCE_PATH:-}\""
grep -qxF "$GZ_RESOURCE_LINE" ~/.bashrc || echo "$GZ_RESOURCE_LINE" >> ~/.bashrc

log "Done. Open a new shell (or 'source ~/.bashrc') and try:"
echo "    ros2 launch surg_sim needle_reach.launch.py"
