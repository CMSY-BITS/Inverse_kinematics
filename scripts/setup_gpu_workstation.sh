#!/usr/bin/env bash
# Sets up the GPU training workstation: a Python venv, CUDA-matched torch,
# transformers (for the real V-JEPA 2 checkpoint — see
# models/jepa_wrapper.py), and this repo installed editable.
#
# Run ON the machine with the GPU (e.g. under WSL2 on the PC with the RTX
# A1000), from the repo root:
#   bash scripts/setup_gpu_workstation.sh
#
# Then, separately (downloads several GB, so it's not automatic here):
#   bash scripts/setup_gpu_workstation.sh --fetch-checkpoint
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV_DIR="${VENV_DIR:-.venv}"
CHECKPOINT="${VJEPA2_CHECKPOINT:-facebook/vjepa2-vitl-fpc64-256}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

log "Checking for an NVIDIA GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. On WSL2: install the Windows NVIDIA driver" \
         "on the Windows side (not inside WSL2) — WSL2 uses it directly." \
         "See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Map the driver's supported CUDA version to a PyTorch wheel index. This
# is a coarse map (pytorch.org adds new cuXXX indexes over time — check
# https://pytorch.org/get-started/locally/ if this drifts, or set
# TORCH_CUDA_INDEX yourself to skip the detection entirely).
if [ -n "${TORCH_CUDA_INDEX:-}" ]; then
    CUDA_INDEX="$TORCH_CUDA_INDEX"
else
    DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_MAJOR=$(nvidia-smi -q | grep -m1 "CUDA Version" | grep -oE '[0-9]+\.[0-9]+' | cut -d. -f1)
    case "${CUDA_MAJOR:-12}" in
        12) CUDA_INDEX="https://download.pytorch.org/whl/cu124" ;;
        11) CUDA_INDEX="https://download.pytorch.org/whl/cu118" ;;
        *)  CUDA_INDEX="https://download.pytorch.org/whl/cu124" ;;  # newest-supported guess
    esac
    log "Driver reports CUDA $DRIVER_CUDA -> using $CUDA_INDEX (override with TORCH_CUDA_INDEX=... if wrong)"
fi

log "Creating venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

log "Installing torch (CUDA build) — this is the large download"
pip install torch --index-url "$CUDA_INDEX"

log "Installing transformers (ships transformers.VJEPA2Model) and this repo's own deps"
pip install transformers accelerate
pip install -r requirements.txt
pip install -e .

log "Verifying torch sees the GPU"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("total memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    raise SystemExit(
        "torch was installed but cannot see a GPU. Under WSL2, confirm the "
        "Windows-side NVIDIA driver is installed (not a Linux driver inside "
        "WSL2) and that this shell is actually running under WSL2, not a "
        "plain Linux VM: `uname -r` should mention 'microsoft'."
    )
PY

if [ "${1:-}" = "--fetch-checkpoint" ]; then
    log "Downloading V-JEPA 2 checkpoint: $CHECKPOINT (several GB, cached under ~/.cache/huggingface)"
    python - <<PY
from models.jepa_wrapper import VJEPA2Encoder
enc = VJEPA2Encoder(checkpoint="$CHECKPOINT", device="cuda")
print("loaded:", enc.checkpoint, "hidden_size:", enc.latent_dim)
PY
else
    log "Skipping checkpoint download (pass --fetch-checkpoint to fetch $CHECKPOINT now)"
fi

log "Done. Activate this environment in new shells with: source $VENV_DIR/bin/activate"
