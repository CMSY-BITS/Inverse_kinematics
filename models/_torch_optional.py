"""Shared guarded-torch import.

This project is written on a GPU-less cloud container and trained on a
16 GB GPU workstation, so every module that needs torch imports it through
here instead of failing at import time. `TORCH_AVAILABLE` is False on a
machine without torch (or without a GPU build); any class that actually
needs torch raises a clear `ImportError` from its `__init__`, not from
`import models...`, so the rest of the package (kinematics, CEM planner,
metrics, tests) stays usable without it.
"""
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on the cloud container
    torch = None
    nn = None
    TORCH_AVAILABLE = False

# Base class for torch modules in this package: `nn.Module` when torch is
# installed, plain `object` otherwise. Subclasses must call
# `require_torch(ClassName)` as the first line of `__init__`.
TorchModuleBase = nn.Module if TORCH_AVAILABLE else object


def require_torch(class_name: str) -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"{class_name} needs torch, which isn't installed on this machine. "
            "This module is meant to run on the GPU training workstation — "
            "see requirements.txt (`pip install torch`) — not the cloud "
            "scaffolding container."
        )
