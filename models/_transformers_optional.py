"""Guarded `transformers` import, parallel to `_torch_optional.py`.

Kept separate from the torch guard because a machine can have torch (e.g.
for the other model classes' torch-only paths) without having transformers
installed, and the error message should say which one is actually missing.
"""
try:
    import transformers

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on the cloud container
    transformers = None
    TRANSFORMERS_AVAILABLE = False


def require_transformers(class_name: str) -> None:
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            f"{class_name} needs the `transformers` package (V-JEPA 2 ships "
            "as transformers.VJEPA2Model as of transformers>=4.55). "
            "pip install transformers on the GPU training workstation — see "
            "requirements.txt."
        )
