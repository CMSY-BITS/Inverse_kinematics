"""Frozen V-JEPA 2 encoder wrapper.

Loads a pretrained V-JEPA 2 (or V-JEPA 2-AC) checkpoint and exposes a plain
`encode(images) -> latents` call. The backbone is always frozen
(`requires_grad_(False)`) — everything downstream (AC head, inverse model,
UDE residual, MMD adapter) trains on top of these features rather than
fine-tuning them, per the ablation ladder in the evaluation plan.

The actual V-JEPA 2 model code/weights are not vendored here (they're a few
GB and belong on the GPU workstation, loaded from Meta's released
checkpoint or a local `.pt` path). This wrapper is the integration point:
swap `_load_backbone` for the real constructor once you have the checkpoint
on that machine, and everything else in `models/` and `eval/` is written
against this interface, not against V-JEPA internals directly.
"""
from __future__ import annotations

from pathlib import Path

from ._torch_optional import TORCH_AVAILABLE, TorchModuleBase, nn, require_torch, torch


class VJEPA2Encoder(TorchModuleBase):
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        variant: str = "vjepa2-ac-vitl",
        device: str = "cuda",
        latent_dim: int = 1024,
    ):
        require_torch("VJEPA2Encoder")
        super().__init__()
        self.variant = variant
        self.device = device
        self.latent_dim = latent_dim
        self.backbone = self._load_backbone(checkpoint_path, variant)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _load_backbone(self, checkpoint_path, variant):
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No V-JEPA 2 checkpoint path given. Download the released "
                f"'{variant}' checkpoint onto the GPU workstation and pass "
                "its path here; this wrapper does not fetch weights itself."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"V-JEPA 2 checkpoint not found: {checkpoint_path}")
        # NOTE: replace with the real V-JEPA 2 model constructor
        # (`torch.hub.load(...)` or the vendored model class) once the
        # checkpoint format is on hand; torch.load here is a placeholder
        # that expects a scripted/traced module for now.
        model = torch.load(checkpoint_path, map_location="cpu")
        return model.to(self.device)

    def encode(self, images):
        """images: (B, T, C, H, W) or (B, C, H, W) pixel tensor, already
        normalized per the V-JEPA 2 preprocessing. Returns (B, latent_dim)
        (or (B, T, latent_dim) for a temporal clip, depending on `variant`).

        Not decorated with `@torch.no_grad()` at class-definition time —
        that would evaluate `torch.no_grad` on import and break importing
        this module without torch installed — so the no-grad context is
        entered here instead, reached only after `__init__` has already
        confirmed torch is available.
        """
        with torch.no_grad():
            images = images.to(self.device)
            return self.backbone(images)
