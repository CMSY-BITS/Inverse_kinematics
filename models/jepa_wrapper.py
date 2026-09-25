"""Frozen V-JEPA 2 encoder wrapper.

Loads a pretrained V-JEPA 2 checkpoint via Hugging Face `transformers`
(`transformers.VJEPA2Model`, which ships the encoder + predictor from
Meta's release) and exposes a plain `encode(frames) -> latents` call. The
backbone is always frozen — everything downstream (AC head, inverse model,
UDE residual, MMD adapter) trains on top of these features rather than
fine-tuning them, per the ablation ladder in the evaluation plan.

Only the *encoder* half is used here (`VJEPA2Model.get_vision_features`,
via `skip_predictor=True` internally); Meta's action-conditioned predictor
(V-JEPA 2-AC) was trained on its own action space, not the PSM's 7-DoF
joint deltas, so `models/ac_head.py` trains a small predictor of our own on
top of these frozen features instead of reusing Meta's AC predictor
directly.

V-JEPA 2 is a *video* model (`tubelet_size=2` in its default config: it
patches pairs of consecutive frames together), so `encode` takes a short
clip, not a single image. Feed it the last 2+ frames from a rolling
buffer in the perception node for real motion information; a single frame
is accepted too (duplicated internally) as a degraded fallback, e.g. at
the very start of an episode before a buffer has filled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._torch_optional import TorchModuleBase, require_torch, torch
from ._transformers_optional import require_transformers, transformers

DEFAULT_CHECKPOINT = "facebook/vjepa2-vitl-fpc64-256"  # hidden_size=1024, matches ac_head/inverse_model defaults


class VJEPA2Encoder(TorchModuleBase):
    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        local_files_only: bool = False,
    ):
        """`checkpoint`: a Hugging Face Hub model id (downloaded and cached
        automatically, e.g. the default) or a local directory containing a
        checkpoint saved with `model.save_pretrained(...)`. Requires both
        torch and transformers — see `requirements.txt`.
        """
        require_torch("VJEPA2Encoder")
        require_transformers("VJEPA2Encoder")
        super().__init__()
        self.device = device
        self.checkpoint = str(checkpoint)

        self.model = transformers.VJEPA2Model.from_pretrained(
            self.checkpoint, local_files_only=local_files_only
        )
        self.processor = transformers.AutoVideoProcessor.from_pretrained(
            self.checkpoint, local_files_only=local_files_only
        )
        self.latent_dim = self.model.config.hidden_size

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode(self, frames) -> np.ndarray:
        """`frames`: one clip (a single HxWx3 frame, or a list of them,
        oldest-to-newest). Returns one pooled feature vector, `(latent_dim,)`.
        """
        return self.encode_batch([frames])[0]

    def encode_batch(self, clips: list) -> np.ndarray:
        """`clips`: a list of clips (each a single frame or a list of
        frames — see `encode`). Returns `(B, latent_dim)`, one pooled
        vector per clip; batching like this lets the CEM cost function
        (`models/cem_planner.py`) encode a whole population of rollouts in
        one forward pass instead of one at a time.
        """
        with torch.no_grad():
            clips = [self._as_frame_list(c) for c in clips]
            pixel_values = self.processor(clips, return_tensors="pt")["pixel_values_videos"].to(self.device)
            tokens = self.model.get_vision_features(pixel_values)  # (B, N, latent_dim)
            pooled = tokens.mean(dim=1)  # (B, latent_dim)
        return pooled.detach().cpu().numpy()

    @staticmethod
    def _as_frame_list(frames):
        # Normalize one clip to a list of frames. A lone frame is
        # duplicated so tubelet_size=2 patching still has a pair to work
        # with; it carries no motion information until a real buffer of
        # >=2 distinct frames is available.
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            return [frames, frames]
        frames = list(frames)
        if len(frames) == 1:
            frames = frames * 2
        return frames
