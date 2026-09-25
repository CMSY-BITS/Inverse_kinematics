"""Action-conditioned predictor head ("the -AC in V-JEPA 2-AC").

Predicts the next latent given the current latent and a candidate action:
`z_{t+1} = f(z_t, a_t)`. This is what the CEM planner (A0) rolls forward
over the horizon to score action sequences by predicted distance to a goal
latent, entirely in latent space — no image decoding needed at plan time.

A small residual MLP is enough here: V-JEPA 2's own predictor already does
the heavy temporal modeling, so this head only needs to fold in the
7-DoF PSM action on top of it.
"""
from __future__ import annotations

from ._torch_optional import TorchModuleBase, nn, require_torch


class ACHead(TorchModuleBase):
    def __init__(self, latent_dim: int = 1024, action_dim: int = 7, hidden_dim: int = 512):
        require_torch("ACHead")
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, a_t):
        """z_t: (B, latent_dim), a_t: (B, action_dim) -> z_{t+1}: (B, latent_dim).
        Residual by construction: the network predicts the *change* in
        latent, since most of the frame is static (tissue, background) and
        only the tool moves.
        """
        x = self._cat(z_t, a_t)
        delta = self.net(x)
        return z_t + delta

    def rollout(self, z0, actions):
        """z0: (B, latent_dim), actions: (B, H, action_dim) -> (B, H, latent_dim)
        predicted latents, one per planning step. Used by the CEM cost
        function so a whole population of candidate action sequences can be
        scored with `H` sequential (but batched-over-samples) forward passes.
        """
        b, h, _ = actions.shape
        preds = []
        z = z0
        for t in range(h):
            z = self.forward(z, actions[:, t, :])
            preds.append(z)
        return self._stack(preds, dim=1)

    @staticmethod
    def _cat(a, b):
        import torch  # safe: only reached once __init__ has confirmed torch is present

        return torch.cat([a, b], dim=-1)

    @staticmethod
    def _stack(tensors, dim):
        import torch

        return torch.stack(tensors, dim=dim)
