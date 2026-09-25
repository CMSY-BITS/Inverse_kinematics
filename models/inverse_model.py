"""Distilled inverse model — ablation rung A1.

CEM (A0) is accurate but runs at planner rate (5-10 Hz per the eval plan's
real-time loop, §4). This model is trained by behavior cloning on
(z_current, z_goal) -> action pairs logged from CEM's own outputs, so it
answers in a single forward pass at >=30 Hz. The closed-loop evaluator
should fall back to CEM when this model's predicted action would move the
tool further from the goal than the previous step (a cheap, no-labels OOD
signal), not just always trust it.
"""
from __future__ import annotations

from ._torch_optional import TorchModuleBase, nn, require_torch


class InverseModel(TorchModuleBase):
    def __init__(self, latent_dim: int = 1024, action_dim: int = 7, hidden_dim: int = 512):
        require_torch("InverseModel")
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # bounded output; scale to actual joint-delta limits downstream
        )

    def forward(self, z_current, z_goal):
        """z_current, z_goal: (B, latent_dim) -> action: (B, action_dim) in [-1, 1],
        meant to be rescaled to the PSM's per-joint delta limits before use.
        """
        import torch

        x = torch.cat([z_current, z_goal], dim=-1)
        return self.net(x)
