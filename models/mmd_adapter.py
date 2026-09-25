"""MMD domain adapter — ablation rung A3.

A small residual adapter sits on top of the frozen V-JEPA 2 features and is
trained to minimize maximum mean discrepancy (MMD) between feature
distributions from different domains (sim, BlenderProc re-render, real
dVRK video), so the downstream inverse model/CEM cost see aligned features
regardless of which domain a frame came from. The backbone itself is never
fine-tuned (see `jepa_wrapper.py`).

`compute_mmd` (numpy, Gaussian RBF kernel) is the reference implementation
and is what tests check; `mmd_loss_torch` mirrors it with autograd so it
can actually be backpropagated through the adapter during training.
"""
from __future__ import annotations

import numpy as np

from ._torch_optional import TorchModuleBase, nn, require_torch


def median_heuristic_sigma(x: np.ndarray, y: np.ndarray, max_points: int = 500) -> float:
    """Median pairwise distance across the pooled sample, a standard
    default bandwidth for the RBF kernel used in MMD."""
    rng = np.random.default_rng(0)
    pooled = np.concatenate([x, y], axis=0)
    if len(pooled) > max_points:
        idx = rng.choice(len(pooled), size=max_points, replace=False)
        pooled = pooled[idx]
    d = np.linalg.norm(pooled[:, None, :] - pooled[None, :, :], axis=-1)
    med = np.median(d[d > 0])
    return float(med) if med > 0 else 1.0


def _rbf_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * sigma**2))


def compute_mmd(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Biased V-statistic estimate of MMD^2 between samples `x` (n, d) and
    `y` (m, d) under a Gaussian RBF kernel. 0 when the two feature
    distributions are indistinguishable at this kernel bandwidth; larger
    values indicate a bigger domain gap.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if sigma is None:
        sigma = median_heuristic_sigma(x, y)
    kxx = _rbf_kernel(x, x, sigma).mean()
    kyy = _rbf_kernel(y, y, sigma).mean()
    kxy = _rbf_kernel(x, y, sigma).mean()
    return float(kxx + kyy - 2 * kxy)


class MMDAdapter(TorchModuleBase):
    """Residual adapter: `z' = z + f(z)`, trained with `mmd_loss_torch`
    against a batch of target-domain features (frozen backbone, adapter
    only)."""

    def __init__(self, latent_dim: int = 1024, hidden_dim: int = 256):
        require_torch("MMDAdapter")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z):
        return z + self.net(z)


def mmd_loss_torch(x, y, sigma: float | None = None):
    """Differentiable MMD^2 between two batches of torch tensors, same
    formula as `compute_mmd`. `x` are adapted source-domain features
    (requires grad), `y` are target-domain features (no grad needed)."""
    import torch

    if sigma is None:
        sigma = median_heuristic_sigma(x.detach().cpu().numpy(), y.detach().cpu().numpy())

    def rbf(a, b):
        d2 = torch.cdist(a, b) ** 2
        return torch.exp(-d2 / (2.0 * sigma**2))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()
    return kxx + kyy - 2 * kxy
