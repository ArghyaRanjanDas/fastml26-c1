"""Models for FastML26 Challenge 1.

Everything here is a *binary* classifier: one logit, signal (HH_4b) vs. the
pooled background.  Sizes are kept small on purpose -- the Friday deliverable
needs the network to fit in one VU9P SLR (~350k LUT / 700k FF / 1900 DSP)
with <=1 us latency, so the intro notebook's 1024-wide MLP is a non-starter.

The DeepSet is the natural trigger architecture: phi() is applied per particle
(one small MLP reused 16x, or unrolled), the sum/mean pool is free in hardware,
and rho() runs once per event.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSet(nn.Module):
    """Baseline DeepSet, same shape as the intro notebook but binary output.

    phi: per-particle 5 -> 64 -> 32 -> 16, mean-pooled over particles
    rho: 16 -> 256 -> 128 -> 32 -> 1
    """

    def __init__(self, n_features: int = 5, phi_dims=(64, 32, 16),
                 rho_dims=(256, 128, 32), dropout: float = 0.1, pool: str = "mean"):
        super().__init__()
        self.pool = pool

        layers, d = [], n_features
        for i, h in enumerate(phi_dims):
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0 and i < len(phi_dims) - 1:
                layers += [nn.Dropout(dropout)]
            d = h
        self.phi = nn.Sequential(*layers)

        layers = []
        for i, h in enumerate(rho_dims):
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0 and i < len(rho_dims) - 1:
                layers += [nn.Dropout(2 * dropout)]
            d = h
        self.rho = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)                       # (B, P, phi_out)
        h = h.mean(dim=1) if self.pool == "mean" else h.sum(dim=1)
        return self.rho(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logit, shape (B,).  Use torch.sigmoid() for a score in [0, 1]."""
        return self.out(self.embed(x)).squeeze(-1)


class SmallMLP(nn.Module):
    """Flattened-input MLP, kept as an FPGA-friendly sanity baseline."""

    def __init__(self, n_particles: int = 16, n_features: int = 5,
                 dims=(64, 32, 16), dropout: float = 0.1):
        super().__init__()
        layers, d = [nn.Flatten()], n_particles * n_features
        for h in dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x)).squeeze(-1)


MODELS = {"deepset": DeepSet, "mlp": SmallMLP}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
