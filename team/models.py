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


class DeepSetPlus(nn.Module):
    """DeepSet with event-level features concatenated *after* pooling.

    phi() runs per particle and is mean-pooled to a fixed vector; the 11
    event-level quantities (HT, leading-4 pT, n_cand, |dxy| summaries, m2, m4)
    are appended to that pooled vector before rho().  Injecting them after the
    pool is what makes them cheap in firmware: they are computed once per event
    from quantities an L1 trigger already has, and they do not multiply into the
    16x-replicated phi() block.

    rho_dims is the full list of hidden widths; the final Linear(-, 1) is added
    on top, so rho_dims=(256, 128) means 256 -> 128 -> 1.
    """

    def __init__(self, n_features: int = 5, n_event_features: int = 11,
                 phi_dims=(64, 32, 16), rho_dims=(256, 128),
                 dropout: float = 0.0, pool: str = "mean",
                 use_event_features: bool = True, event_scale: float = 1.0,
                 pool_norm: bool = False):
        super().__init__()
        self.pool = pool
        self.use_event_features = use_event_features
        self.n_event_features = n_event_features if use_event_features else 0
        # The mean-pooled phi output lands around |h| ~ 0.11 while standardized
        # event features sit at |f| ~ 0.67, so a naive concat lets the event
        # features drive rho ~5x harder and the phi branch never trains properly.
        # Either knob fixes the imbalance; pool_norm is a BatchNorm1d, which at
        # inference is a fixed per-channel affine and folds into the neighbouring
        # Linear, so it costs nothing in firmware.
        self.event_scale = event_scale
        self.pool_norm = pool_norm

        layers, d = [], n_features
        for i, h in enumerate(phi_dims):
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0 and i < len(phi_dims) - 1:
                layers += [nn.Dropout(dropout)]
            d = h
        self.phi = nn.Sequential(*layers)
        # "meanmax" concatenates mean- and max-pooling, doubling the pooled width.
        # Max over particles is comparators only -- no DSP -- so it stays cheap in
        # firmware, but it is an extra hls4ml layer, so keep it for the teacher.
        self.pooled_dim = 2 * d if pool == "meanmax" else d
        self.norm = nn.BatchNorm1d(self.pooled_dim) if pool_norm else nn.Identity()

        layers, d = [], self.pooled_dim + self.n_event_features
        for i, h in enumerate(rho_dims):
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0 and i < len(rho_dims) - 1:
                layers += [nn.Dropout(dropout)]
            d = h
        self.rho = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        h = self.phi(x)
        if self.pool == "mean":
            h = h.mean(dim=1)
        elif self.pool == "sum":
            h = h.sum(dim=1)
        elif self.pool == "meanmax":
            # phi output is post-ReLU, so max over zero-padded slots is harmless
            h = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        else:
            raise ValueError(f"unknown pool {self.pool!r}")
        h = self.norm(h)
        if self.use_event_features:
            if f is None:
                raise ValueError("model was built with use_event_features=True but got f=None")
            h = torch.cat([h, f * self.event_scale], dim=1)
        return self.out(self.rho(h)).squeeze(-1)


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


MODELS = {"deepset": DeepSet, "deepset_plus": DeepSetPlus, "mlp": SmallMLP}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
