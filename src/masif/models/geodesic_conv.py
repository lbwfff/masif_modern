"""Geodesic (polar-coordinate) convolution in PyTorch.

Faithful port of the ``inference`` method from ``MaSIF_site.py``:

1. A fixed grid of ``n_thetas x n_rhos`` Gaussian kernels is placed on the polar
   coordinates ``(rho, theta)`` of each patch vertex.
2. The kernels are applied for ``n_rotations`` rotations of ``theta``; feature
   aggregation over the patch yields one value per kernel (a "gaussian descriptor").
3. The rotations are max-pooled, giving rotational invariance.

All operations are vectorised over the rotation axis (the original looped in Python
over rotations inside the TF graph).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def compute_initial_coordinates(max_rho: float, n_rhos: int, n_thetas: int) -> np.ndarray:
    """Grid kernel centres (rho, theta); reproduces the original meshgrid layout."""
    range_rho = [0.0, max_rho]
    range_theta = [0.0, 2 * math.pi]

    grid_rho = np.linspace(range_rho[0], range_rho[1], num=n_rhos + 1)[1:]
    grid_theta = np.linspace(range_theta[0], range_theta[1], num=n_thetas + 1)[:-1]

    grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
    grid_rho_ = grid_rho_.T  # transposes keep the same ordering as the Matlab code
    grid_theta_ = grid_theta_.T
    coords = np.stack([grid_rho_.ravel(), grid_theta_.ravel()], axis=1)
    return coords


class GaussianGrid(nn.Module):
    """Learnable Gaussian kernel positions/sigmas on the (rho, theta) grid."""

    def __init__(
        self,
        max_rho: float,
        n_rhos: int,
        n_thetas: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_rhos = n_rhos
        self.n_thetas = n_thetas
        self.n_gauss = n_rhos * n_thetas
        self.eps = eps
        coords = compute_initial_coordinates(max_rho, n_rhos, n_thetas)
        sigma_rho_init = max_rho / 8.0
        sigma_theta_init = 1.0

        self.mu_rho = nn.Parameter(torch.tensor(coords[:, 0], dtype=torch.float32).unsqueeze(0))
        self.mu_theta = nn.Parameter(torch.tensor(coords[:, 1], dtype=torch.float32).unsqueeze(0))
        self.sigma_rho = nn.Parameter(
            torch.full((1, self.n_gauss), float(sigma_rho_init), dtype=torch.float32)
        )
        self.sigma_theta = nn.Parameter(
            torch.full((1, self.n_gauss), float(sigma_theta_init), dtype=torch.float32)
        )

    def activations(
        self, rho: torch.Tensor, theta: torch.Tensor, mask: torch.Tensor, rotations: torch.Tensor
    ) -> torch.Tensor:
        """Gaussian activations for all rotations.

        Args:
            rho: (B, V)
            theta: (B, V)
            mask: (B, V, 1)
            rotations: (R,)
        Returns:
            (B, R, V, G) activations, mean-normalised over vertices.
        """
        B, V = rho.shape
        R = rotations.shape[0]
        # theta rotated by each rotation (B, R, V)
        th = theta[:, None, :, None] + rotations[None, :, None, None]
        th = torch.remainder(th, 2 * math.pi)
        r = rho[:, None, :, None]

        act = torch.exp(
            -torch.square(r - self.mu_rho) / (torch.square(self.sigma_rho) + self.eps)
        ) * torch.exp(-torch.square(th - self.mu_theta) / (torch.square(self.sigma_theta) + self.eps))
        act = act * mask[:, None, :, None]  # (B, R, V, G)
        denom = act.sum(dim=2, keepdim=True) + self.eps
        act = act / denom  # mean normalisation over patch vertices
        return act


class GeodesicConv(nn.Module):
    """A single geodesic-convolution layer.

    Two operating modes (mirroring the original network):

    * ``per_channel=True``: each input channel gets its own Gaussian grid and its own
      ``(G, G)`` weight matrix; outputs per channel are concatenated (used by layer 1).
    * ``per_channel=False``: one shared grid/weight matrix over all input channels
      (used by layers 2+).
    """

    def __init__(
        self,
        in_channels: int,
        max_rho: float,
        n_rhos: int,
        n_thetas: int,
        n_rotations: int,
        per_channel: bool,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.per_channel = per_channel
        self.n_rotations = n_rotations
        self.n_rhos = n_rhos
        self.n_thetas = n_thetas
        self.n_gauss = n_rhos * n_thetas
        self.register_buffer("rotations", torch.tensor(
            np.arange(n_rotations) * 2 * math.pi / n_rotations, dtype=torch.float32
        ))

        if per_channel:
            self.grids = nn.ModuleList(
                [GaussianGrid(max_rho, n_rhos, n_thetas) for _ in range(in_channels)]
            )
            self.linears = nn.ModuleList(
                [nn.Linear(self.n_gauss, self.n_gauss, bias=True) for _ in range(in_channels)]
            )
        else:
            self.grid = GaussianGrid(max_rho, n_rhos, n_thetas)
            # maps the flattened per-channel gaussian descriptors (C*G) -> (C*G)
            self.linear = nn.Linear(in_channels * self.n_gauss, in_channels * self.n_gauss, bias=True)

    def _single_channel(
        self, x: torch.Tensor, grid: GaussianGrid, linear: nn.Linear, rho, theta, mask
    ) -> torch.Tensor:
        all_conv = []
        for rot in self.rotations:
            act = grid.activations(rho, theta, mask, rot[None])  # (B, 1, V, G)
            gd = torch.einsum("brvg,bv->brg", act, x).squeeze(1)  # (B, G)
            gd = linear(gd)
            gd = torch.relu(gd)
            all_conv.append(gd)
        return torch.stack(all_conv, dim=1).amax(dim=1)  # rotation-invariant max-pool

    def forward(self, x, rho, theta, mask) -> torch.Tensor:
        """Map ``(B, V, C)`` patch features to descriptors.

        ``per_channel=True`` -> ``(B, C * G)``
        ``per_channel=False`` -> ``(B, C * G)`` with the rotation-invariant max-pool
        applied (reshaping to ``(B, C, G)`` for downstream mean-pooling is left to the
        caller, matching the original network structure).
        """
        if self.per_channel:
            outs = [
                self._single_channel(x[:, :, c], self.grids[c], self.linears[c], rho, theta, mask)
                for c in range(self.in_channels)
            ]
            return torch.cat(outs, dim=1)

        B = x.shape[0]
        all_conv = []
        for rot in self.rotations:
            act = self.grid.activations(rho, theta, mask, rot[None])  # (B, 1, V, G)
            gd = torch.einsum("brvg,bvc->brcg", act, x).squeeze(1)  # (B, C, G)
            gd = gd.reshape(B, self.in_channels * self.n_gauss)
            gd = self.linear(gd)
            gd = torch.relu(gd)
            all_conv.append(gd)
        return torch.stack(all_conv, dim=1).amax(dim=1)  # (B, C*G), rotation-invariant
