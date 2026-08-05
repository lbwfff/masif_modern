"""MaSIF-site network (interface-site prediction).

PyTorch port of ``MaSIF_site.py``. The network stacks ``n_conv_layers`` geodesic
convolutions; layer 1 is applied per input feature channel (each channel gets its own
Gaussian grid), subsequent layers operate on the descriptor of each patch member
rebuilt via the patch index tensor (``indices``), exactly as in the original.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from masif.config import ModelConfig
from masif.models.geodesic_conv import GeodesicConv


class MaSIFSite(nn.Module):
    def __init__(
        self,
        max_rho: float,
        n_thetas: int = 16,
        n_rhos: int = 5,
        n_rotations: int = 16,
        n_conv_layers: int = 3,
        feat_mask=None,
        n_labels: int = 2,
    ):
        super().__init__()
        feat_mask = feat_mask or [1.0] * 5
        self.n_feat = int(sum(feat_mask))
        self.n_gauss = n_thetas * n_rhos
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_rotations = n_rotations
        self.n_conv_layers = n_conv_layers

        # --- layer 1: per-channel geodesic convolutions ---------------------- #
        self.conv1 = GeodesicConv(
            in_channels=self.n_feat,
            max_rho=max_rho,
            n_rhos=n_rhos,
            n_thetas=n_thetas,
            n_rotations=n_rotations,
            per_channel=True,
        )
        self.fc1 = nn.Linear(self.n_feat * self.n_gauss, self.n_gauss)
        self.fc2 = nn.Linear(self.n_gauss, self.n_feat)

        # --- layers 2+: shared-grid convolutions ----------------------------- #
        self.convs = nn.ModuleList()
        if n_conv_layers > 1:
            for _ in range(n_conv_layers - 1):
                self.convs.append(
                    GeodesicConv(
                        in_channels=self.n_feat,
                        max_rho=max_rho,
                        n_rhos=n_rhos,
                        n_thetas=n_thetas,
                        n_rotations=n_rotations,
                        per_channel=False,
                    )
                )

        # --- classification head --------------------------------------------- #
        self.fc_head = nn.Linear(self.n_feat, self.n_thetas)
        self.logits = nn.Linear(self.n_thetas, n_labels)

        # feature mask as a buffer for convenience (keeps forward x channels fixed)
        self.register_buffer(
            "_feat_mask", torch.tensor([m for m in feat_mask], dtype=torch.float32)
        )

    def forward(
        self,
        input_feat: torch.Tensor,
        rho: torch.Tensor,
        theta: torch.Tensor,
        mask: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the network on a batch of patches.

        Args:
            input_feat: (B, V, F) per-patch feature tensor
            rho, theta: (B, V) polar coordinates
            mask: (B, V, 1)
            indices: (B, V) row indices used to rebuild patches for layers 2+
        Returns:
            logits (B, n_labels)
        """
        B = input_feat.shape[0]
        # layer 1
        desc = self.conv1(input_feat, rho, theta, mask)  # (B, F*G)
        desc = torch.relu(self.fc1(desc))  # (B, G)
        desc = torch.relu(self.fc2(desc))  # (B, F)

        # subsequent layers operate on the per-vertex descriptor rebuilt as patches
        for conv in self.convs:
            # rebuild each center's patch from its neighbours' descriptors (B, V, F)
            rows = torch.arange(B, device=desc.device)[:, None]
            patch = desc[rows, indices]
            out = conv(patch, rho, theta, mask)  # (B, F*G)
            out = out.reshape(B, self.n_feat, self.n_gauss).mean(dim=2)  # (B, F)
            desc = out

        desc = torch.relu(self.fc_head(desc))  # (B, n_thetas)
        return self.logits(desc)

    def score(self, input_feat, rho, theta, mask, indices) -> torch.Tensor:
        """Sigmoid probability that each patch center is on the interface."""
        logits = self.forward(input_feat, rho, theta, mask, indices)
        return torch.sigmoid(logits[:, 0])
