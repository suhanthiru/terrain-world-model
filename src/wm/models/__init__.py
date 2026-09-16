"""Model zoo. Every entry exposes the same rollout(x0, actions) -> (frames, hidden_norms)."""

from __future__ import annotations

import torch.nn as nn

from .baselines import IdentityModel, MeanTerrainModel
from .blocks import count_parameters
from .latent import LatentWorldModel
from .unet import UNetWorldModel

__all__ = [
    "IdentityModel", "MeanTerrainModel", "LatentWorldModel", "UNetWorldModel",
    "build_model", "count_parameters",
]

UNTRAINED = {"identity": IdentityModel, "mean_terrain": MeanTerrainModel}


def build_model(arch: str, **kwargs) -> nn.Module:
    """arch is one of: latentA, latentB, unet, identity, mean_terrain."""
    if arch in UNTRAINED:
        return UNTRAINED[arch]()
    if arch.startswith("latent"):
        protocol = arch[-1].upper()
        if protocol not in ("A", "B"):
            raise ValueError(f"latent arch must end in A or B, got {arch!r}")
        return LatentWorldModel(protocol=protocol, **kwargs)
    if arch == "unet":
        kwargs.pop("latent_dim", None)
        return UNetWorldModel(**kwargs)
    raise ValueError(f"unknown architecture {arch!r}")
