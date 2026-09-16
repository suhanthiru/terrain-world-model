"""Shared building blocks.

Upsampling is nearest-neighbour followed by a 3x3 convolution, never a transposed
convolution. Checkerboard artifacts are one of the failure modes this project sets out
to detect and count, so building the models out of the operator best known for
manufacturing them would be self-defeating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm that degrades gracefully when the channel count is not divisible."""
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)
        self.norm = group_norm(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DownBlock(ConvBlock):
    """Halves the spatial resolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(in_ch, out_ch, kernel=4, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)


class UpBlock(nn.Module):
    """Doubles the spatial resolution without transposed convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, kernel=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(F.interpolate(x, scale_factor=2, mode="nearest"))


class CoordChannels(nn.Module):
    """Two fixed channels carrying normalised y and x position.

    The simulator is not translation invariant -- it has walls, and a dig happens at an
    absolute place on the patch -- so the convolutions are given the coordinates rather
    than being asked to infer them from border effects.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        line = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(line, line, indexing="ij")
        self.register_buffer("coords", torch.stack([yy, xx])[None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)


class ActionEncoder(nn.Module):
    """Shared by every architecture, so the action pathway is never a confound."""

    def __init__(self, action_dim: int, width: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class FiLM(nn.Module):
    """Per-channel scale and shift conditioned on the action embedding."""

    def __init__(self, embed_dim: int, channels: int) -> None:
        super().__init__()
        self.to_params = nn.Linear(embed_dim, 2 * channels)
        # Start as the identity so conditioning is learned rather than imposed.
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_params(embed).chunk(2, dim=-1)
        return x * (1.0 + scale[..., None, None]) + shift[..., None, None]


def zero_init(conv: nn.Conv2d) -> nn.Conv2d:
    """Make a model start life as exactly the identity baseline.

    The predicted delta is zero everywhere at initialisation, which is the right answer
    for the roughly 95% of cells that do not move in a given step. It also makes epoch
    zero directly comparable against the identity baseline, and removes a class of
    early-training divergence in the unrolled loss.
    """
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
