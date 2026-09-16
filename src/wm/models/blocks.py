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


# How far below a standard initialisation the near-zero layers start.
OUTPUT_INIT_SCALE = 0.002


def near_zero_init(conv: nn.Conv2d, scale: float = OUTPUT_INIT_SCALE) -> nn.Conv2d:
    """Start the model predicting almost no change at all.

    The identity is the right answer for roughly 95% of cells in any given step, so
    beginning there means training starts from a good place instead of unlearning noise,
    and it removes a class of early divergence in the unrolled loss.

    It is deliberately *near* zero rather than zero. An exactly zero output layer looks
    appealing -- the model is then bit-identical to the identity baseline -- but it does
    not train: the gradient reaching everything upstream is proportional to this layer's
    own weights, so with them at zero the encoder, GRU and action pathway all receive
    exactly zero gradient. Measured, that left the loss flat at 0.2327 over two thousand
    steps with the activity ratio pinned at 1e-8. Adam eventually inches the layer off
    zero, but the whole network is effectively frozen while it does.

    At this scale the five-step prediction starts within about a third of a millimetre
    of the input surface -- against a typical per-step change of nearly three
    centimetres -- so the intent survives while gradients flow from the first step.
    """
    nn.init.kaiming_uniform_(conv.weight, a=5**0.5)
    with torch.no_grad():
        conv.weight.mul_(scale)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def near_zero_linear(layer: nn.Linear, scale: float = OUTPUT_INIT_SCALE) -> nn.Linear:
    """A linear layer that starts near zero without cutting its input off from gradient."""
    nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
    with torch.no_grad():
        layer.weight.mul_(scale)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class FiLM(nn.Module):
    """Per-channel scale and shift conditioned on the action embedding."""

    def __init__(self, embed_dim: int, channels: int) -> None:
        super().__init__()
        # Near-identity at initialisation so conditioning is learned rather than
        # imposed -- but not *exactly* identity. Zeroing these weights makes the FiLM a
        # pass-through whose gradient with respect to the embedding is zero, which
        # leaves the entire action encoder with no gradient at all and silently kills
        # the action pathway this baseline depends on.
        self.to_params = near_zero_linear(nn.Linear(embed_dim, 2 * channels))

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_params(embed).chunk(2, dim=-1)
        return x * (1.0 + scale[..., None, None]) + shift[..., None, None]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
