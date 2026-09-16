"""The latent world model.

A convolutional encoder compresses the heightmap to a small vector, a GRU carries that
vector forward under an action embedding, and a decoder reads out a *delta* heightmap
which is added to the current surface.

Two rollout protocols are supported, and the distinction is the point of the experiment:

``B`` -- encode once, then roll the latent forward and never look at a heightmap again.
This is the actual world-model claim: that ``z`` is a sufficient statistic for the
dynamics. It is also the only protocol under which you could plan in latent space
without decoding, which is the practical reason to want a latent model at all.

``A`` -- re-encode the model's own prediction at every step. This is a one-step
predictor unrolled on its own output. It measures error compounding in observation
space, and the model gets to look at what it produced and correct course.

Both are trained, because the pixel-space baseline has no latent to roll and is
structurally forced into ``A``. Comparing a latent model under ``B`` against a U-Net
under ``A`` would confound architecture with protocol and license no conclusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wm.data.normalize import DELTA_SCALE, HEIGHT_SCALE

from .blocks import ActionEncoder, ConvBlock, CoordChannels, DownBlock, UpBlock, group_norm, zero_init

ENCODER_WIDTHS = (24, 48, 64, 96, 128)
DECODER_WIDTHS = (128, 96, 64, 48, 24)
BOTTLENECK_CH = 32
BOTTLENECK_HW = 4
ACTION_EMBED = 64
GRU_HIDDEN = 256


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, size: int = 128) -> None:
        super().__init__()
        self.coords = CoordChannels(size)
        widths = ENCODER_WIDTHS
        layers, in_ch = [], 3
        for out_ch in widths:
            layers.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch
        self.down = nn.Sequential(*layers)
        self.squeeze = nn.Conv2d(in_ch, BOTTLENECK_CH, 1)
        self.to_latent = nn.Linear(BOTTLENECK_CH * BOTTLENECK_HW**2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(self.coords(x))
        return self.to_latent(self.squeeze(h).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, BOTTLENECK_CH * BOTTLENECK_HW**2)
        self.expand = nn.Conv2d(BOTTLENECK_CH, DECODER_WIDTHS[0], 1)
        self.norm = group_norm(DECODER_WIDTHS[0])
        self.act = nn.SiLU()
        ups, in_ch = [], DECODER_WIDTHS[0]
        for out_ch in DECODER_WIDTHS[1:]:
            ups.append(UpBlock(in_ch, out_ch))
            in_ch = out_ch
        ups.append(UpBlock(in_ch, in_ch))
        self.up = nn.Sequential(*ups)
        self.out = zero_init(nn.Conv2d(in_ch, 1, 3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(-1, BOTTLENECK_CH, BOTTLENECK_HW, BOTTLENECK_HW)
        h = self.act(self.norm(self.expand(h)))
        return self.out(self.up(h)).squeeze(1)


class LatentWorldModel(nn.Module):
    """Encoder, action embedding, GRU transition, delta decoder."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 11,
        *,
        protocol: str = "B",
        use_action: bool = True,
        size: int = 128,
    ) -> None:
        super().__init__()
        if protocol not in ("A", "B"):
            raise ValueError(f"protocol must be 'A' or 'B', got {protocol!r}")
        self.protocol = protocol
        self.use_action = use_action
        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim, size=size)
        self.decoder = Decoder(latent_dim)
        # Kept even when use_action is False: the ablation feeds a zero action vector so
        # the architecture, parameter count and optimiser state stay identical and the
        # only thing that changes is the information available.
        self.action_encoder = ActionEncoder(action_dim, out_dim=ACTION_EMBED)
        self.cell = nn.GRUCell(latent_dim + ACTION_EMBED, GRU_HIDDEN)
        self.readout = nn.Linear(GRU_HIDDEN, latent_dim)
        self.init_hidden = nn.Linear(latent_dim, GRU_HIDDEN)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x is offset-removed metres."""
        return self.encoder((x / HEIGHT_SCALE).unsqueeze(1))

    def rollout(
        self,
        x0: torch.Tensor,
        actions: torch.Tensor,
        *,
        protocol: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict K future surfaces from a start surface and K actions.

        Returns the predicted surfaces (B, K, H, W) in offset-removed metres, and the
        GRU hidden-state norms (B, K). Those norms are a cheap diagnostic: if they grow
        without bound past the trained horizon, divergence has a mechanism rather than
        just a description.
        """
        protocol = protocol or self.protocol
        z = self.encode(x0)
        hidden = torch.tanh(self.init_hidden(z))

        current = x0
        frames, norms = [], []
        for step in range(actions.shape[1]):
            a = actions[:, step]
            if not self.use_action:
                a = torch.zeros_like(a)
            embed = self.action_encoder(a)

            if protocol == "A" and step > 0:
                z = self.encode(current)
            hidden = self.cell(torch.cat([z, embed], dim=-1), hidden)
            z = z + self.readout(hidden)

            # Accumulate in float32 regardless of autocast: the signal is centimetre
            # scale on top of metre-scale heights, which is exactly where reduced
            # precision would quietly destroy it.
            delta = self.decoder(z).float() * DELTA_SCALE
            current = current.float() + delta
            frames.append(current)
            norms.append(hidden.detach().norm(dim=-1))

        return torch.stack(frames, dim=1), torch.stack(norms, dim=1)

    def forward(self, x0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.rollout(x0, actions)[0]
