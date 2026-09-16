"""Pixel-space U-Net baseline.

Same job as the latent model -- predict a delta heightmap from a surface and an action --
but with skip connections instead of a bottleneck, so high-frequency detail routes around
the compressed representation rather than through it.

It has no latent to roll forward, so its state *is* the image and it is structurally
confined to protocol A. That is why a protocol-A latent model is trained alongside it:
otherwise the pixel-versus-latent comparison would be measuring the protocol as much as
the architecture.

The comparison is matched on parameters, and that is worth being explicit about, because
it is not matched on compute. The U-Net convolves at full 128x128 resolution throughout
while the latent model does its recurrence on a 64-vector, so at equal parameter counts
the U-Net gets several times the FLOPs per step. It also gets its action broadcast into
every block by FiLM, against a single embedding concatenated into one GRU. Both
asymmetries favour the U-Net. They are reported rather than engineered away -- the
interesting question is not which architecture wins a horse race but what the bottleneck
costs and what it buys.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wm.data.normalize import DELTA_SCALE, HEIGHT_SCALE

from .blocks import ActionEncoder, ConvBlock, CoordChannels, FiLM, UpBlock, near_zero_init

ACTION_EMBED = 64

# Chosen to land within a percent of the latent model's parameter count.
DEFAULT_WIDTHS = (26, 40, 55, 82, 110)

# Recompute block activations in the backward pass instead of storing them. This is the
# same arithmetic for perhaps thirty percent more compute, and it takes a five-step
# unrolled U-Net from about 8.7 GB down to a couple of GB.
#
# It is on by default because this is not a dedicated machine. Sharing the card with an
# Ollama server and a dozen browser GPU processes left roughly 11 of 12 GB spoken for,
# and the resulting allocator thrash slowed an epoch from 195 s to over 50 minutes. A
# smaller batch would also have fixed it, but batch size is held at 32 across every run
# so it cannot confound the comparison; recomputation costs only time.
GRADIENT_CHECKPOINTING = True


class FiLMPair(nn.Module):
    """Two convolutions with action conditioning applied after each."""

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.first = ConvBlock(in_ch, out_ch)
        self.film_first = FiLM(embed_dim, out_ch)
        self.second = ConvBlock(out_ch, out_ch)
        self.film_second = FiLM(embed_dim, out_ch)

    def _conv_pair(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        # Deliberately not named _apply: nn.Module already owns that name and uses it
        # for .cuda() and .to().
        x = self.film_first(self.first(x), embed)
        return self.film_second(self.second(x), embed)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        if GRADIENT_CHECKPOINTING and self.training and torch.is_grad_enabled():
            return checkpoint(self._conv_pair, x, embed, use_reentrant=False)
        return self._conv_pair(x, embed)


class UNetWorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int = 11,
        widths: tuple[int, ...] = DEFAULT_WIDTHS,
        *,
        use_action: bool = True,
        size: int = 128,
    ) -> None:
        super().__init__()
        self.use_action = use_action
        self.protocol = "A"
        self.coords = CoordChannels(size)
        self.action_encoder = ActionEncoder(action_dim, out_dim=ACTION_EMBED)

        w0, w1, w2, w3, w4 = widths
        self.enc1 = FiLMPair(3, w0, ACTION_EMBED)
        self.enc2 = FiLMPair(w0, w1, ACTION_EMBED)
        self.enc3 = FiLMPair(w1, w2, ACTION_EMBED)
        self.enc4 = FiLMPair(w2, w3, ACTION_EMBED)
        self.bottleneck = FiLMPair(w3, w4, ACTION_EMBED)

        self.up4 = UpBlock(w4, w3)
        self.dec4 = FiLMPair(2 * w3, w3, ACTION_EMBED)
        self.up3 = UpBlock(w3, w2)
        self.dec3 = FiLMPair(2 * w2, w2, ACTION_EMBED)
        self.up2 = UpBlock(w2, w1)
        self.dec2 = FiLMPair(2 * w1, w1, ACTION_EMBED)
        self.up1 = UpBlock(w1, w0)
        self.dec1 = FiLMPair(2 * w0, w0, ACTION_EMBED)
        self.out = near_zero_init(nn.Conv2d(w0, 1, 1))

    def predict_delta(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.use_action:
            action = torch.zeros_like(action)
        embed = self.action_encoder(action)

        h = self.coords((x / HEIGHT_SCALE).unsqueeze(1))
        s1 = self.enc1(h, embed)
        s2 = self.enc2(F.avg_pool2d(s1, 2), embed)
        s3 = self.enc3(F.avg_pool2d(s2, 2), embed)
        s4 = self.enc4(F.avg_pool2d(s3, 2), embed)
        b = self.bottleneck(F.avg_pool2d(s4, 2), embed)

        d = self.dec4(torch.cat([self.up4(b), s4], dim=1), embed)
        d = self.dec3(torch.cat([self.up3(d), s3], dim=1), embed)
        d = self.dec2(torch.cat([self.up2(d), s2], dim=1), embed)
        d = self.dec1(torch.cat([self.up1(d), s1], dim=1), embed)
        return self.out(d).squeeze(1)

    def rollout(
        self,
        x0: torch.Tensor,
        actions: torch.Tensor,
        *,
        protocol: str | None = None,
        teacher_frames: torch.Tensor | None = None,
        teacher_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = x0
        frames = []
        for step in range(actions.shape[1]):
            delta = self.predict_delta(current, actions[:, step]).float() * DELTA_SCALE
            predicted = current.float() + delta
            frames.append(predicted)

            current = predicted
            if teacher_frames is not None and teacher_prob > 0.0:
                use_truth = torch.rand(predicted.shape[0], device=predicted.device) < teacher_prob
                current = torch.where(
                    use_truth[:, None, None], teacher_frames[:, step], predicted
                )
        # No recurrent state to report; returned for a uniform interface with the latent model.
        empty = torch.zeros(x0.shape[0], actions.shape[1], device=x0.device)
        return torch.stack(frames, dim=1), empty

    def forward(self, x0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.rollout(x0, actions)[0]
