"""Untrained reference predictors.

Both matter more than they look. The identity baseline is not a strawman: its error
curve is bounded and saturating while a learned autoregressive model can diverge without
limit, so the horizon at which a model's error *crosses* identity is a clean one-number
answer to "how far can this world model be trusted".

More importantly, identity scores perfectly on every physics-plausibility metric. It
emits a surface the simulator itself produced, so it has no repose violations and no
spectral drift, and on the volume metric it lands on exactly -(swell - 1). Any physics
metric that identity aces is a metric that can be gamed by predicting nothing, which is
why none of them is ever reported without an accuracy metric on the same axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityModel(nn.Module):
    """Predict no change at all."""

    protocol = "A"
    use_action = False

    def rollout(self, x0, actions, *, protocol=None, teacher_frames=None, teacher_prob=0.0):
        k = actions.shape[1]
        frames = x0.unsqueeze(1).expand(-1, k, -1, -1).clone()
        return frames, torch.zeros(x0.shape[0], k, device=x0.device)

    def forward(self, x0, actions):
        return self.rollout(x0, actions)[0]


class MeanTerrainModel(nn.Module):
    """Predict the spatial mean of the start surface everywhere.

    Anchors the bottom of the blur scale: its high-frequency energy is zero by
    construction, so it calibrates what total spectral collapse looks like.
    """

    protocol = "A"
    use_action = False

    def rollout(self, x0, actions, *, protocol=None, teacher_frames=None, teacher_prob=0.0):
        k = actions.shape[1]
        flat = x0.mean(dim=(1, 2))[:, None, None, None].expand(-1, k, x0.shape[1], x0.shape[2])
        return flat.clone(), torch.zeros(x0.shape[0], k, device=x0.device)

    def forward(self, x0, actions):
        return self.rollout(x0, actions)[0]
