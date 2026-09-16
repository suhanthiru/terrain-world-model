"""Running a model over whole episodes and scoring it."""

from __future__ import annotations

import numpy as np
import torch

from terrain.grid import DEFAULT_GRID
from wm.data.dataset import EpisodeStore
from wm.eval import metrics


@torch.no_grad()
def rollout_episodes(
    model,
    store: EpisodeStore,
    *,
    device: str = "cuda",
    start_step: int = 0,
    horizon: int | None = None,
    batch_size: int = 8,
    protocol: str | None = None,
) -> np.ndarray:
    """Predicted surfaces for every episode, shape (E, horizon, H, W) in metres.

    Predictions come back on the same datum as the stored frames: the per-episode
    offset is subtracted before the model sees anything and added back afterwards.
    """
    model.eval()
    horizon = horizon or (store.n_steps - start_step)
    out = np.empty((len(store), horizon, *DEFAULT_GRID.shape), dtype=np.float32)

    for begin in range(0, len(store), batch_size):
        episodes = range(begin, min(begin + batch_size, len(store)))
        frames = np.stack([store.episode_frames(e) for e in episodes])
        start = torch.from_numpy(frames[:, start_step]).to(device).float()
        offset = start.mean(dim=(1, 2))
        actions = torch.stack([
            store.actions[e, start_step:start_step + horizon] for e in episodes
        ]).to(device)

        pred, _ = model.rollout(start - offset[:, None, None], actions, protocol=protocol)
        out[begin:begin + len(list(episodes))] = (
            (pred + offset[:, None, None, None]).cpu().numpy()
        )
    return out


def score_episode(
    pred: np.ndarray,
    frames: np.ndarray,
    removed: np.ndarray,
    tan_theta: float,
    swell: float,
    start_step: int = 0,
) -> dict:
    """Every metric for one episode, as arrays indexed by horizon."""
    start = frames[start_step]
    truth = frames[start_step + 1: start_step + 1 + pred.shape[0]]
    removed = removed[start_step: start_step + pred.shape[0]]

    scores = metrics.accuracy(pred, truth, start)
    scores.update(metrics.volume_error(pred, start, removed, swell))
    scores["repose_violation"] = metrics.repose_violation(pred, tan_theta)
    scores["repose_violation_true"] = metrics.repose_violation(truth, tan_theta)
    scores["hf_ratio"] = metrics.high_frequency_ratio(pred, truth)
    scores["checkerboard"] = metrics.checkerboard_score(pred, truth)
    scores["sign_alternation"] = metrics.error_sign_alternation(pred, truth)
    scores["activity_ratio"] = metrics.activity_ratio(pred, truth, start)
    scores["centroid_shift"] = metrics.excavation_centroid_shift(pred, truth, start)
    scores["shift_corrected_ncc"] = metrics.shift_corrected_correlation(pred, truth, start)

    displaced = metrics.displaced_volumes(pred, start)
    truth_displaced = metrics.displaced_volumes(truth, start)
    scores["effective_swell"] = displaced["effective_swell"]
    with np.errstate(divide="ignore", invalid="ignore"):
        scores["displaced_error"] = np.where(
            truth_displaced["down"] > 1e-9,
            (displaced["down"] - truth_displaced["down"]) / truth_displaced["down"],
            np.nan,
        )
    return scores
