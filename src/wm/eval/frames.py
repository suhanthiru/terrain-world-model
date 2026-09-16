"""Saving the frames behind a failure.

A taxonomy that reports only counts is hard to trust. These panels are written at
evaluation time, while the predictions are still in memory, so the figure scripts never
have to load a checkpoint to produce them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wm.eval.failures import PRECEDENCE

WORST_PER_CLASS = 3

# Which statistic makes an episode the worst example of each class, and which way is bad.
RANK_BY = {
    "diverged": ("max_ae", 1),
    "volume_explosion": ("eps_net", 1),
    "volume_collapse": ("eps_net", -1),
    "checkerboard": ("checkerboard", 1),
    "identity_collapse": ("activity_ratio", -1),
    "blur_to_mean": ("hf_ratio", -1),
    "spatial_displacement": ("centroid_shift", 1),
}


def _panel(ax, field, title, cmap="terrain", limits=None):
    low, high = limits if limits else (np.nanmin(field), np.nanmax(field))
    image = ax.imshow(field, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def save_failure_panel(
    path: Path,
    start: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    label: str,
    uid: str,
    horizon: int,
) -> None:
    """Start, truth, prediction, signed error, and where each thinks material moved."""
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.1))
    span = (min(start.min(), truth.min(), pred.min()), max(start.max(), truth.max(), pred.max()))

    _panel(axes[0], start, "start surface", limits=span)
    _panel(axes[1], truth, f"simulator at k={horizon}", limits=span)
    _panel(axes[2], pred, f"model at k={horizon}", limits=span)

    error = pred - truth
    scale = max(float(np.abs(error).max()), 1e-9)
    image = _panel(axes[3], error, f"error (max {scale * 1000:.0f} mm)",
                   cmap="RdBu_r", limits=(-scale, scale))
    fig.colorbar(image, ax=axes[3], fraction=0.046, shrink=0.85)

    # Where each of them moved material, so a displaced dig is visible at a glance.
    moved = np.stack([truth - start, pred - start])
    extent = max(float(np.abs(moved).max()), 1e-9)
    image = _panel(axes[4], (pred - start) - (truth - start),
                   "model change minus true change", cmap="PuOr_r",
                   limits=(-extent, extent))
    fig.colorbar(image, ax=axes[4], fraction=0.046, shrink=0.85)

    fig.suptitle(f"{label}  --  {uid}", fontsize=9, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def dump_worst(
    out_dir: Path,
    failures: list[dict],
    per_episode: "pd.DataFrame",  # noqa: F821
    predictions: np.ndarray,
    store,
    horizon: int,
) -> list[str]:
    """One panel for each of the worst few episodes in every failure class."""
    written = []
    step = min(horizon, predictions.shape[1]) - 1
    at_horizon = [f for f in failures if f["horizon"] == horizon]

    for label in PRECEDENCE:
        flagged = [f for f in at_horizon if f[f"flag_{label}"]]
        if not flagged:
            continue
        metric, direction = RANK_BY[label]
        rows = per_episode[per_episode["horizon"] == step + 1].set_index("episode")

        def rank(entry):
            value = rows[metric].get(entry["episode"], np.nan)
            return -np.inf if not np.isfinite(value) else direction * value

        for entry in sorted(flagged, key=rank, reverse=True)[:WORST_PER_CLASS]:
            episode = entry["episode"]
            frames = store.episode_frames(episode)
            name = f"k{horizon}_{label}_{entry['episode_uid']}.png"
            save_failure_panel(
                out_dir / name, frames[0].astype(np.float64),
                frames[step + 1].astype(np.float64),
                predictions[episode, step].astype(np.float64),
                label, entry["episode_uid"], horizon,
            )
            written.append(name)
    return written
