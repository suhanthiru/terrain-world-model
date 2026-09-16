"""Sorting divergence into named failure modes.

Thresholds are calibrated on the dev split using the *baseline* models only, then frozen
and applied unchanged to everything else. Tuning them against the model under test would
make the taxonomy flatter whatever it found.

The conjunctions matter. ``blur_to_mean`` requires both missing fine structure *and*
under-activity, so a model that correctly predicts a genuinely smooth region is not
mislabelled. ``spatial_displacement`` requires a large centroid shift *and* a high
shift-corrected correlation, which is precisely what separates a right-shaped hole in the
wrong place from undifferentiated garbage.
"""

from __future__ import annotations

import numpy as np

AUDIT_HORIZONS = (20, 50)

THRESHOLDS = {
    "diverged_height_m": 10.0,
    "volume_explosion": 0.25,
    "volume_collapse": -0.25,
    "checkerboard_score": 3.0,
    "sign_alternation": 0.85,
    "identity_activity": 0.15,
    "blur_hf_ratio": 0.5,
    "blur_activity": 0.6,
    "displacement_cells": 4.0,
    "displacement_ncc": 0.6,
}

# Applied in this order when collapsing the multi-label vector to one headline class.
PRECEDENCE = (
    "diverged",
    "volume_explosion",
    "volume_collapse",
    "checkerboard",
    "identity_collapse",
    "blur_to_mean",
    "spatial_displacement",
)


def classify(scores: dict, pred: np.ndarray, horizon: int) -> dict:
    """Every flag that fires for one episode at one audit horizon."""
    k = min(horizon, pred.shape[0]) - 1
    peak = float(np.abs(pred[: k + 1]).max())
    finite = bool(np.isfinite(pred[: k + 1]).all())

    def at(name):
        """A metric at the audit horizon, or NaN.

        Every comparison against NaN is False, so an episode whose statistic could not be
        computed simply goes unflagged rather than being guessed at.
        """
        value = scores[name][k]
        return float(value) if np.isfinite(value) else np.nan

    eps = at("eps_net")
    activity = at("activity_ratio")
    hf_ratio = at("hf_ratio")

    flags = {
        "diverged": (not finite) or peak > THRESHOLDS["diverged_height_m"],
        "volume_explosion": eps > THRESHOLDS["volume_explosion"],
        "volume_collapse": eps < THRESHOLDS["volume_collapse"],
        "checkerboard": (at("checkerboard") > THRESHOLDS["checkerboard_score"]
                         or at("sign_alternation") > THRESHOLDS["sign_alternation"]),
        "identity_collapse": activity < THRESHOLDS["identity_activity"],
        "blur_to_mean": (hf_ratio < THRESHOLDS["blur_hf_ratio"]
                         and activity < THRESHOLDS["blur_activity"]),
        "spatial_displacement": (at("centroid_shift") > THRESHOLDS["displacement_cells"]
                                 and at("shift_corrected_ncc") > THRESHOLDS["displacement_ncc"]),
    }
    flags = {name: bool(value) for name, value in flags.items()}

    label = next((name for name in PRECEDENCE if flags[name]), "ok")
    return {
        "horizon": horizon,
        "label": label,
        "any_flag": label != "ok",
        **{f"flag_{name}": flags[name] for name in PRECEDENCE},
    }


def summarise(rows: list[dict]) -> dict:
    """Class histogram, per-flag rates, and how often flags co-occur."""
    if not rows:
        return {}
    labels = [r["label"] for r in rows]
    names = list(PRECEDENCE)
    matrix = np.array([[r[f"flag_{n}"] for n in names] for r in rows], dtype=float)

    co_occurrence = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j > i and matrix[:, i].sum() and matrix[:, j].sum():
                both = float((matrix[:, i] * matrix[:, j]).mean())
                if both > 0:
                    co_occurrence[f"{a}+{b}"] = round(both, 4)

    return {
        "n_episodes": len(rows),
        "any_flag_rate": float(np.mean([r["any_flag"] for r in rows])),
        "labels": {name: labels.count(name) for name in ["ok", *names] if labels.count(name)},
        "flag_rates": {name: float(matrix[:, i].mean()) for i, name in enumerate(names)},
        "co_occurrence": co_occurrence,
    }
