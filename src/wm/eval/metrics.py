"""Evaluation metrics.

Everything here runs in float64 on de-normalised metres. The volume metric in particular
takes its ground truth from the simulator's exact float64 sidecar and never from the
stored heightmaps, so storage quantisation cannot leak into the headline number. Measured
on generated episodes, quantisation contributes about 0.009% of the material excavated,
against an effect we expect to report at the percent level.

A warning that shapes how these are reported: most physics-plausibility metrics here can
be aced by predicting nothing at all. The identity baseline emits a surface the simulator
itself produced, so it has exactly zero repose violations, and on volume it lands on
exactly -(swell - 1). Neither number means anything without an accuracy metric beside it.

The high-frequency ratio is the exception and is worth keeping for that reason: digging
adds fine structure to the terrain, so a stale surface is progressively too smooth and
identity decays from 0.12 at one step to 0.001 at fifty rather than sitting at one.
"""

from __future__ import annotations

import numpy as np

from terrain.grid import DEFAULT_GRID, Grid
from terrain.relax import max_slope

ACTIVE_THRESHOLD = 1e-3      # metres; "this cell actually moved"
REPOSE_TOLERANCE = 0.05      # fractional slack before calling a slope a violation
HIGH_FREQ_CUTOFF = 0.25      # cycles per cell: wavelengths shorter than 20 cm


def accuracy(pred: np.ndarray, truth: np.ndarray, start: np.ndarray) -> dict:
    """Per-horizon error curves. Shapes (K, H, W), (K, H, W), (H, W)."""
    err = np.abs(pred - truth)
    moved = np.abs(truth - start) > ACTIVE_THRESHOLD
    masked = np.array([
        err[k][moved[k]].mean() if moved[k].any() else np.nan for k in range(err.shape[0])
    ])
    return {
        "mae": err.mean(axis=(1, 2)),
        "rmse": np.sqrt((err ** 2).mean(axis=(1, 2))),
        "max_ae": err.max(axis=(1, 2)),
        # The number that actually separates models: all-cell error is dominated by the
        # ~95% of cells that never move, where predicting nothing is already correct.
        "masked_mae": masked,
    }


def volume_error(
    pred: np.ndarray,
    start: np.ndarray,
    removed: np.ndarray,
    swell: float,
    grid: Grid = DEFAULT_GRID,
) -> dict:
    """The headline metric: how much material the model invents or loses.

    ``removed`` is the simulator's exact per-step bucket volume, so the cumulative
    excavation C(k) and the true volume change are both float64 ground truth.

        eps_net(k) = [ dV_pred(k) - dV_true(k) ] / C(k)

    The denominator is the crux. Dividing by the total volume of terrain in the box --
    about 40 cubic metres against a few cubic metres moved over an episode -- would make
    every error read as a hundredth of a percent and destroy the story. Dividing by the
    material actually excavated makes it read directly as: for every cubic metre the
    machine moved, the model created this much out of nothing.

    Note that predicting no change at all scores exactly -(swell - 1) at every horizon,
    which is -25% here. That is an exact, free reference line on the figure, and any
    model worse than it is losing to doing nothing.
    """
    cumulative = np.cumsum(removed)
    true_change = (swell - 1.0) * cumulative
    pred_change = (pred - start).sum(axis=(1, 2)) * grid.cell_area

    safe = np.where(cumulative > 1e-9, cumulative, np.nan)
    return {
        "eps_net": (pred_change - true_change) / safe,
        "dv_pred": pred_change,
        "dv_true": true_change,
        "cumulative_removed": cumulative,
    }


def displaced_volumes(frames: np.ndarray, start: np.ndarray) -> dict:
    """Material moving down and up between consecutive predicted frames.

    Deliberately *not* compared against the simulator's ``removed``: that is the bucket
    volume before any slumping, whereas this conflates the cut with the material that
    subsequently ran downhill. The same estimator is applied to the true frames so the
    comparison is like for like.
    """
    series = np.concatenate([start[None], frames], axis=0)
    step = np.diff(series, axis=0)
    down = np.clip(-step, 0.0, None).sum(axis=(1, 2)) * DEFAULT_GRID.cell_area
    up = np.clip(step, 0.0, None).sum(axis=(1, 2)) * DEFAULT_GRID.cell_area
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(down > 1e-9, up / down, np.nan)
    return {"down": down, "up": up, "effective_swell": ratio}


def repose_violation(pred: np.ndarray, tan_theta: float, grid: Grid = DEFAULT_GRID) -> np.ndarray:
    """Fraction of cells whose predicted slope exceeds the angle of repose.

    Tests a hard physical constraint the model was never told about, and reads as: by
    this horizon the model is producing terrain that would physically collapse over this
    fraction of its area.
    """
    limit = tan_theta * (1.0 + REPOSE_TOLERANCE)
    return np.array([float((max_slope(f, grid.dx) > limit).mean()) for f in pred])


_WINDOW_CACHE: dict[int, np.ndarray] = {}
_BAND_CACHE: dict[int, np.ndarray] = {}


def _high_freq_band(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A Hann window and the high-frequency mask for an n x n field.

    The window is not optional. The domain does not wrap -- there are walls -- so an
    unwindowed FFT sees a step discontinuity where the array edges meet and smears its
    energy across every frequency including this band. That leakage swamps the actual
    terrain texture: measured on a smooth surface, a sigma=2 Gaussian blur left the
    apparent high-frequency energy unchanged at 4.3e3, because essentially all of it was
    boundary artifact rather than signal. Tapering the edges to zero removes the
    discontinuity and the metric starts measuring what it claims to.
    """
    if n not in _WINDOW_CACHE:
        hann = np.hanning(n)
        _WINDOW_CACHE[n] = np.outer(hann, hann)
        freq = np.fft.fftfreq(n)
        _BAND_CACHE[n] = np.hypot(*np.meshgrid(freq, freq, indexing="ij")) > HIGH_FREQ_CUTOFF
    return _WINDOW_CACHE[n], _BAND_CACHE[n]


def _radial_high_freq(field: np.ndarray) -> float:
    window, band = _high_freq_band(field.shape[0])
    tapered = (field - field.mean()) * window
    spectrum = np.abs(np.fft.fft2(tapered)) ** 2
    return float(spectrum[band].sum())


def high_frequency_ratio(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Predicted fine-scale energy over true fine-scale energy.

    One statistic that catches two opposite failures: well below one means the model is
    blurring towards the mean, well above one means it is manufacturing checkerboard or
    noise. Around one means the texture is right.
    """
    out = np.empty(pred.shape[0])
    for k in range(pred.shape[0]):
        truth_energy = _radial_high_freq(truth[k])
        out[k] = _radial_high_freq(pred[k]) / truth_energy if truth_energy > 1e-20 else np.nan
    return out


def checkerboard_score(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Energy in the highest representable spatial frequency, relative to truth."""
    kernel = np.array([[1.0, -1.0], [-1.0, 1.0]])

    def energy(field):
        conv = (field[:-1, :-1] * kernel[0, 0] + field[:-1, 1:] * kernel[0, 1]
                + field[1:, :-1] * kernel[1, 0] + field[1:, 1:] * kernel[1, 1])
        return float(np.sqrt((conv ** 2).mean()))

    return np.array([
        energy(pred[k]) / e if (e := energy(truth[k])) > 1e-12 else np.nan
        for k in range(pred.shape[0])
    ])


def error_sign_alternation(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fraction of adjacent cell pairs where the error flips sign.

    White noise sits near 0.5; a checkerboard pattern pushes it towards 1.
    """
    out = np.empty(pred.shape[0])
    for k in range(pred.shape[0]):
        err = pred[k] - truth[k]
        flips = np.concatenate([
            (err[:, :-1] * err[:, 1:] < 0).ravel(),
            (err[:-1, :] * err[1:, :] < 0).ravel(),
        ])
        out[k] = float(flips.mean())
    return out


def activity_ratio(pred: np.ndarray, truth: np.ndarray, start: np.ndarray) -> np.ndarray:
    """How much the model moved, against how much actually moved. Zero means it did nothing."""
    moved_pred = np.abs(pred - start).sum(axis=(1, 2))
    moved_true = np.abs(truth - start).sum(axis=(1, 2))
    return np.where(moved_true > 1e-9, moved_pred / np.maximum(moved_true, 1e-12), np.nan)


def excavation_centroid_shift(
    pred: np.ndarray, truth: np.ndarray, start: np.ndarray, grid: Grid = DEFAULT_GRID
) -> np.ndarray:
    """Distance in cells between where the model dug and where the simulator dug."""
    def centroid(field):
        mass = np.clip(-(field - start), 0.0, None)
        total = mass.sum()
        if total <= 1e-9:
            return None
        idx = np.indices(field.shape)
        return np.array([(idx[0] * mass).sum() / total, (idx[1] * mass).sum() / total])

    out = np.empty(pred.shape[0])
    for k in range(pred.shape[0]):
        a, b = centroid(pred[k]), centroid(truth[k])
        out[k] = np.nan if a is None or b is None else float(np.linalg.norm(a - b))
    return out


def shift_corrected_correlation(pred: np.ndarray, truth: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Best correlation between predicted and true change over all rigid shifts.

    High correlation alongside a large centroid shift is the signature of a
    right-shaped hole in the wrong place, which is a different failure from noise.
    """
    out = np.empty(pred.shape[0])
    for k in range(pred.shape[0]):
        a = pred[k] - start
        b = truth[k] - start
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if denom <= 1e-18:
            out[k] = np.nan
            continue
        cross = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
        out[k] = float(cross.max() / denom)
    return out
