"""Angle-of-repose relaxation.

Material is moved downhill until no adjacent cell pair exceeds the critical slope
tan(theta). The scheme is Jacobi over an 8-neighbour lattice, and it is exactly
mass-conserving by construction: every edge produces a single scalar flow that is
subtracted from one endpoint and added to the other, so there is no separate outflow
and inflow computation that could disagree.

Two implementation details are load-bearing:

* Slicing, never ``np.roll``. ``roll`` wraps, which silently gives periodic boundaries.
  Those conserve mass perfectly -- so every conservation test still passes -- while
  teleporting material across the domain. Slicing gives no-flux walls for free.

* Diagonal edges use the same *slope* limit, so their critical drop is
  ``tan(theta) * dx * sqrt(2)``, and they are weighted at half the axial edges. That
  weighting is not cosmetic: it drops the Laplacian's largest eigenvalue from 12 to 8,
  which raises the stable step from alpha < 1/6 to alpha < 1/4.

With 4 neighbours instead of 8 the equilibrium pile is a 45-degree-rotated square
pyramid rather than a cone, and the effective diagonal angle of repose comes out at
36.5 degrees when theta = 30 was requested -- a 6.5 degree error in the very parameter
the terrain families vary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Jacobi step size. The linearised operator is (I - alpha * L) with L the weighted
# graph Laplacian, whose symbol for w_diag = 1/2 is 6 - 2a - 2b - 2ab (a = cos kx,
# b = cos ky). Its maximum is 8, attained along a = -1, so explicit-Euler stability
# needs alpha < 2/8 = 0.25. We run at 80% of that.
ALPHA = 0.20
W_DIAG = 0.5

# Convergence is declared when the worst slope violation falls below TOL_FRAC of the
# axial critical drop -- about 33 microns at theta = 33 degrees, comfortably under the
# 0.1 mm at which frames are stored.
TOL_FRAC = 1e-3
MAX_SWEEPS = 4000

# The violation check costs roughly a third of a sweep, so it is not worth doing
# every iteration.
CHECK_EVERY = 8

_SQRT2 = math.sqrt(2.0)

# Edge families: (slice of the "from" cells, slice of the "to" cells, is_diagonal).
# Together these cover every adjacent pair in the 8-neighbourhood exactly once.
_EDGES = (
    (np.s_[:, :-1], np.s_[:, 1:], False),      # west  -> east
    (np.s_[:-1, :], np.s_[1:, :], False),      # north -> south
    (np.s_[:-1, :-1], np.s_[1:, 1:], True),    # NW    -> SE
    (np.s_[:-1, 1:], np.s_[1:, :-1], True),    # NE    -> SW
)


@dataclass(frozen=True)
class RelaxInfo:
    """What the relaxation did, for the per-step bookkeeping record."""

    n_sweeps: int
    residual: float
    hit_cap: bool


def critical_drops(tan_theta: float, dx: float) -> tuple[float, float]:
    """Maximum height difference an axial and a diagonal cell pair may hold."""
    return tan_theta * dx, tan_theta * dx * _SQRT2


def max_violation(h: np.ndarray, d_axial: float, d_diag: float) -> float:
    """Largest amount by which any adjacent pair exceeds its critical drop."""
    worst = 0.0
    for sa, sb, diagonal in _EDGES:
        limit = d_diag if diagonal else d_axial
        gap = np.abs(h[sa] - h[sb]) - limit
        local = float(gap.max())
        if local > worst:
            worst = local
    return max(worst, 0.0)


def _sweep(h: np.ndarray, d_axial: float, d_diag: float, alpha: float, w_diag: float) -> None:
    """One Jacobi pass, in place. Exactly mass-conserving."""
    delta = np.zeros_like(h)
    for sa, sb, diagonal in _EDGES:
        limit = d_diag if diagonal else d_axial
        weight = alpha * w_diag if diagonal else alpha
        diff = h[sa] - h[sb]
        # Exactly one of these is non-zero, since the limit is positive.
        flow = weight * (np.clip(diff - limit, 0.0, None) - np.clip(-diff - limit, 0.0, None))
        delta[sa] -= flow
        delta[sb] += flow
    h += delta


def relax(
    h: np.ndarray,
    tan_theta: float,
    dx: float,
    *,
    alpha: float = ALPHA,
    w_diag: float = W_DIAG,
    tol_frac: float = TOL_FRAC,
    max_sweeps: int = MAX_SWEEPS,
    check_every: int = CHECK_EVERY,
) -> tuple[np.ndarray, RelaxInfo]:
    """Relax a surface until no adjacent pair exceeds the angle of repose.

    Returns a new array; the input is not modified.

    Running a fixed small number of passes instead would be a mistake: information
    travels one cell per sweep and the operator is diffusive, so a pile of radius R
    needs O(R^2 / alpha) sweeps -- hundreds, not two or three. Stopping early records a
    transient rather than a repose surface, makes theta nearly unidentifiable from the
    data, and turns the sweep count into a hidden state variable.
    """
    if alpha * _lambda_max(w_diag) >= 2.0:
        raise ValueError(
            f"alpha={alpha} with w_diag={w_diag} exceeds the stability bound "
            f"{2.0 / _lambda_max(w_diag):.4f}; the relaxation would diverge"
        )

    d_axial, d_diag = critical_drops(tan_theta, dx)
    tol = tol_frac * d_axial
    out = h.astype(np.float64, copy=True)

    swept = 0
    while swept < max_sweeps:
        for _ in range(min(check_every, max_sweeps - swept)):
            _sweep(out, d_axial, d_diag, alpha, w_diag)
            swept += 1
        residual = max_violation(out, d_axial, d_diag)
        # Guard explicitly: once the field blows up the residual is NaN, and every
        # ordinary comparison against NaN is False, so a naive check reports success.
        if not math.isfinite(residual):
            raise FloatingPointError(
                f"relaxation diverged after {swept} sweeps (alpha={alpha}, w_diag={w_diag})"
            )
        if residual < tol:
            return out, RelaxInfo(n_sweeps=swept, residual=residual, hit_cap=False)

    residual = max_violation(out, d_axial, d_diag)
    return out, RelaxInfo(n_sweeps=swept, residual=residual, hit_cap=True)


def _lambda_max(w_diag: float) -> float:
    """Largest eigenvalue of the weighted 8-neighbour graph Laplacian.

    symbol(a, b) = 4 - 2a - 2b + 4 * w_diag * (1 - ab)   for a = cos kx, b = cos ky
    """
    corners = [(a, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
    # The symbol is bilinear in (a, b), so its extremes sit at the corners, plus the
    # a = -1 edge where it is constant in b for w_diag = 1/2.
    return max(4.0 - 2.0 * a - 2.0 * b + 4.0 * w_diag * (1.0 - a * b) for a, b in corners)


def max_slope(h: np.ndarray, dx: float) -> np.ndarray:
    """Steepest slope at each cell, over its 8 neighbours.

    Diagonal neighbours are sqrt(2)*dx apart, so dividing by the true separation makes
    this comparable against tan(theta) directly. The evaluation harness uses it to ask
    how much of a predicted surface would physically collapse.
    """
    diag = dx * _SQRT2
    out = np.zeros_like(h)
    for sa, sb, diagonal in _EDGES:
        sep = diag if diagonal else dx
        slope = np.abs(h[sa] - h[sb]) / sep
        np.maximum(out[sa], slope, out=out[sa])
        np.maximum(out[sb], slope, out=out[sb])
    return out
