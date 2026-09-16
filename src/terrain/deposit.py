"""Placing excavated material back on the terrain.

The dumped material forms a cone at the repose slope, but it is applied as

    h_new = max(h, z - tan(theta) * r)

rather than by adding a cone to the existing surface. That choice matters more than it
looks. Slopes add: dropping a repose-slope cone onto repose-slope ground gives you
2*tan(theta), a surface the relaxation then has to spend hundreds of sweeps demolishing.
Measured on a sloped terrain, the additive version raised the worst slope violation from
4.8e-4 to 4.2e-2, while the max version introduced exactly none -- the maximum of two
K-Lipschitz surfaces is K-Lipschitz, so a deposit onto relaxed ground is *already* at
repose and the relaxation that follows it is a no-op.

It is also the physically right answer: material poured from a point fills the low side
and runs off the high side. It does not float a rigid cone on top of a hill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid


@dataclass(frozen=True)
class DepositInfo:
    placed: float       # volume actually added, m^3
    apex: float         # elevation of the cone tip, m
    n_active: int       # cells that received material


def deposit(
    h: np.ndarray,
    grid: Grid,
    dump_y: float,
    dump_x: float,
    volume: float,
    tan_theta: float,
) -> tuple[np.ndarray, DepositInfo]:
    """Place exactly ``volume`` of material as a repose-slope cone at the dump point.

    The apex elevation is solved in closed form. Writing b_c = h_c + tan(theta)*r_c for
    the level at which cell c starts receiving material, the placed volume is

        V(z) = sum_c max(0, z - b_c) * cell_area

    which is the classic water-filling functional: non-decreasing and piecewise linear
    in z with a kink at each b_c, so the root is unique. Sorting b ascending, the level
    with the k smallest cells active is z_k = (V/A + sum_{i<k} b_i) / k, and the correct
    k is the largest one whose cell is genuinely submerged.

    A cone clipped by the domain wall is not a failure case: the solver simply returns a
    higher apex, piling the same volume up steeper in the corner rather than losing it.
    """
    if volume <= 0.0:
        return h.astype(np.float64, copy=True), DepositInfo(placed=0.0, apex=float("nan"), n_active=0)

    yc, xc = grid.cell_centres()
    radius = np.hypot(yc - dump_y, xc - dump_x)

    base = (h + tan_theta * radius).ravel()
    base_sorted = np.sort(base)
    running = np.cumsum(base_sorted)
    k = np.arange(1, base_sorted.size + 1, dtype=np.float64)
    levels = (volume / grid.cell_area + running) / k

    # count_nonzero rather than argmax over the predicate: flat ground gives many cells
    # an identical radius, so `base` carries heavy exact ties and an interval-style
    # search can land on the wrong side of them. The predicate is prefix-true, so
    # counting is both correct and tie-safe.
    n_active = int(np.count_nonzero(levels > base_sorted))
    apex = float(levels[n_active - 1])

    out = np.maximum(h, apex - tan_theta * radius)
    placed = float(np.sum(out - h, dtype=np.float64) * grid.cell_area)
    return out, DepositInfo(placed=placed, apex=apex, n_active=n_active)


def deposit_apex_by_bisection(
    h: np.ndarray,
    grid: Grid,
    dump_y: float,
    dump_x: float,
    volume: float,
    tan_theta: float,
    iterations: int = 200,
) -> float:
    """Apex elevation found by bisection. A test oracle for the closed form, nothing more."""
    yc, xc = grid.cell_centres()
    radius = np.hypot(yc - dump_y, xc - dump_x)
    base = h + tan_theta * radius
    lo = float(base.min())
    hi = float(base.max()) + volume / (grid.cell_area * h.size) + 1.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        placed = float(np.sum(np.clip(mid - base, 0.0, None), dtype=np.float64) * grid.cell_area)
        if placed < volume:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
