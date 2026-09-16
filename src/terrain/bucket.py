"""Swept bucket removal.

The bucket follows a rigid trajectory: the cut floor sits at a fixed depth below the
height at the *entry point*, not below the local surface at each cell. That is what a
real machine does -- boom and arm geometry set the path, the bucket does not feel its
way along the ground -- and it has two consequences worth having.

It makes the cut richer on sloped terrain, which is exactly where it matters. Digging
uphill, the rising ground cuts into the rigid floor and more material comes out; digging
downhill the bucket leaves the ground and removes nothing. Terrain shape therefore
couples into the outcome of an action, which is the signal a world model has to learn.
Cutting a constant depth below the local surface removes that coupling entirely -- the
same volume comes out regardless of what the terrain looks like.

It also makes the cut idempotent. ``min(h, floor)`` applied twice removes nothing the
second time, which is a crisp property to test against. Local-relative cutting is not
idempotent: repeat the same action and it digs forever, straight into the inescapable
pit failure mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from .grid import Grid

# Fraction of the bucket half-width given over to the rounded shoulder. Real buckets
# have a flat cutting edge, and the near-vertical side walls left by a flat bottom are
# the interesting part of the dynamics -- they are what the relaxation then has to
# collapse. A fully tapered bucket would be both unphysical and less informative.
SHOULDER_FRAC = 0.15

# The entry height is sampled this far *behind* the entry point, along the sweep axis.
# Sampling at the entry itself is subtly wrong: bilinear interpolation reads a 2x2
# neighbourhood, part of which the sweep lowers, so re-running the same action would
# reference a surface it had already dug and cut deeper again. Standing off by two cells
# puts the reference outside the swept region, which makes the cut exactly idempotent.
REFERENCE_STANDOFF = 0.10


@dataclass(frozen=True)
class CutInfo:
    removed: float            # volume actually taken, m^3 -- equals min(v_geom, capacity)
    v_geom: float             # volume the swept geometry would have taken, uncapped
    capacity_clipped: bool
    s_fill: float             # along-track distance at which the bucket filled, m
    empty_cut: bool
    n_cells: int              # cells the cut touched


def _cross_track_taper(abs_t: np.ndarray, half_width: float) -> np.ndarray:
    """Flat across most of the bucket, easing to zero over a raised-cosine shoulder."""
    shoulder = SHOULDER_FRAC * half_width
    taper = np.ones_like(abs_t)
    if shoulder <= 0.0:
        return taper
    flat_to = half_width - shoulder
    edge = abs_t > flat_to
    if np.any(edge):
        frac = np.clip((abs_t[edge] - flat_to) / shoulder, 0.0, 1.0)
        taper[edge] = 0.5 * (1.0 + np.cos(math.pi * frac))
    return taper


def swept_cut(
    h: np.ndarray,
    grid: Grid,
    entry_y: float,
    entry_x: float,
    heading: float,
    sweep_len: float,
    depths: tuple[float, float, float],
    width: float,
    capacity: float,
) -> tuple[np.ndarray, CutInfo]:
    """Remove the material swept out by one bucket pass.

    ``depths`` are the cut depths at s/L = 1/3, 2/3 and 1. Depth at the entry is pinned
    to zero, since the tip enters at the surface. The four points are interpolated with
    a shape-preserving PCHIP spline so the profile cannot overshoot to a negative depth
    between control points.

    When the swept volume exceeds bucket capacity the sweep is truncated *along track*:
    the bucket fills as it advances, and once full, advancing further adds nothing.
    Scaling the depth down uniformly instead would be non-causal -- the machine would
    have to know in advance to cut shallower over the whole path.
    """
    out = h.astype(np.float64, copy=True)
    yc, xc = grid.cell_centres()
    rel_y = yc - entry_y
    rel_x = xc - entry_x

    # Along-track and cross-track coordinates, in metres.
    sin_p, cos_p = math.sin(heading), math.cos(heading)
    along = rel_y * sin_p + rel_x * cos_p
    cross = rel_y * cos_p - rel_x * sin_p

    half_width = 0.5 * width
    in_sweep = (along >= 0.0) & (along <= sweep_len) & (np.abs(cross) <= half_width)
    idx = np.flatnonzero(in_sweep.ravel())
    if idx.size == 0:
        return out, CutInfo(0.0, 0.0, False, sweep_len, True, 0)

    along_in = along.ravel()[idx]
    abs_cross_in = np.abs(cross.ravel()[idx])

    knots = np.array([0.0, sweep_len / 3.0, 2.0 * sweep_len / 3.0, sweep_len])
    values = np.array([0.0, depths[0], depths[1], depths[2]], dtype=np.float64)
    profile = PchipInterpolator(knots, values)(np.clip(along_in, 0.0, sweep_len))

    reference = grid.bilinear(
        h, entry_y - REFERENCE_STANDOFF * sin_p, entry_x - REFERENCE_STANDOFF * cos_p
    )
    floor = reference - profile * _cross_track_taper(abs_cross_in, half_width)
    cut = np.clip(h.ravel()[idx] - floor, 0.0, None)

    area = grid.cell_area
    v_geom = float(np.sum(cut, dtype=np.float64) * area)
    if v_geom <= 0.0:
        return out, CutInfo(0.0, 0.0, False, sweep_len, True, 0)

    capacity_clipped = v_geom > capacity
    s_fill = sweep_len
    if capacity_clipped:
        # Fill the bucket in along-track order, then partially fill the single cell the
        # capacity boundary falls inside. That last term is what makes `removed` come
        # out at exactly min(v_geom, capacity) instead of quantised to whole cells --
        # truncating at whole cross-track bands instead would quantise at ~30% of
        # capacity, which is far too coarse to be invisible.
        order = np.argsort(along_in, kind="stable")
        running = np.cumsum(cut[order]) * area
        full = int(np.searchsorted(running, capacity, side="right"))
        kept = np.zeros_like(cut)
        kept[order[:full]] = cut[order[:full]]
        if full < order.size:
            already = running[full - 1] if full > 0 else 0.0
            kept[order[full]] = (capacity - already) / area
            s_fill = float(along_in[order[full]])
        cut = kept

    removed = float(np.sum(cut, dtype=np.float64) * area)
    out.ravel()[idx] -= cut
    return out, CutInfo(
        removed=removed,
        v_geom=v_geom,
        capacity_clipped=capacity_clipped,
        s_fill=s_fill,
        empty_cut=False,
        n_cells=int(np.count_nonzero(cut)),
    )
