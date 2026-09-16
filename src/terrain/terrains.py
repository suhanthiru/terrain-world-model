"""Initial surfaces.

Every family is constructed *already at the angle of repose*, and any global ramp is
kept well below critical. That is a hard requirement rather than a tidiness preference:
relaxation transports material one cell per sweep and is diffusive, so the cost of
settling a disturbance of radius R is O(R^2). A trench cut into a globally near-critical
ramp needs transport across the whole domain and blows through any sensible sweep cap.
Starting from a surface that is already settled keeps every relaxation local.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .grid import Grid

FAMILIES = ("flat", "trench", "slope", "pile")

# Families the models train on, and the ones held out to test terrain generalisation.
TRAIN_FAMILIES = ("flat", "trench")
OOD_FAMILIES = ("slope", "pile")

# A global ramp steeper than this fraction of critical makes relaxation pathological.
MAX_RAMP_FRAC = 0.6

# Constructed flanks sit just *below* repose rather than exactly on it. Building them at
# exactly tan(theta) and then adding roughness tips long stretches marginally over
# critical, and fixing that needs transport along the whole flank -- measured at 1400
# sweeps for a trench and 3000 for a pile, against 8 for a surface that starts settled.
FLANK_FRAC = 0.85

# Roughness is applied as smoothed noise. The amplitude is set from the smoothing
# length so the induced slope stays a modest fraction of critical, leaving headroom
# above FLANK_FRAC.
ROUGHNESS_SIGMA_CELLS = (3.0, 8.0)
ROUGHNESS_SLOPE_FRAC = (0.03, 0.12)


def _roughness(rng: np.random.Generator, grid: Grid, tan_theta: float) -> np.ndarray:
    """Smooth, sub-critical surface noise."""
    sigma = rng.uniform(*ROUGHNESS_SIGMA_CELLS)
    field = gaussian_filter(rng.standard_normal(grid.shape), sigma=sigma, mode="nearest")
    gradient = np.hypot(*np.gradient(field, grid.dx))
    peak = float(gradient.max())
    if peak <= 0.0:
        return np.zeros(grid.shape, dtype=np.float64)
    target = rng.uniform(*ROUGHNESS_SLOPE_FRAC) * tan_theta
    return field * (target / peak)


def make_terrain(
    family: str,
    rng: np.random.Generator,
    tan_theta: float,
    grid: Grid,
) -> tuple[np.ndarray, dict]:
    """Build an initial surface and return it with the parameters that produced it."""
    yc, xc = grid.cell_centres()
    params: dict[str, float] = {}

    if family == "flat":
        height = np.zeros(grid.shape, dtype=np.float64)

    elif family == "trench":
        depth = rng.uniform(0.25, 0.70)
        half_bottom = rng.uniform(0.15, 0.60)
        centre = rng.uniform(0.35, 0.65) * grid.extent
        angle = rng.uniform(-np.pi, np.pi)
        # Distance from the trench axis, so the channel can run in any direction.
        across = np.abs((xc - centre) * np.cos(angle) + (yc - centre) * np.sin(angle))
        flank = np.clip(across - half_bottom, 0.0, None) * (FLANK_FRAC * tan_theta)
        height = -np.clip(depth - flank, 0.0, None)
        params.update(depth=depth, half_bottom=half_bottom, centre=centre, angle=angle)

    elif family == "slope":
        ramp = rng.uniform(0.25, MAX_RAMP_FRAC) * tan_theta
        angle = rng.uniform(-np.pi, np.pi)
        height = ramp * (xc * np.cos(angle) + yc * np.sin(angle))
        height -= height.mean()
        params.update(ramp=ramp, angle=angle)

    elif family == "pile":
        height = np.zeros(grid.shape, dtype=np.float64)
        n_piles = int(rng.integers(1, 3))
        for i in range(n_piles):
            apex = rng.uniform(0.35, 0.90)
            py = rng.uniform(0.25, 0.75) * grid.extent
            px = rng.uniform(0.25, 0.75) * grid.extent
            radius = np.hypot(yc - py, xc - px)
            # max(), not +=, so overlapping piles merge into one repose surface rather
            # than stacking into a doubled slope.
            height = np.maximum(height, apex - FLANK_FRAC * tan_theta * radius)
            params.update({f"apex_{i}": apex, f"pile_y_{i}": py, f"pile_x_{i}": px})
        params["n_piles"] = n_piles

    else:
        raise ValueError(f"unknown terrain family {family!r}; expected one of {FAMILIES}")

    height = height + _roughness(rng, grid, tan_theta)
    return height - height.mean(), params
