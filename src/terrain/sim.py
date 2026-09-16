"""The excavation simulator.

One action is one dig-swing-dump cycle, applied in four stages:

    cut -> slump -> deposit -> slump

Mass bookkeeping is exact in float64. The invariant is *not* that ``sum(h)`` stays
constant -- swell deliberately creates height, because loosened soil occupies more
volume than it did in the bank. What holds instead is

    V_after - V_before == (swell - 1) * removed

to about 1e-15 m^3 over a full episode. Getting that right is the whole point: it is
what makes a learned model's conservation violation attributable to the model rather
than to the ground truth it was trained on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .action import Action
from .bucket import swept_cut
from .deposit import deposit
from .grid import DEFAULT_GRID, Grid
from .relax import RelaxInfo, relax
from .terrains import make_terrain

# Bucket capacity in cubic metres per metre of bucket width. Tuned so the capacity limit
# binds on about a third of cycles (measured 33.3% over the four terrain families): the
# nonlinearity gets well exercised without dominating the dataset.
CAPACITY_PER_WIDTH = 0.22

# Angle of repose and bulk swell ranges spanned by the terrain families.
THETA_RANGE_DEG = (25.0, 45.0)
SWELL_RANGE = (1.15, 1.35)


@dataclass
class StepRecord:
    """Everything the generator needs to audit one cycle."""

    removed: float
    v_geom: float
    deposited: float
    v_before: float
    v_after: float
    residual: float           # mass-balance error; should sit at machine precision
    capacity_clipped: bool
    s_fill: float
    empty_cut: bool
    n_cells_cut: int
    apex: float
    n_active: int
    sweeps_cut: int
    sweeps_deposit: int
    slope_residual_cut: float
    slope_residual_deposit: float
    hit_cap: bool
    h_min: float
    h_max: float

    def as_dict(self) -> dict:
        return asdict(self)


def apply_action(
    height: np.ndarray,
    action: Action,
    *,
    grid: Grid = DEFAULT_GRID,
    tan_theta: float,
    swell: float,
    capacity: float | None = None,
) -> tuple[np.ndarray, StepRecord]:
    """Advance a surface by one dig-swing-dump cycle. Pure: the input is not modified."""
    if capacity is None:
        capacity = CAPACITY_PER_WIDTH * action.width

    v_before = grid.volume(height)

    cut, cut_info = swept_cut(
        height, grid,
        entry_y=action.entry_y, entry_x=action.entry_x, heading=action.heading,
        sweep_len=action.sweep_len, depths=action.depths, width=action.width,
        capacity=capacity,
    )
    slumped, relax_cut = relax(cut, tan_theta, grid.dx)

    piled, dep_info = deposit(
        slumped, grid, action.dump_y, action.dump_x, cut_info.removed * swell, tan_theta
    )
    # This second relaxation is almost always a no-op: max() of two surfaces at repose
    # is itself at repose. It is kept because a deposit landing on a *fresh* cut face
    # can still need a little settling, and because skipping it would make the
    # dynamics depend on which of those two cases you happen to be in.
    final, relax_dep = relax(piled, tan_theta, grid.dx)

    v_after = grid.volume(final)
    record = StepRecord(
        removed=cut_info.removed,
        v_geom=cut_info.v_geom,
        deposited=dep_info.placed,
        v_before=v_before,
        v_after=v_after,
        residual=v_after - v_before - (swell - 1.0) * cut_info.removed,
        capacity_clipped=cut_info.capacity_clipped,
        s_fill=cut_info.s_fill,
        empty_cut=cut_info.empty_cut,
        n_cells_cut=cut_info.n_cells,
        apex=dep_info.apex,
        n_active=dep_info.n_active,
        sweeps_cut=relax_cut.n_sweeps,
        sweeps_deposit=relax_dep.n_sweeps,
        slope_residual_cut=relax_cut.residual,
        slope_residual_deposit=relax_dep.residual,
        hit_cap=relax_cut.hit_cap or relax_dep.hit_cap,
        h_min=float(final.min()),
        h_max=float(final.max()),
    )
    return final, record


@dataclass
class SoilParams:
    theta_deg: float
    swell: float

    @property
    def tan_theta(self) -> float:
        return float(np.tan(np.radians(self.theta_deg)))


class ExcavationSim:
    """A single episode: an initial surface plus a sequence of dig-swing-dump cycles."""

    def __init__(
        self,
        family: str,
        soil: SoilParams,
        seed: int,
        *,
        grid: Grid = DEFAULT_GRID,
    ) -> None:
        self.grid = grid
        self.family = family
        self.soil = soil
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.terrain_params: dict = {}
        self.height = self._initial_surface()
        self.records: list[StepRecord] = []

    def _initial_surface(self) -> np.ndarray:
        raw, params = make_terrain(self.family, self.rng, self.soil.tan_theta, self.grid)
        self.terrain_params = params
        settled, info = relax(raw, self.soil.tan_theta, self.grid.dx)
        if info.hit_cap:
            raise RuntimeError(
                f"initial {self.family} surface failed to settle in {info.n_sweeps} sweeps"
            )
        return settled

    def step(self, action: Action) -> tuple[np.ndarray, StepRecord]:
        self.height, record = apply_action(
            self.height, action,
            grid=self.grid, tan_theta=self.soil.tan_theta, swell=self.soil.swell,
        )
        self.records.append(record)
        return self.height, record
