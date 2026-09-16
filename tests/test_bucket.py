"""Properties of the swept bucket cut."""

import numpy as np

from terrain.action import BOUNDS
from terrain.bucket import swept_cut
from terrain.grid import DEFAULT_GRID as GRID

BASE_CUT = dict(
    entry_y=1.5, entry_x=1.5, heading=0.6, sweep_len=1.8,
    depths=(0.12, 0.20, 0.05), width=0.6, capacity=1e9,
)


def _random_cut(rng):
    return dict(
        entry_y=rng.uniform(*BOUNDS["entry_y"]),
        entry_x=rng.uniform(*BOUNDS["entry_x"]),
        heading=rng.uniform(*BOUNDS["heading"]),
        sweep_len=rng.uniform(*BOUNDS["sweep_len"]),
        depths=tuple(np.sort(rng.uniform(0.0, 0.35, 3))),
        width=rng.uniform(*BOUNDS["width"]),
        capacity=rng.uniform(0.02, 0.30),
    )


def test_cut_is_idempotent_when_the_bucket_does_not_fill():
    """A rigid trajectory sweeps the same volume of space, which is now empty.

    This is the property that rules out the inescapable-pit failure mode: a cut defined
    relative to the *local* surface digs deeper every time it is repeated.
    """
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(150):
        terrain = rng.normal(0.0, 0.08, GRID.shape)
        spec = _random_cut(rng)
        spec["capacity"] = 1e9
        once, info = swept_cut(terrain, GRID, **spec)
        if info.empty_cut:
            continue
        twice, again = swept_cut(once, GRID, **spec)
        # Not bitwise: (h - (h - floor)) - floor lands within an ulp of zero rather
        # than on it, so the second pass shaves off round-off and nothing more.
        assert again.removed < 1e-15
        assert np.abs(twice - once).max() < 1e-12
        checked += 1
    assert checked > 100


def test_capacity_limits_removal_exactly():
    terrain = GRID.zeros()
    _, full = swept_cut(terrain, GRID, **BASE_CUT)
    for capacity in (0.30, 0.08, 0.02, 0.001):
        _, info = swept_cut(terrain, GRID, **{**BASE_CUT, "capacity": capacity})
        assert abs(info.removed - min(full.v_geom, capacity)) < 1e-12
        assert info.capacity_clipped == (full.v_geom > capacity)


def test_capacity_truncates_along_track():
    """The bucket fills as it advances, so a smaller bucket stops sooner."""
    terrain = GRID.zeros()
    fills = [swept_cut(terrain, GRID, **{**BASE_CUT, "capacity": c})[1].s_fill
             for c in (0.02, 0.05, 0.09)]
    assert fills == sorted(fills)
    assert all(0.0 <= f <= BASE_CUT["sweep_len"] for f in fills)


def test_repeated_passes_clear_the_trench_then_stop():
    """A capacity-limited bucket needs several passes, but must not dig without bound."""
    spec = dict(entry_y=3.2, entry_x=1.0, heading=0.0, sweep_len=2.0,
                depths=(0.15, 0.30, 0.30), width=0.7, capacity=0.06)
    terrain = GRID.zeros()
    removals = []
    for _ in range(15):
        terrain, info = swept_cut(terrain, GRID, **spec)
        removals.append(info.removed)
    assert removals[-1] == 0.0
    assert terrain.min() >= -max(spec["depths"]) - 1e-12


def test_terrain_shape_changes_what_comes_out():
    """Digging uphill removes more than downhill: the whole point of a rigid path."""
    _, xc = GRID.cell_centres()
    ramp = xc * 0.30
    spec = dict(entry_y=3.2, entry_x=3.2, sweep_len=1.8,
                depths=(0.12, 0.20, 0.05), width=0.6, capacity=1e9)
    uphill = swept_cut(ramp, GRID, heading=0.0, **spec)[1].removed
    downhill = swept_cut(ramp, GRID, heading=np.pi, **spec)[1].removed
    assert uphill > 10.0 * max(downhill, 1e-9)


def test_cut_only_ever_removes():
    rng = np.random.default_rng(3)
    for _ in range(50):
        terrain = rng.normal(0.0, 0.1, GRID.shape)
        out, info = swept_cut(terrain, GRID, **_random_cut(rng))
        assert np.all(out <= terrain + 1e-15)
        assert info.removed >= 0.0
        assert info.removed <= info.v_geom + 1e-12


def test_sweep_off_the_grid_is_handled():
    """Cuts near the wall are legitimate and informative, not something to reject."""
    terrain = GRID.zeros()
    out, info = swept_cut(terrain, GRID, entry_y=0.05, entry_x=0.05, heading=np.pi,
                          sweep_len=2.4, depths=(0.2, 0.3, 0.3), width=0.9, capacity=1e9)
    assert info.empty_cut or info.removed >= 0.0
    assert np.isfinite(out).all()


def test_ground_already_below_the_bucket_path_is_left_alone():
    """The bucket cannot take what is not in its way, so a deeper pit survives intact."""
    terrain = GRID.zeros()
    pit = np.s_[55:75, 20:110]
    terrain[pit] = -0.9  # far below any floor this action reaches
    # Enter on the flat, sweep across the pit. The entry reference must sit on the
    # flat ground, or the bucket path drops with it and digs the pit floor deeper.
    spec = dict(entry_y=2.0, entry_x=3.0, heading=np.pi / 2, sweep_len=2.0,
                depths=(0.10, 0.20, 0.10), width=0.8, capacity=1e9)
    out, info = swept_cut(terrain, GRID, **spec)
    assert np.array_equal(out[pit], terrain[pit])
    assert info.removed > 0.0  # it still cut the flat ground on either side
