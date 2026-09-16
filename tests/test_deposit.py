"""Properties of the water-filling deposit."""

import numpy as np
import pytest

from terrain.deposit import deposit, deposit_apex_by_bisection
from terrain.grid import DEFAULT_GRID as GRID
from terrain.relax import critical_drops, max_violation, relax

TAN_33 = float(np.tan(np.radians(33.0)))
PLACES = {"centre": (3.2, 3.2), "edge": (0.05, 3.2), "corner": (0.02, 0.02)}
VOLUMES = (1e-5, 1e-3, 0.05, 0.5, 2.0)


def _pit():
    h = GRID.zeros()
    h[30:70, 40:80] -= 0.30
    return h


def _terrains():
    rng = np.random.default_rng(0)
    return {
        "flat": GRID.zeros(),  # heavy exact ties in the radius
        "offset": GRID.zeros() + 0.7,
        "rough": rng.normal(0.0, 0.05, GRID.shape),
        "dug": relax(_pit(), TAN_33, GRID.dx)[0],
    }


@pytest.mark.parametrize("place", list(PLACES))
@pytest.mark.parametrize("volume", VOLUMES)
def test_places_exactly_the_requested_volume(place, volume):
    """Including where the wall clips the cone: it piles higher, it never loses mass."""
    y, x = PLACES[place]
    out, info = deposit(GRID.zeros(), GRID, y, x, volume, TAN_33)
    assert abs(info.placed - volume) / volume < 1e-12
    assert abs(GRID.volume(out) - volume) < 1e-12


@pytest.mark.parametrize("name", list(_terrains()))
def test_exact_on_every_terrain(name):
    terrain = _terrains()[name]
    before = GRID.volume(terrain)
    out, info = deposit(terrain, GRID, 3.2, 3.2, 0.35, TAN_33)
    assert abs(info.placed - 0.35) / 0.35 < 1e-12
    assert abs(GRID.volume(out) - before - 0.35) < 1e-12


def test_clipped_cone_piles_higher_rather_than_wider():
    apexes = [deposit(GRID.zeros(), GRID, y, x, 2.0, TAN_33)[1].apex
              for y, x in (PLACES["centre"], PLACES["edge"], PLACES["corner"])]
    assert apexes[0] < apexes[1] < apexes[2]


def test_closed_form_matches_the_bisection_oracle():
    for name, terrain in _terrains().items():
        for volume in VOLUMES:
            _, info = deposit(terrain, GRID, 3.2, 3.2, volume, TAN_33)
            oracle = deposit_apex_by_bisection(terrain, GRID, 3.2, 3.2, volume, TAN_33)
            assert abs(info.apex - oracle) < 1e-9, (name, volume)


def test_deposit_never_makes_the_surface_steeper():
    """max() of two surfaces at repose is at repose, which is why relaxing after is free.

    Adding a cone instead would stack slope on slope: on sloping ground that gives
    roughly 2*tan(theta), and the relaxation then has to take it back apart.
    """
    d_axial, d_diag = critical_drops(TAN_33, GRID.dx)
    for name, terrain in _terrains().items():
        settled, _ = relax(terrain, TAN_33, GRID.dx)
        before = max_violation(settled, d_axial, d_diag)
        out, _ = deposit(settled, GRID, 3.2, 3.2, 0.35, TAN_33)
        assert max_violation(out, d_axial, d_diag) <= before + 1e-15, name


def test_relaxing_after_a_deposit_is_a_no_op():
    settled, _ = relax(_pit(), TAN_33, GRID.dx)
    piled, _ = deposit(settled, GRID, 2.0, 2.0, 0.35, TAN_33)
    _, info = relax(piled, TAN_33, GRID.dx)
    assert info.n_sweeps <= 8


def test_zero_volume_is_a_no_op():
    terrain = _terrains()["rough"]
    out, info = deposit(terrain, GRID, 3.2, 3.2, 0.0, TAN_33)
    assert info.placed == 0.0
    assert info.n_active == 0
    assert np.array_equal(out, terrain)


def test_deposit_only_ever_adds():
    terrain = _terrains()["rough"]
    out, _ = deposit(terrain, GRID, 3.2, 3.2, 0.35, TAN_33)
    assert np.all(out >= terrain - 1e-15)
