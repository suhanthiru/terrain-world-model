"""Properties of the angle-of-repose relaxation."""

import numpy as np
import pytest

from terrain.grid import DEFAULT_GRID as GRID
from terrain.relax import (
    ALPHA, W_DIAG, TOL_FRAC, _lambda_max, critical_drops, max_slope, max_violation, relax,
)

TAN_33 = float(np.tan(np.radians(33.0)))


def _insults():
    """Surfaces that are badly out of equilibrium, one per interesting mechanism."""
    spike = GRID.zeros(); spike[64, 64] = 3.0
    pit = GRID.zeros(); pit[40:88, 54:74] -= 0.35
    step = GRID.zeros(); step[:, 64:] = 0.8
    rng = np.random.default_rng(0)
    rough = rng.normal(0.0, 0.4, GRID.shape)
    return {"spike": spike, "pit": pit, "step": step, "rough": rough}


@pytest.mark.parametrize("name", list(_insults()))
def test_relaxation_conserves_mass_exactly(name):
    """Conservation is structural, so this should sit at float64 round-off."""
    h = _insults()[name]
    before = GRID.volume(h)
    out, info = relax(h, TAN_33, GRID.dx)
    assert not info.hit_cap
    assert abs(GRID.volume(out) - before) < 1e-12


@pytest.mark.parametrize("theta_deg", [25.0, 33.0, 45.0])
def test_no_pair_exceeds_the_repose_angle(theta_deg):
    tan_theta = float(np.tan(np.radians(theta_deg)))
    d_axial, d_diag = critical_drops(tan_theta, GRID.dx)
    for h in _insults().values():
        out, info = relax(h, tan_theta, GRID.dx)
        assert not info.hit_cap
        # Convergence is declared at TOL_FRAC of the axial critical drop, so the
        # converged surface may sit that far above critical and no further.
        assert max_violation(out, d_axial, d_diag) < TOL_FRAC * d_axial
        assert max_slope(out, GRID.dx).max() <= tan_theta * (1.0 + 2.0 * TOL_FRAC)


def test_relaxation_does_not_teleport_material():
    """Catches np.roll, whose wraparound is mass-conserving but physically wrong.

    Periodic boundaries pass every conservation test while moving material clean across
    the domain, so this is the check that actually pins the boundary condition down.
    """
    h = GRID.zeros()
    h[10, 10] = 5.0
    out, _ = relax(h, TAN_33, GRID.dx)
    far = out[100:, 100:]
    assert np.all(far == 0.0)


def test_relaxation_is_offset_invariant():
    """The simulator has no bedrock, so adding a constant must commute with relaxing."""
    h = _insults()["rough"]
    base, _ = relax(h, TAN_33, GRID.dx)
    for offset in (-1.0, 0.5, 2.0):
        shifted, _ = relax(h + offset, TAN_33, GRID.dx)
        assert np.abs(shifted - offset - base).max() < 1e-12


def test_settled_surface_does_not_drift():
    """Re-relaxing a settled surface converges immediately and barely moves it.

    It is not a strict fixed point: convergence is declared at a tolerance, so a few
    edges are still marginally active and another pass nudges them. What matters is
    that the movement stays bounded by that tolerance instead of accumulating, which is
    what would make the sweep count a hidden state variable.
    """
    tol = TOL_FRAC * critical_drops(TAN_33, GRID.dx)[0]
    once, _ = relax(_insults()["pit"], TAN_33, GRID.dx)
    twice, info = relax(once, TAN_33, GRID.dx)
    assert info.n_sweeps <= 8
    assert np.abs(twice - once).max() < 2.0 * tol
    thrice, _ = relax(twice, TAN_33, GRID.dx)
    assert np.abs(thrice - once).max() < 2.0 * tol


def test_stability_bound_is_enforced():
    """alpha above 2 / lambda_max diverges, so it must be refused rather than run."""
    assert _lambda_max(W_DIAG) == 8.0
    assert _lambda_max(1.0) == 12.0
    assert ALPHA < 2.0 / _lambda_max(W_DIAG)
    with pytest.raises(ValueError, match="stability bound"):
        relax(GRID.zeros(), TAN_33, GRID.dx, alpha=0.26)
    # The 8-neighbour lattice with equally weighted diagonals is the tighter case.
    with pytest.raises(ValueError, match="stability bound"):
        relax(GRID.zeros(), TAN_33, GRID.dx, alpha=0.20, w_diag=1.0)
