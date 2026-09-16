"""Episode-level invariants: the ones the headline metric depends on."""

import numpy as np
import pytest

from terrain.action import Action, sample
from terrain.grid import DEFAULT_GRID as GRID
from terrain.relax import TOL_FRAC, max_slope
from terrain.sim import ExcavationSim, SoilParams, apply_action
from terrain.terrains import FAMILIES

CASES = [("flat", 33.0), ("trench", 28.0), ("slope", 40.0), ("pile", 25.0)]


def _episode(family, theta_deg, seed=11, n=20, action_seed=99, swell=1.25):
    sim = ExcavationSim(family, SoilParams(theta_deg, swell), seed=seed)
    rng = np.random.default_rng(action_seed)
    for _ in range(n):
        sim.step(sample(rng))
    return sim


@pytest.mark.parametrize("family,theta_deg", CASES)
def test_mass_balance_holds_to_machine_precision(family, theta_deg):
    """The invariant is not sum(h) = const -- swell deliberately creates height.

    Loosened soil occupies more volume than it did in the bank, so what must hold is
    V_after - V_before == (swell - 1) * removed. This is the number that makes a learned
    model's conservation violation attributable to the model rather than to the data.
    """
    sim = _episode(family, theta_deg)
    for record in sim.records:
        assert abs(record.residual) < 1e-12


@pytest.mark.parametrize("family,theta_deg", CASES)
def test_mass_error_does_not_accumulate_over_an_episode(family, theta_deg):
    swell = 1.25
    sim = ExcavationSim(family, SoilParams(theta_deg, swell), seed=11)
    rng = np.random.default_rng(99)
    expected = GRID.volume(sim.height)
    for _ in range(50):
        height, record = sim.step(sample(rng))
        expected += (swell - 1.0) * record.removed
        assert abs(GRID.volume(height) - expected) < 1e-11


def test_deposited_volume_matches_the_swelled_cut():
    sim = _episode("flat", 33.0)
    for record in sim.records:
        assert abs(record.deposited - record.removed * 1.25) < 1e-12


@pytest.mark.parametrize("offset", [-1.0, 0.5, 2.0])
def test_a_step_is_invariant_to_a_constant_height_offset(offset):
    """No bedrock anywhere, so raising the whole world must just raise the answer.

    This is what licenses subtracting a per-window mean height when normalising inputs
    for the model. If it ever fails, that normalisation is a lie the model gets punished
    for at test time.
    """
    rng = np.random.default_rng(4)
    terrain = rng.normal(0.0, 0.1, GRID.shape)
    kwargs = dict(tan_theta=float(np.tan(np.radians(33.0))), swell=1.25)
    for _ in range(10):
        action = sample(rng)
        base, base_rec = apply_action(terrain, action, **kwargs)
        lifted, lifted_rec = apply_action(terrain + offset, action, **kwargs)
        assert np.abs(lifted - offset - base).max() < 1e-11
        assert abs(lifted_rec.removed - base_rec.removed) < 1e-12


def test_episodes_are_reproducible_from_their_seed():
    left = _episode("trench", 30.0, seed=7)
    right = _episode("trench", 30.0, seed=7)
    assert np.array_equal(left.height, right.height)
    assert [r.removed for r in left.records] == [r.removed for r in right.records]

    other = _episode("trench", 30.0, seed=8)
    assert not np.array_equal(left.height, other.height)


@pytest.mark.parametrize("family,theta_deg", CASES)
def test_surfaces_stay_at_repose_throughout(family, theta_deg):
    sim = _episode(family, theta_deg)
    tan_theta = sim.soil.tan_theta
    assert max_slope(sim.height, GRID.dx).max() <= tan_theta * (1.0 + 2.0 * TOL_FRAC)
    assert not any(r.hit_cap for r in sim.records)


@pytest.mark.parametrize("family", FAMILIES)
def test_episodes_do_not_run_away(family):
    """Heights must stay inside the range the int16 frame encoding can hold."""
    sim = _episode(family, 30.0, n=50)
    assert np.abs(sim.height).max() < 6.0
    assert all(abs(r.h_min) < 6.0 and abs(r.h_max) < 6.0 for r in sim.records)


def test_digging_the_same_spot_forever_is_self_limiting():
    """The classic failure mode for a cut defined against the local surface."""
    sim = ExcavationSim("flat", SoilParams(30.0, 1.25), seed=2)
    action = Action(
        entry_y=3.2, entry_x=2.0, heading=0.0, sweep_len=2.0,
        depth_1=0.15, depth_2=0.30, depth_3=0.25, width=0.7,
        dump_y=5.5, dump_x=5.5,
    )
    depths = []
    for _ in range(40):
        height, _ = sim.step(action)
        depths.append(float(height.min()))
    assert depths[-1] > -2.0
    # The crater widens instead of deepening, so late passes barely move the floor.
    assert abs(depths[-1] - depths[-5]) < 0.02


def test_an_action_that_removes_nothing_is_still_a_valid_step():
    """Worth keeping in the dataset: "this action does nothing here" is learnable."""
    terrain = GRID.zeros()
    action = Action(
        entry_y=3.2, entry_x=3.2, heading=0.0, sweep_len=1.0,
        depth_1=0.0, depth_2=0.0, depth_3=0.0, width=0.5,
        dump_y=1.0, dump_x=1.0,
    )
    height, record = apply_action(
        terrain, action, tan_theta=float(np.tan(np.radians(30.0))), swell=1.25
    )
    assert record.empty_cut
    assert record.removed == 0.0
    assert record.deposited == 0.0
    assert np.array_equal(height, terrain)


def test_a_zero_depth_pass_skims_the_high_spots():
    """With the floor at the entry height, anything standing above it still comes off."""
    sim = ExcavationSim("flat", SoilParams(30.0, 1.25), seed=3)
    action = Action(
        entry_y=3.2, entry_x=3.2, heading=0.0, sweep_len=1.0,
        depth_1=0.0, depth_2=0.0, depth_3=0.0, width=0.5,
        dump_y=1.0, dump_x=1.0,
    )
    _, record = sim.step(action)
    assert 0.0 < record.removed < 1e-4
    assert abs(record.residual) < 1e-12
