"""Properties of the on-disk format.

The point of most of these is that storage precision must not reach the headline volume
metric. Frames are quantised; the volume bookkeeping is not.
"""

import numpy as np
import pytest

from terrain.grid import DEFAULT_GRID as GRID
from wm.data.generate import SPLITS, assert_seed_ranges_disjoint, generate_episode
from wm.data.storage import (
    HEIGHT_LIMIT, HEIGHT_SCALE, build_window_index, decode_heights, encode_heights,
)


def test_quantisation_error_is_uniform_and_bounded():
    """Fixed-point, not float16: the error must not grow with height.

    float16 would be the same two bytes but its error scales with magnitude, and -- worse
    -- is identical for every cell sitting at the same unrepresentable height. Excavation
    terrain is full of flat regions, so those errors add coherently instead of cancelling.
    """
    rng = np.random.default_rng(0)
    for level in (0.0, 0.5, 2.0, 5.0):
        values = level + rng.normal(0.0, 0.05, (64, 64))
        error = np.abs(decode_heights(encode_heights(values)) - values)
        # Half a quantisation step, plus the float32 representation error of the decode
        # itself -- frames come back as float32 because that is what feeds the model.
        bound = HEIGHT_SCALE / 2 + float(np.spacing(np.float32(max(level, 1.0))))
        assert error.max() <= bound

    # The same field in float16 gets steadily worse as it rises; fixed point does not.
    high = 5.0 + rng.normal(0.0, 0.05, (64, 64))
    assert np.abs(high.astype(np.float16) - high).max() > 10 * HEIGHT_SCALE / 2


def test_encoding_refuses_runaway_heights():
    """A silent clip would corrupt the volume metric; a loud failure will not."""
    with pytest.raises(ValueError, match="exceeds"):
        encode_heights(np.full((8, 8), HEIGHT_LIMIT + 1.0))


def test_flat_regions_do_not_accumulate_coherent_error():
    """The specific failure mode fixed-point is chosen to avoid."""
    flat = np.full(GRID.shape, 1.2345678)
    error = GRID.volume(decode_heights(encode_heights(flat)).astype(np.float64)) - GRID.volume(flat)
    # A whole-grid coherent offset of half a quantisation step is the worst possible case.
    assert abs(error) <= GRID.n**2 * GRID.cell_area * HEIGHT_SCALE / 2 + 1e-9
    assert abs(error) < 0.06  # cubic metres, against episodes that move several


def test_window_index_covers_every_valid_start():
    index = build_window_index(n_episodes=3, n_steps=50, k=5)
    assert index.shape == (3 * 46, 2)
    assert index[:, 1].max() == 45  # start 45 reads frames 45..50
    assert index.dtype == np.int32


def test_seed_ranges_are_disjoint_by_construction():
    """Random seeds across 7,200 episodes collide about 1% of the time.

    Small, but an episode landing on both sides of a train/test boundary is exactly the
    leak that invalidates every generalisation number, so the bands are assigned rather
    than drawn.
    """
    assert_seed_ranges_disjoint()
    bases = sorted(spec.seed_base for spec in SPLITS.values())
    assert len(set(bases)) == len(bases)
    for lo, hi in zip(bases, bases[1:]):
        assert hi - lo >= max(s.n_episodes for s in SPLITS.values())


@pytest.mark.slow
def test_generated_episode_round_trips_through_storage():
    episode = generate_episode("dev", SPLITS["dev"].seed_base, "flat")
    frames = episode["frames"]
    volumes = episode["volumes"]

    assert frames.shape == (51, *GRID.shape)
    assert np.abs(volumes["residual"]).max() < 1e-12
    assert np.allclose(volumes["deposited"], volumes["removed"] * 1.25, atol=1e-12)

    # The metric is a difference, so the start-frame offset cancels on both sides and
    # only the per-frame quantisation noise survives.
    quantised = decode_heights(encode_heights(frames)).astype(np.float64)
    exact_change = volumes["total"] - volumes["total"][0]
    stored_change = (quantised.sum(axis=(1, 2)) - quantised[0].sum()) * GRID.cell_area
    excavated = np.cumsum(volumes["removed"])

    contamination = np.abs(stored_change[1:] - exact_change[1:]) / excavated
    assert contamination.max() < 1e-3, "storage noise is within a thousandth of the signal"
