"""Properties of the evaluation metrics.

These run against episodes generated on the spot, so they do not depend on a dataset
having been built.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from terrain.action import sample
from terrain.grid import DEFAULT_GRID as GRID
from terrain.sim import ExcavationSim, SoilParams
from wm.eval import metrics
from wm.eval.failures import classify

SWELL = 1.25
THETA = 30.0
N_STEPS = 12


@pytest.fixture(scope="module")
def episode():
    sim = ExcavationSim("flat", SoilParams(THETA, SWELL), seed=5)
    rng = np.random.default_rng(17)
    frames = [sim.height.copy()]
    removed = []
    for _ in range(N_STEPS):
        height, record = sim.step(sample(rng))
        frames.append(height.copy())
        removed.append(record.removed)
    return np.array(frames), np.array(removed), float(np.tan(np.radians(THETA)))


def test_identity_scores_exactly_minus_the_swell_excess(episode):
    """The gate that has to pass before any model exists.

    Predicting no change means predicting that none of the swell ever appeared, so the
    volume error is exactly -(swell - 1) at every horizon. If this is not exact, the
    metric is wrong, and no amount of model training would reveal it.
    """
    frames, removed, _ = episode
    start = frames[0]
    pred = np.repeat(start[None], N_STEPS, axis=0)
    result = metrics.volume_error(pred, start, removed, SWELL)
    assert np.allclose(result["eps_net"], -(SWELL - 1.0), atol=1e-12)


def test_perfect_prediction_scores_zero_volume_error(episode):
    frames, removed, _ = episode
    result = metrics.volume_error(frames[1:], frames[0], removed, SWELL)
    # Bounded by the storage quantisation, not by the metric.
    assert np.abs(result["eps_net"]).max() < 1e-9


def test_volume_error_is_signed_and_reads_as_invented_material(episode):
    """Adding material must read positive, removing it negative."""
    frames, removed, _ = episode
    extra = frames[1:] + 0.01  # a centimetre of dirt from nowhere, everywhere
    result = metrics.volume_error(extra, frames[0], removed, SWELL)
    invented = 0.01 * GRID.n**2 * GRID.cell_area
    assert np.all(result["eps_net"] > 0)
    assert np.isclose(result["dv_pred"][-1] - frames[1:].sum(axis=(1, 2))[-1] * GRID.cell_area
                      + frames[0].sum() * GRID.cell_area, invented, rtol=1e-6)


def test_the_simulator_satisfies_its_own_repose_constraint(episode):
    """Ground truth must be physical, or the physics metrics measure the simulator."""
    frames, _, tan_theta = episode
    assert metrics.repose_violation(frames[1:], tan_theta).max() == 0.0


def test_identity_has_no_repose_violations(episode):
    """Which is exactly why a physics metric is never reported on its own."""
    frames, _, tan_theta = episode
    pred = np.repeat(frames[0][None], N_STEPS, axis=0)
    assert metrics.repose_violation(pred, tan_theta).max() == 0.0


def test_repose_violation_catches_an_over_steep_surface(episode):
    frames, _, tan_theta = episode
    steep = frames[1:].copy()
    steep[:, 60:64, :] += 1.0  # a metre-high wall over one cell
    assert metrics.repose_violation(steep, tan_theta).min() > 0.0


def test_high_frequency_ratio_separates_blur_from_noise(episode):
    frames, _, _ = episode
    truth = frames[1:]
    assert np.allclose(metrics.high_frequency_ratio(truth, truth), 1.0)

    blurred = np.stack([gaussian_filter(f, 2.0) for f in truth])
    assert np.nanmean(metrics.high_frequency_ratio(blurred, truth)) < 0.9

    noisy = truth + np.random.default_rng(0).normal(0.0, 0.002, truth.shape)
    assert np.nanmean(metrics.high_frequency_ratio(noisy, truth)) > 1.1


def test_checkerboard_score_fires_on_a_checkerboard(episode):
    frames, _, _ = episode
    truth = frames[1:]
    board = np.indices(GRID.shape).sum(axis=0) % 2
    speckled = truth + 0.01 * (2 * board - 1)
    assert np.nanmean(metrics.checkerboard_score(speckled, truth)) > 3.0
    assert np.nanmean(metrics.error_sign_alternation(speckled, truth)) > 0.85


def test_activity_ratio_is_zero_for_identity_and_one_for_truth(episode):
    frames, _, _ = episode
    start, truth = frames[0], frames[1:]
    assert np.allclose(metrics.activity_ratio(np.repeat(start[None], N_STEPS, 0), truth, start), 0.0)
    assert np.allclose(metrics.activity_ratio(truth, truth, start), 1.0)


def test_centroid_shift_detects_a_hole_in_the_wrong_place(episode):
    frames, _, _ = episode
    start, truth = frames[0], frames[1:]
    shifted = np.stack([np.roll(f - start, 12, axis=1) + start for f in truth])
    assert np.nanmean(metrics.excavation_centroid_shift(shifted, truth, start)) > 4.0
    # The shape is intact, which is what separates displacement from noise.
    assert np.nanmean(metrics.shift_corrected_correlation(shifted, truth, start)) > 0.9


def test_effective_swell_recovers_the_true_value(episode):
    """Material going up over material going down should track the swell factor."""
    frames, _, _ = episode
    ratio = metrics.displaced_volumes(frames[1:], frames[0])["effective_swell"]
    assert 1.0 < np.nanmedian(ratio) < 1.6


def test_failure_classifier_labels_the_obvious_cases(episode):
    frames, removed, tan_theta = episode
    start, truth = frames[0], frames[1:]

    def scores_for(pred):
        from wm.eval.rollout import score_episode
        return score_episode(pred, frames, removed, tan_theta, SWELL)

    identity = np.repeat(start[None], N_STEPS, axis=0)
    assert classify(scores_for(identity), identity, N_STEPS)["label"] == "identity_collapse"

    perfect = truth.copy()
    assert classify(scores_for(perfect), perfect, N_STEPS)["label"] == "ok"

    exploded = truth + 0.05
    assert classify(scores_for(exploded), exploded, N_STEPS)["flag_volume_explosion"]

    blown = truth * 1e3
    assert classify(scores_for(blown), blown, N_STEPS)["label"] == "diverged"
