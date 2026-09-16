"""Training-loop properties.

Mostly one thing: the sparse-delta prediction problem has a degenerate solution -- predict
no change anywhere -- and the choice of loss decides whether the optimiser falls into it.
That was found the hard way and is easy to reintroduce, so it gets a regression test.
"""

import numpy as np
import pytest
import torch

from wm.data.normalize import DELTA_SCALE, HEIGHT_SCALE, window_offset
from wm.models import build_model, count_parameters
from wm.train import RunConfig, lr_at, make_optimizer, rollout_loss, rollout_mae

ACTIVE_FRACTION = 0.05  # roughly what the simulator produces per step


def sparse_target(rng, batch=6, k=1, size=64):
    """A start surface plus a localised change, like one dig on flat ground."""
    start = torch.from_numpy(rng.normal(0.0, 0.1, (batch, size, size))).float()
    delta = torch.zeros(batch, k, size, size)
    for b in range(batch):
        y, x = rng.integers(8, size - 20, size=2)
        delta[b, :, y:y + 14, x:x + 10] = -0.1
    return start.cuda(), (start[:, None] + delta).cuda()


def _train(loss_kind, steps=300, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    start, target = sparse_target(rng)
    model = build_model("latentB", size=start.shape[-1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    actions = torch.randn(start.shape[0], target.shape[1], 11, device="cuda")
    for _ in range(steps):
        pred, _ = model.rollout(start, actions)
        loss, _ = rollout_loss(pred, target, loss_kind)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        pred, _ = model.rollout(start, actions)
        moved_pred = (pred - start[:, None]).abs().sum()
        moved_true = (target - start[:, None]).abs().sum()
        return {
            "mae": float(rollout_mae(pred, target)),
            "activity": float(moved_pred / moved_true),
        }


@pytest.mark.slow
def test_plain_l1_collapses_to_predicting_no_change():
    """The finding this project had to work around.

    About 95% of cells hold an exact zero delta, and L1's gradient has the same magnitude
    however wrong a cell is, so the static majority outvotes the few that moved. Doing
    nothing is not merely a good score, it is a fixed point the optimiser converges to.
    """
    result = _train("l1")
    assert result["activity"] < 0.05


@pytest.mark.slow
def test_huber_escapes_the_identity_solution():
    result = _train("huber")
    assert result["activity"] > 0.5
    assert result["mae"] < _train("l1")["mae"]


def test_untrained_model_starts_close_to_predicting_no_change():
    """Near the identity, which is right for most cells -- but not exactly on it.

    Exactly zero output weights would cut every upstream layer off from gradient
    entirely, since what reaches them is proportional to those weights.
    """
    torch.manual_seed(0)
    model = build_model("latentB").cuda()
    x0 = torch.randn(4, 128, 128, device="cuda") * 0.2
    actions = torch.randn(4, 5, 11, device="cuda")
    pred, _ = model.rollout(x0, actions)
    deviation = (pred - x0[:, None]).abs().max().item()
    assert 0.0 < deviation < 0.002


def test_every_parameter_receives_gradient():
    """Zero-initialised layers silently sever pathways; this catches that class of bug."""
    for arch in ("latentA", "latentB", "unet"):
        torch.manual_seed(0)
        model = build_model(arch).cuda()
        x0 = torch.randn(2, 128, 128, device="cuda") * 0.2
        actions = torch.randn(2, 3, 11, device="cuda")
        target = x0[:, None].expand(-1, 3, -1, -1) + torch.randn(2, 3, 128, 128, device="cuda") * 0.02
        loss, _ = rollout_loss(model.rollout(x0, actions)[0], target)
        loss.backward()
        dead = [n for n, p in model.named_parameters()
                if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        assert not dead, f"{arch} has parameters with no gradient: {dead}"


def test_curriculum_never_exceeds_the_configured_horizon():
    five = RunConfig(run_id="x", k_train=5)
    assert [five.curriculum_k(e) for e in (0, 2, 3, 5, 7, 29)] == [1, 1, 2, 3, 5, 5]

    one = RunConfig(run_id="x", k_train=1)
    assert all(one.curriculum_k(e) == 1 for e in range(30))


def test_teacher_forcing_reaches_zero():
    """Ending above zero would mean never training in the regime that is evaluated."""
    cfg = RunConfig(run_id="x", teacher_forcing_epochs=10)
    assert cfg.teacher_prob(0) == 1.0
    assert cfg.teacher_prob(5) == pytest.approx(0.5)
    assert cfg.teacher_prob(10) == 0.0
    assert cfg.teacher_prob(29) == 0.0


def test_learning_rate_warms_up_then_decays():
    cfg = RunConfig(run_id="x", lr=3e-4, warmup_steps=500, min_lr=3e-6)
    assert lr_at(0, cfg, 30000) == pytest.approx(cfg.lr / 500)
    assert lr_at(499, cfg, 30000) == pytest.approx(cfg.lr)
    assert lr_at(29999, cfg, 30000) == pytest.approx(cfg.min_lr, abs=1e-7)


def test_weight_decay_skips_norms_and_biases():
    model = build_model("latentB")
    opt = make_optimizer(model, RunConfig(run_id="x", weight_decay=1e-4))
    decayed, plain = opt.param_groups
    assert decayed["weight_decay"] == 1e-4
    assert plain["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decayed["params"])
    assert all(p.ndim <= 1 for p in plain["params"])


def test_latent_and_unet_parameter_counts_match():
    """The pixel-versus-latent ablation is only meaningful at matched capacity."""
    latent = count_parameters(build_model("latentB", latent_dim=64))
    unet = count_parameters(build_model("unet"))
    assert abs(unet - latent) / latent < 0.01


def test_window_offset_leaves_frame_to_frame_change_untouched():
    """A per-frame mean would move the target; a single per-window offset cannot."""
    rng = np.random.default_rng(0)
    frames = rng.normal(0.0, 0.3, (4, 6, 32, 32))
    offset = window_offset(frames[:, 0])
    shifted = frames - offset[:, None, None, None]
    assert np.allclose(np.diff(frames, axis=1), np.diff(shifted, axis=1))
    assert np.allclose(shifted[:, 0].mean(axis=(1, 2)), 0.0, atol=1e-12)


def test_early_stopping_cannot_fire_before_the_curriculum_finishes():
    """Otherwise a five-step run stops before ever training at five steps."""
    cfg = RunConfig(run_id="x", k_train=5, early_stop_patience=5)
    assert cfg.curriculum_complete_epoch() == 7
    # The horizon is still ramping through epoch 6, so stopping must not be possible
    # until well past it.
    assert cfg.curriculum_complete_epoch() + cfg.early_stop_patience == 12

    assert RunConfig(run_id="x", k_train=1).curriculum_complete_epoch() == 0
    assert RunConfig(run_id="x", k_train=3).curriculum_complete_epoch() == 5
