"""Training loop.

Every run gets the same optimiser, schedule, batch size and *number of gradient steps*.
Holding the step count fixed matters for the data-size sweep in particular: if an epoch
were one pass over the data, a run on 250 episodes would get a small fraction of the
updates a run on 2,000 gets, and the sweep would confound how much data there is with
how long the model trained.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.data.dataset import Batch, EpisodeStore, WindowSampler
from wm.data.normalize import DELTA_SCALE
from wm.models import build_model, count_parameters

# The widest horizon any run trains on. Windows are always sampled this long and the
# loss is truncated for shorter curricula, so the distribution of start states never
# shifts as the curriculum advances.
MAX_K = 5

STEPS_PER_EPOCH = 1000
ACTIVE_CELL_THRESHOLD = 1e-3  # metres; "this cell actually moved"


@dataclass
class RunConfig:
    run_id: str
    arch: str = "latentB"
    latent_dim: int = 64
    use_action: bool = True
    k_train: int = 5
    train_size: int = 2000
    seed: int = 0

    data_root: str = "data/v1"
    train_split: str = "train"
    dev_split: str = "dev"

    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 30
    warmup_steps: int = 500
    min_lr: float = 3e-6
    grad_clip: float = 1.0

    # Protocol B cannot be teacher-forced, so it gets a curriculum on horizon instead.
    curriculum: tuple[tuple[int, int], ...] = ((0, 1), (3, 2), (5, 3), (7, 5))
    # Protocol A can be, and must end at zero or it is never trained in the regime it
    # is evaluated in.
    teacher_forcing_epochs: int = 10

    loss: str = "huber"
    # See volume_penalty(): quarantined, and never on for a run whose volume number is
    # reported as a headline result.
    uses_volume_loss: bool = False
    volume_loss_weight: float = 0.3
    early_stop_patience: int = 5

    def curriculum_k(self, epoch: int) -> int:
        if self.k_train == 1:
            return 1
        k = 1
        for start, value in self.curriculum:
            if epoch >= start:
                k = value
        return min(k, self.k_train)

    def teacher_prob(self, epoch: int) -> float:
        if self.teacher_forcing_epochs <= 0:
            return 0.0
        return max(0.0, 1.0 - epoch / self.teacher_forcing_epochs)


# Transition point of the Huber loss, in delta-scale units -- so exactly one delta
# scale, 2 cm. Not tuned against any reported metric.
HUBER_BETA = 1.0


def rollout_loss(
    pred: torch.Tensor, target: torch.Tensor, kind: str = "huber"
) -> tuple[torch.Tensor, list[float]]:
    """Per-step error over the rollout, in units of the delta scale.

    Huber rather than plain L1, and this is a measured decision rather than a taste.
    Only about 5% of cells move in a given step, so 95% of targets are an exact zero
    delta. L1's gradient has constant magnitude regardless of how wrong a cell is, so
    the static majority outvotes the handful of cells that actually changed, and
    predicting no change everywhere is not merely a way to score well -- it is a genuine
    optimisation attractor. Measured on a fixed batch of eight windows the model could
    not escape it at any learning rate: after 400 steps the predicted delta had been
    driven to 4e-6 m against a target of 5.2e-3 m, the loss sat exactly on the identity
    baseline, and the activity ratio was 0.000.

    Huber is L1 for large errors, so the big displacements a dig produces are not
    blurred, but quadratic near zero, so a nearly-correct static cell contributes a
    nearly-zero gradient instead of a full-sized one. On the same batch it reached 3.4 mm
    masked error at an activity ratio of 1.04. Plain L1 and L2 remain selectable so the
    collapse is reproducible as an ablation row.

    The *reported* metrics are unchanged and remain absolute error: this decides how the
    models are trained, not how they are scored.

    Uniform weight across the horizon. Discounting later steps would penalise precisely
    the horizons the project is about, and 1/k normalisation would hide error growth.
    """
    err = (pred - target) / DELTA_SCALE
    if kind == "huber":
        per_step = F.smooth_l1_loss(err, torch.zeros_like(err), beta=HUBER_BETA, reduction="none")
        per_step = per_step.mean(dim=(0, 2, 3))
    elif kind == "l1":
        per_step = err.abs().mean(dim=(0, 2, 3))
    elif kind == "l2":
        per_step = err.pow(2).mean(dim=(0, 2, 3))
    else:
        raise ValueError(f"unknown loss {kind!r}")
    return per_step.mean(), [float(v) for v in per_step.detach().cpu()]


def rollout_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute error in delta-scale units. The reported quantity, never the objective."""
    return ((pred - target) / DELTA_SCALE).abs().mean()


def volume_penalty(pred: torch.Tensor, target: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """How far the predicted change in total material is from the true change.

    QUARANTINED. This must not appear in the objective of any run whose volume error is
    reported as a headline number. The claim under test is that a generically trained
    world model violates mass conservation; optimising the conservation residual directly
    reduces that claim to "we optimised X and X is low", which is the first thing a
    reader would object to.

    It exists so the question can be asked separately and honestly -- does forcing volume
    to balance also fix the angle-of-repose violations? -- and make_tables.py refuses to
    put runs with and without it in the same table.
    """
    pred_change = (pred - start.unsqueeze(1)).mean(dim=(2, 3))
    true_change = (target - start.unsqueeze(1)).mean(dim=(2, 3))
    return ((pred_change - true_change) / DELTA_SCALE).abs().mean()


@torch.no_grad()
def diagnostics(pred: torch.Tensor, target: torch.Tensor, start: torch.Tensor) -> dict:
    """Numbers worth watching that are deliberately not in the loss."""
    err = (pred - target).abs()
    moved = (target - start.unsqueeze(1)).abs() > ACTIVE_CELL_THRESHOLD
    activity = (pred - start.unsqueeze(1)).abs().sum() / (target - start.unsqueeze(1)).abs().sum().clamp_min(1e-9)
    return {
        "mae_mm": float(err.mean()) * 1000.0,
        "masked_mae_mm": float(err[moved].mean()) * 1000.0 if moved.any() else float("nan"),
        "active_fraction": float(moved.float().mean()),
        "activity_ratio": float(activity),
    }


def lr_at(step: int, cfg: RunConfig, total: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


def make_optimizer(model: nn.Module, cfg: RunConfig) -> torch.optim.Optimizer:
    """Weight decay on weights only -- never on norms or biases."""
    decayed, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (plain if param.ndim <= 1 else decayed).append(param)
    return torch.optim.AdamW(
        [{"params": decayed, "weight_decay": cfg.weight_decay},
         {"params": plain, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,
    )


class Trainer:
    def __init__(self, cfg: RunConfig, out_dir: Path, device: str = "cuda") -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        root = Path(cfg.data_root)
        self.train_store = EpisodeStore(root, cfg.train_split, limit=cfg.train_size)
        self.dev_store = EpisodeStore(root, cfg.dev_split)
        self.train_sampler = WindowSampler(
            self.train_store, MAX_K, cfg.batch_size, self.device, seed=cfg.seed
        )
        self.dev_sampler = WindowSampler(
            self.dev_store, MAX_K, cfg.batch_size, self.device, seed=cfg.seed + 7919
        )

        self.model = build_model(
            cfg.arch, latent_dim=cfg.latent_dim, use_action=cfg.use_action
        ).to(self.device)
        self.optimizer = make_optimizer(self.model, cfg)
        self.total_steps = cfg.epochs * STEPS_PER_EPOCH
        self.step = 0
        self.history: list[dict] = []

    def _forward(self, batch: Batch, k: int, teacher_prob: float):
        start = batch.frames[:, 0]
        target = batch.frames[:, 1:k + 1]
        actions = batch.actions[:, :k]
        teacher = batch.frames[:, 1:k + 1] if teacher_prob > 0.0 else None
        pred, norms = self.model.rollout(
            start, actions, teacher_frames=teacher, teacher_prob=teacher_prob
        )
        return start, target, pred, norms

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        k = self.cfg.curriculum_k(epoch)
        teacher_prob = self.cfg.teacher_prob(epoch) if self.model.protocol == "A" else 0.0

        running, grads = [], []
        started = time.perf_counter()
        for _ in range(STEPS_PER_EPOCH):
            lr = lr_at(self.step, self.cfg, self.total_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            batch = self.train_sampler.sample()
            start, target, pred, _ = self._forward(batch, k, teacher_prob)
            loss, _ = rollout_loss(pred, target, self.cfg.loss)
            if self.cfg.uses_volume_loss:
                loss = loss + self.cfg.volume_loss_weight * volume_penalty(pred, target, start)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            running.append(float(loss.detach()))
            grads.append(float(norm))
            self.step += 1

        return {
            "epoch": epoch, "k": k, "teacher_prob": round(teacher_prob, 3),
            "lr": lr, "train_loss": float(np.mean(running)),
            "grad_norm": float(np.mean(grads)),
            "seconds": round(time.perf_counter() - started, 1),
        }

    @torch.no_grad()
    def evaluate(self, n_batches: int = 24) -> dict:
        """Dev-set five-step L1. This alone decides the checkpoint.

        Never the volume metric and never the fifty-step error: selecting a checkpoint
        on a proxy for the headline number is how a result stops meaning anything.
        """
        self.model.eval()
        losses, diags = [], []
        for batch in self.dev_sampler.fixed_batches(n_batches):
            start, target, pred, _ = self._forward(batch, self.cfg.k_train, 0.0)
            # Selection is on absolute error, whatever the training objective was, so
            # checkpoints stay comparable across the loss ablation.
            losses.append(float(rollout_mae(pred, target)))
            diags.append(diagnostics(pred, target, start))
        summary = {"dev_loss": float(np.mean(losses))}
        for key in diags[0]:
            summary[f"dev_{key}"] = float(np.nanmean([d[key] for d in diags]))
        return summary

    def fit(self) -> dict:
        best = math.inf
        best_epoch = -1
        log_path = self.out_dir / "train_log.jsonl"
        log_path.write_text("")

        baseline = self.evaluate()
        print(f"  epoch  -1 (untrained = identity)  dev_loss={baseline['dev_loss']:.4f} "
              f"mae={baseline['dev_mae_mm']:.2f}mm", flush=True)

        for epoch in range(self.cfg.epochs):
            record = self.train_epoch(epoch)
            record.update(self.evaluate())
            self.history.append(record)
            with log_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"  epoch {epoch:3d}  k={record['k']}  train={record['train_loss']:.4f}  "
                  f"dev={record['dev_loss']:.4f}  mae={record['dev_mae_mm']:.2f}mm  "
                  f"masked={record['dev_masked_mae_mm']:.1f}mm  {record['seconds']:.0f}s", flush=True)

            if record["dev_loss"] < best - 1e-5:
                best, best_epoch = record["dev_loss"], epoch
                torch.save({"model": self.model.state_dict(), "epoch": epoch,
                            "dev_loss": best, "config": asdict(self.cfg)},
                           self.out_dir / "ckpt_best.pt")
            elif epoch - best_epoch >= self.cfg.early_stop_patience:
                print(f"  early stop: no improvement since epoch {best_epoch}", flush=True)
                break

        torch.save({"model": self.model.state_dict(), "epoch": epoch,
                    "config": asdict(self.cfg)}, self.out_dir / "ckpt_last.pt")
        summary = {
            "run_id": self.cfg.run_id,
            "best_dev_loss": best,
            "best_epoch": best_epoch,
            "epochs_run": len(self.history),
            "identity_dev_loss": baseline["dev_loss"],
            "parameters": count_parameters(self.model),
            "train_episodes": len(self.train_store),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
