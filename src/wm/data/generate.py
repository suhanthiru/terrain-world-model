"""Episode generation and the train / test split definitions.

Splits are laid out as a factorial over the two shift axes the project cares about --
terrain family and soil angle -- plus a held-out action policy. Every evaluation split
is 200 episodes and none of them is ever trained on.

Seed ranges are disjoint by construction rather than by chance. Drawing seeds at random
across 7,200 episodes carries about a one percent chance of a collision, which is small
but catastrophic if it lands, since a duplicated episode would sit on both sides of a
train/test boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from terrain.action import RAW_DIM, sample
from terrain.grid import DEFAULT_GRID
from terrain.sim import ExcavationSim, SoilParams

N_STEPS = 50

# Swell is held fixed across the whole dataset. It could vary -- the simulator supports
# it -- but a constant swell is what makes the identity baseline land on exactly
# -(swell - 1) = -25% on the volume metric at every horizon, which is a free and exact
# reference line on the headline figure. Varying it would smear that line for no gain on
# any axis the ablations actually test.
SWELL = 1.25

TRAIN_THETA = 30.0
TRANSFER_THETA = 40.0

# Bands are 1e8 apart, leaving room for retry seeds without any chance of one split
# reaching into another.
_BAND = 100_000_000
_RETRY_STRIDE = 10_000_000
MAX_RETRIES = 3


@dataclass(frozen=True)
class SplitSpec:
    n_episodes: int
    families: tuple[str, ...]
    theta_deg: float
    policy: str
    seed_base: int

    def episode_plan(self) -> list[tuple[int, str]]:
        """(seed, family) for each episode, families alternating so counts stay even."""
        return [
            (self.seed_base + i, self.families[i % len(self.families)])
            for i in range(self.n_episodes)
        ]


SPLITS: dict[str, SplitSpec] = {
    "train": SplitSpec(6000, ("flat", "trench"), TRAIN_THETA, "heuristic", 0 * _BAND),
    "dev": SplitSpec(200, ("flat", "trench"), TRAIN_THETA, "heuristic", 1 * _BAND),
    "test_indist": SplitSpec(200, ("flat", "trench"), TRAIN_THETA, "heuristic", 2 * _BAND),
    "test_ood_terrain": SplitSpec(200, ("slope", "pile"), TRAIN_THETA, "heuristic", 3 * _BAND),
    "test_ood_soil": SplitSpec(200, ("flat", "trench"), TRANSFER_THETA, "heuristic", 4 * _BAND),
    "test_ood_both": SplitSpec(200, ("slope", "pile"), TRANSFER_THETA, "heuristic", 5 * _BAND),
    "test_ood_policy": SplitSpec(200, ("flat", "trench"), TRAIN_THETA, "uniform", 6 * _BAND),
}

# Nested prefixes of the train split, so the data-size sweep varies only how much data
# there is and never which data it is.
TRAIN_SIZES = (250, 500, 1000, 2000, 4000, 6000)


def assert_seed_ranges_disjoint() -> None:
    seen: dict[int, str] = {}
    for name, spec in SPLITS.items():
        for offset in range(MAX_RETRIES + 1):
            lo = spec.seed_base + offset * _RETRY_STRIDE
            for seed in (lo, lo + spec.n_episodes - 1):
                for other, owner in seen.items():
                    if owner != name and abs(seed - other) < spec.n_episodes:
                        raise AssertionError(f"seed bands for {name} and {owner} overlap")
            seen[lo] = name


def generate_episode(split: str, seed: int, family: str) -> dict:
    """Run one full episode and return everything needed to store it.

    Retries with a seed from the same split's reserved band if the relaxation fails to
    settle. An unsettled surface is not at repose, so keeping it would quietly poison
    both the physics metrics and the dynamics.
    """
    spec = SPLITS[split]
    soil = SoilParams(spec.theta_deg, SWELL)
    grid = DEFAULT_GRID

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        trial_seed = seed + attempt * _RETRY_STRIDE
        try:
            sim = ExcavationSim(family, soil, seed=trial_seed, grid=grid)
            action_rng = np.random.default_rng([trial_seed, 1])

            frames = np.empty((N_STEPS + 1, *grid.shape), dtype=np.float64)
            actions = np.empty((N_STEPS, RAW_DIM), dtype=np.float64)
            volumes = {
                "removed": np.empty(N_STEPS), "v_geom": np.empty(N_STEPS),
                "deposited": np.empty(N_STEPS), "residual": np.empty(N_STEPS),
                "total": np.empty(N_STEPS + 1),
            }

            frames[0] = sim.height
            volumes["total"][0] = grid.volume(sim.height)
            clipped = 0
            empty = 0
            for t in range(N_STEPS):
                action = sample(action_rng, policy=spec.policy)
                height, record = sim.step(action)
                if record.hit_cap:
                    raise RuntimeError(f"relaxation hit the sweep cap at step {t}")
                frames[t + 1] = height
                actions[t] = action.to_vector()
                volumes["removed"][t] = record.removed
                volumes["v_geom"][t] = record.v_geom
                volumes["deposited"][t] = record.deposited
                volumes["residual"][t] = record.residual
                volumes["total"][t + 1] = record.v_after
                clipped += record.capacity_clipped
                empty += record.empty_cut

            return {
                "frames": frames,
                "actions": actions.astype(np.float32),
                "volumes": volumes,
                "meta": {
                    "episode_uid": f"{split}-{seed:09d}",
                    "seed": trial_seed,
                    "requested_seed": seed,
                    "attempts": attempt + 1,
                    "split": split,
                    "terrain_family": family,
                    "theta_deg": spec.theta_deg,
                    "swell": SWELL,
                    "policy": spec.policy,
                    "terrain_params": {k: float(v) for k, v in sim.terrain_params.items()},
                    "h_min": float(frames.min()),
                    "h_max": float(frames.max()),
                    "n_capacity_clipped": int(clipped),
                    "n_empty_cuts": int(empty),
                    "max_abs_residual": float(np.abs(volumes["residual"]).max()),
                },
            }
        except (RuntimeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(f"episode {split}/{seed} failed after {MAX_RETRIES + 1} attempts: {last_error}")


def _worker(task: tuple[str, int, str]) -> dict:
    """Top-level so it survives the pickling that Windows spawn requires."""
    return generate_episode(*task)
