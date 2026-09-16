"""Validation figures for the simulator itself.

    python scripts/make_sim_figures.py

The mass-balance panel is the one that matters. A learned model's conservation error only
means something if the ground truth it was trained on conserves mass to machine
precision, so that number gets published rather than asserted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from terrain.action import sample  # noqa: E402
from terrain.grid import DEFAULT_GRID as GRID  # noqa: E402
from terrain.relax import TOL_FRAC, max_slope  # noqa: E402
from terrain.sim import ExcavationSim, SoilParams  # noqa: E402
from terrain.terrains import FAMILIES  # noqa: E402

N_STEPS = 50
SWELL = 1.25

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def run_episode(family: str, theta_deg: float, seed: int = 11, action_seed: int = 99):
    sim = ExcavationSim(family, SoilParams(theta_deg, SWELL), seed=seed)
    rng = np.random.default_rng(action_seed)
    frames = [sim.height.copy()]
    for _ in range(N_STEPS):
        frames.append(sim.step(sample(rng))[0].copy())
    return sim, np.array(frames)


def terrain_gallery(out: Path) -> None:
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(3.1 * len(FAMILIES), 6.2))
    for column, family in enumerate(FAMILIES):
        _, frames = run_episode(family, 30.0)
        span = (frames.min(), frames.max())
        for row, step in enumerate((0, N_STEPS)):
            ax = axes[row, column]
            ax.imshow(frames[step], cmap="terrain", vmin=span[0], vmax=span[1],
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(family, fontsize=10)
            ax.set_ylabel(f"step {step}", fontsize=8)
    fig.suptitle("Terrain families before and after fifty dig-swing-dump cycles", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "terrain_families.png", bbox_inches="tight")
    plt.close(fig)


def conservation_figure(out: Path) -> None:
    cases = [("flat", 33.0), ("trench", 28.0), ("slope", 40.0), ("pile", 25.0)]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))

    for family, theta in cases:
        sim, frames = run_episode(family, theta)
        removed = np.array([r.removed for r in sim.records])
        residual = np.array([r.residual for r in sim.records])
        label = f"{family}, {theta:.0f} deg"

        expected = GRID.volume(frames[0]) + (SWELL - 1.0) * np.cumsum(removed)
        actual = np.array([GRID.volume(f) for f in frames[1:]])
        axes[0].plot(np.arange(1, N_STEPS + 1), np.abs(actual - expected), lw=1.3, label=label)

        axes[1].plot(np.arange(1, N_STEPS + 1), np.abs(residual), lw=1.3, label=label)

        slope = np.array([max_slope(f, GRID.dx).max() for f in frames]) / sim.soil.tan_theta
        axes[2].plot(slope, lw=1.3, label=label)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("cumulative mass error (m$^3$)")
    axes[0].set_title("Volume tracks (swell - 1) x excavated")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("per-step residual (m$^3$)")
    axes[1].set_title("Mass balance holds at machine precision")
    axes[2].axhline(1.0 + 2 * TOL_FRAC, color="crimson", ls="--", lw=1)
    axes[2].set_ylim(0.0, 1.2)
    axes[2].set_ylabel("steepest slope / tan(theta)")
    axes[2].set_title("Every surface sits at or below repose")

    for ax in axes:
        ax.set_xlabel("dig-swing-dump cycle")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "sim_conservation.png", bbox_inches="tight")
    plt.close(fig)


def dynamics_figure(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))
    for family, theta in [("flat", 33.0), ("trench", 28.0), ("slope", 40.0), ("pile", 25.0)]:
        sim, frames = run_episode(family, theta)
        label = f"{family}, {theta:.0f} deg"
        axes[0].plot([f.min() for f in frames], lw=1.2, label=label)
        axes[0].plot([f.max() for f in frames], lw=1.2, ls="--",
                     color=axes[0].lines[-1].get_color())
        axes[1].plot([r.removed for r in sim.records], lw=1.1, label=label)
        axes[2].plot(np.cumsum([r.capacity_clipped for r in sim.records]), lw=1.2, label=label)

    axes[0].set_ylabel("height extremes (m)")
    axes[0].set_title("Episodes stay bounded (solid min, dashed max)")
    axes[1].set_ylabel("material removed (m$^3$)")
    axes[1].set_title("Per-cycle bucket volume")
    axes[2].set_ylabel("cycles with a full bucket")
    axes[2].set_title("Capacity limit binds on about a third")
    for ax in axes:
        ax.set_xlabel("dig-swing-dump cycle")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "sim_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    terrain_gallery(args.out)
    conservation_figure(args.out)
    dynamics_figure(args.out)
    print(f"simulator figures written to {args.out}")


if __name__ == "__main__":
    main()
