"""Build the figures from disk.

    python scripts/make_figures.py

Like make_tables.py, this imports neither torch nor the model package. If a figure
cannot be drawn from results/index.csv and the per-run summaries, that is a sign the
evaluation did not record something it should have.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

TRAINING_HORIZON = 5
IDENTITY_VOLUME_ERROR = -0.25
BASELINES = ("identity", "mean_terrain")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def mark_training_horizon(ax) -> None:
    """Models train to five steps and are evaluated to fifty. Say so on the figure."""
    ax.axvline(TRAINING_HORIZON, color="0.4", ls=":", lw=1)
    ax.annotate("trained to here", xy=(TRAINING_HORIZON, 0.02), xycoords=("data", "axes fraction"),
                rotation=90, fontsize=7, color="0.4", ha="right", va="bottom")


def load_curves(runs_dir: Path, split: str) -> dict[str, dict]:
    """Full per-horizon curves from each run's summary.json.

    Still only reading files the evaluation wrote -- no checkpoint is loaded and torch is
    never imported, so figures regenerate in seconds without a GPU.
    """
    curves = {}
    for path in sorted(runs_dir.glob(f"*/eval/{split}/summary.json")):
        curves[path.parents[2].name] = json.loads(path.read_text())
    return curves


def curve(curves: dict, run: str, metric: str, stat: str = "median"):
    entry = curves.get(run, {})
    horizons = sorted(int(k) for k in entry)
    values = [entry[str(k)].get(metric, {}).get(stat, np.nan) for k in horizons]
    return np.array(horizons), np.array(values, dtype=float)


def series(frame: pd.DataFrame, run: str, split: str, column: str):
    at = frame[(frame["run_id"] == run) & (frame["eval_split"] == split)].sort_values("horizon")
    return at["horizon"].to_numpy(), at[column].to_numpy()


def horizon_curves(curves: dict, split: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    runs = [r for r in curves if r not in BASELINES]

    for ax, (metric, label) in zip(axes, [("mae", "all cells"),
                                          ("masked_mae", "cells that moved")]):
        for run in runs:
            k, value = curve(curves, run, metric)
            if len(k):
                ax.plot(k, value * 1000, lw=1.4, label=run)
        for name, style in zip(BASELINES, ["--", ":"]):
            k, value = curve(curves, name, metric)
            if len(k):
                ax.plot(k, value * 1000, style, color="0.3", lw=1.2, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step")
        ax.set_ylabel("median absolute error (mm)")
        ax.set_title(f"Prediction error, {label}")
        mark_training_horizon(ax)

    axes[1].legend(fontsize=6.5, loc="upper left", ncol=1)
    fig.suptitle(f"Horizon error on {split}", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"horizon_{split}.png", bbox_inches="tight")
    plt.close(fig)


def volume_figure(curves: dict, split: str, out: Path) -> None:
    """The headline. Signed, so hallucinated material reads positive."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for run in [r for r in curves if r not in BASELINES]:
        k, median = curve(curves, run, "eps_net")
        if not len(k):
            continue
        line, = ax.plot(k, median * 100, lw=1.6, label=run)
        _, low = curve(curves, run, "eps_net", "p10")
        _, high = curve(curves, run, "eps_net", "p90")
        if len(low) == len(k):
            ax.fill_between(k, low * 100, high * 100, alpha=0.12, color=line.get_color())

    ax.axhline(0, color="0.2", lw=1)
    ax.axhline(IDENTITY_VOLUME_ERROR * 100, color="crimson", ls="--", lw=1.2)
    ax.annotate("predicting no change: exactly -25%", xy=(0.98, IDENTITY_VOLUME_ERROR * 100),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=7, color="crimson")
    mark_training_horizon(ax)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("volume error, % of material excavated")
    ax.set_title("Does the model conserve mass?")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"volume_{split}.png", bbox_inches="tight")
    plt.close(fig)


def accuracy_versus_physics(frame: pd.DataFrame, split: str, out: Path, horizon: int = 20) -> None:
    """Never plot a physics metric on its own.

    Predicting nothing scores perfectly on repose violation and beats most models on
    volume, so a physics number without an accuracy number beside it rewards doing
    nothing. A useful model has to reach the lower-left of both panels.
    """
    at = frame[(frame["eval_split"] == split) & (frame["horizon"] == horizon)]
    if at.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, (column, label) in zip(axes, [
        ("eps_net_median", f"|volume error| at k={horizon} (%)"),
        ("repose_violation_median", f"repose violation at k={horizon} (% of cells)"),
    ]):
        for _, row in at.iterrows():
            if column not in row or not np.isfinite(row[column]):
                continue
            x = row["mae_median"] * 1000
            y = abs(row[column]) * 100
            baseline = row["run_id"] in BASELINES
            ax.scatter(x, y, s=46 if baseline else 30,
                       marker="X" if baseline else "o",
                       color="crimson" if baseline else "steelblue", zorder=3)
            ax.annotate(row["run_id"], (x, y), fontsize=5.5,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel(f"median MAE at k={horizon} (mm)")
        ax.set_ylabel(label)
        ax.set_title("useful models land lower-left")
    fig.suptitle(f"Accuracy against physics plausibility, {split}", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"accuracy_vs_physics_{split}.png", bbox_inches="tight")
    plt.close(fig)


def sweep_figure(frame: pd.DataFrame, split: str, out: Path, horizon: int = 20) -> None:
    at = frame[(frame["eval_split"] == split) & (frame["horizon"] == horizon)
               & (frame["arch"] == "latentB") & (frame["k_train"] == 5)
               & frame["use_action"].fillna(True)]
    if at.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    sizes = at[at["latent_dim"] == 64].sort_values("train_size")
    if len(sizes) > 1:
        axes[0].plot(sizes["train_size"], sizes["mae_median"] * 1000, "o-", lw=1.5)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("training episodes (nested prefixes)")

    dims = at[at["train_size"] == 2000].sort_values("latent_dim")
    if len(dims) > 1:
        axes[1].plot(dims["latent_dim"], dims["mae_median"] * 1000, "s-", lw=1.5, color="darkorange")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("latent dimension")

    for ax in axes:
        ax.set_ylabel(f"median MAE at k={horizon} (mm)")
    fig.suptitle("Ablation sweeps", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sweeps.png", bbox_inches="tight")
    plt.close(fig)


def transfer_figure(frame: pd.DataFrame, out: Path, horizon: int = 20) -> None:
    splits = ["test_indist", "test_ood_soil", "test_ood_terrain", "test_ood_both", "test_ood_policy"]
    labels = ["in\ndistribution", "soil\ntransfer", "terrain\ntransfer", "both", "held-out\npolicy"]
    at = frame[(frame["horizon"] == horizon) & frame["eval_split"].isin(splits)]
    if at.empty:
        return
    runs = [r for r in at["run_id"].unique() if r not in BASELINES][:6]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.8 / max(len(runs), 1)
    for i, run in enumerate(runs):
        values = [at[(at["run_id"] == run) & (at["eval_split"] == s)]["mae_median"].mean() * 1000
                  for s in splits]
        ax.bar(np.arange(len(splits)) + i * width, values, width, label=run)
    identity = [at[(at["run_id"] == "identity") & (at["eval_split"] == s)]["mae_median"].mean() * 1000
                for s in splits]
    if np.isfinite(identity).any():
        ax.plot(np.arange(len(splits)) + 0.4 - width / 2, identity, "kX--", ms=7,
                lw=1, label="identity")
    ax.set_xticks(np.arange(len(splits)) + 0.4 - width / 2)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(f"median MAE at k={horizon} (mm)")
    ax.set_title("Distribution shift: trained on flat and trench at 30 degrees")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "transfer.png", bbox_inches="tight")
    plt.close(fig)


def failure_figure(runs_dir: Path, split: str, out: Path) -> None:
    rows = []
    for failures in sorted(runs_dir.glob(f"*/eval/{split}/failures.json")):
        run = failures.parents[2].name
        payload = json.loads(failures.read_text())
        for horizon, summary in payload.get("by_horizon", {}).items():
            for label, count in summary.get("labels", {}).items():
                rows.append({"run_id": run, "horizon": int(horizon), "label": label,
                             "fraction": count / max(summary["n_episodes"], 1)})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    horizons = sorted(frame["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 4.0), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        at = frame[frame["horizon"] == horizon].pivot_table(
            index="run_id", columns="label", values="fraction", aggfunc="sum").fillna(0.0)
        at.plot(kind="barh", stacked=True, ax=ax, width=0.75, legend=(horizon == horizons[-1]))
        ax.set_title(f"failure modes at k={horizon}")
        ax.set_xlabel("fraction of episodes")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
        if ax.get_legend():
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / f"failures_{split}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--split", default="test_indist")
    args = parser.parse_args()

    index = args.results / "index.csv"
    if not index.exists():
        raise SystemExit(f"{index} not found -- run scripts/evaluate.py first")
    frame = pd.read_csv(index)

    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    curves = load_curves(args.runs, args.split)
    horizon_curves(curves, args.split, out)
    volume_figure(curves, args.split, out)
    accuracy_versus_physics(frame, args.split, out)
    sweep_figure(frame, args.split, out)
    transfer_figure(frame, out)
    failure_figure(args.runs, args.split, out)
    print(f"figures written to {out}")
    for path in sorted(out.glob("*.png")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
