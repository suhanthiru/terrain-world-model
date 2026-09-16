"""Build the result tables from disk.

    python scripts/make_tables.py

This script deliberately imports neither torch nor the model package. Everything it
needs is in results/index.csv, which means tables regenerate in seconds, without a GPU,
and cannot silently disagree with what was actually evaluated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

HORIZONS = (1, 5, 20, 50)
TRAINING_HORIZON = 5
IDENTITY_VOLUME_ERROR = -0.25  # exactly -(swell - 1); see wm.eval.metrics.volume_error


def load(results: Path) -> pd.DataFrame:
    path = results / "index.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/evaluate.py first")
    return pd.read_csv(path)


def guard_volume_loss(frame: pd.DataFrame, allow_mixed: bool = False) -> None:
    """Refuse to put volume-loss and ordinary runs in the same table.

    The headline claim is that a generically trained world model violates mass
    conservation. A run that optimised the conservation residual directly is not
    evidence for or against that, and quietly averaging it in would turn the result into
    "we optimised X and X is low". Good intentions do not survive a deadline, so this is
    a failing check rather than a note in a docstring.
    """
    if allow_mixed or "uses_volume_loss" not in frame:
        return
    kinds = set(frame["uses_volume_loss"].dropna().unique())
    if len(kinds) > 1:
        raise SystemExit(
            "refusing to mix runs trained with and without the volume loss in one table; "
            "pass --allow-mixed if that is genuinely what you want"
        )


def pivot(frame: pd.DataFrame, split: str, metric: str, stat: str = "median") -> pd.DataFrame:
    column = f"{metric}_{stat}"
    subset = frame[(frame["eval_split"] == split) & frame["horizon"].isin(HORIZONS)]
    if subset.empty or column not in subset:
        return pd.DataFrame()
    return subset.pivot_table(index="run_id", columns="horizon", values=column, aggfunc="median")


def main_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Accuracy and physics side by side, because neither means anything alone."""
    mae = pivot(frame, split, "mae")
    masked = pivot(frame, split, "masked_mae")
    volume = pivot(frame, split, "eps_net")
    repose = pivot(frame, split, "repose_violation")

    rows = []
    for run in mae.index:
        meta = frame[frame["run_id"] == run].iloc[0]
        row = {"run_id": run, "params": int(meta["parameters"])}
        for k in HORIZONS:
            if k in mae.columns:
                row[f"MAE@{k} (mm)"] = round(mae.loc[run, k] * 1000, 3)
        if 20 in masked.columns:
            row["masked MAE@20 (mm)"] = round(masked.loc[run, 20] * 1000, 2)
        for k in (5, 20, 50):
            if k in volume.columns:
                row[f"volume err@{k} (%)"] = round(volume.loc[run, k] * 100, 2)
        if 20 in repose.columns:
            row["repose viol@20 (%)"] = round(repose.loc[run, 20] * 100, 3)
        row["beats identity to k"] = crossover(frame, run, split)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"MAE@{TRAINING_HORIZON} (mm)")


def crossover(frame: pd.DataFrame, run: str, split: str) -> str:
    """Largest horizon at which the model still beats predicting no change.

    Identity's error curve is bounded and saturating while a learned autoregressive
    model can diverge without limit, so this is the honest one-number answer to how far
    a world model can be trusted.
    """
    model = frame[(frame["run_id"] == run) & (frame["eval_split"] == split)]
    baseline = frame[(frame["run_id"] == "identity") & (frame["eval_split"] == split)]
    if model.empty or baseline.empty:
        return "n/a"
    merged = model.merge(baseline, on="horizon", suffixes=("", "_id")).sort_values("horizon")
    better = merged[merged["mae_median"] < merged["mae_median_id"]]
    if better.empty:
        return "never"
    worse = merged[merged["mae_median"] >= merged["mae_median_id"]]
    return ">50" if worse.empty else str(int(worse["horizon"].min()) - 1)


def transfer_table(frame: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """The terrain x soil factorial, so main effects separate from the interaction."""
    splits = ["test_indist", "test_ood_terrain", "test_ood_soil", "test_ood_both", "test_ood_policy"]
    at = frame[(frame["horizon"] == horizon) & frame["eval_split"].isin(splits)]
    if at.empty:
        return pd.DataFrame()
    out = at.pivot_table(index="run_id", columns="eval_split", values="mae_median", aggfunc="median")
    out = (out * 1000).round(2)
    return out.reindex(columns=[s for s in splits if s in out.columns])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--split", default="test_indist")
    parser.add_argument("--allow-mixed", action="store_true")
    parser.add_argument("--volume-loss-arm", action="store_true",
                        help="report the quarantined volume-loss runs instead")
    args = parser.parse_args()

    frame = load(args.results)
    # The quarantined arm is dropped by default rather than raising: it optimised the
    # conservation residual directly, so its volume number says nothing about whether a
    # generically trained model conserves mass. It is reported on its own.
    if "uses_volume_loss" in frame:
        quarantined = frame["uses_volume_loss"].fillna(False).astype(bool)
        keep = quarantined if args.volume_loss_arm else ~quarantined
        dropped = sorted(frame.loc[~keep, "run_id"].unique())
        frame = frame[keep]
        if dropped and not args.volume_loss_arm:
            print(f"excluding {len(dropped)} quarantined volume-loss run(s): "
                  + ", ".join(dropped))
            print("  (report them on their own with --volume-loss-arm)\n")
    guard_volume_loss(frame, args.allow_mixed)

    out_dir = args.results / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "main": main_table(frame, args.split),
        "transfer_mae_at_20": transfer_table(frame),
    }
    for name, table in tables.items():
        if table.empty:
            continue
        table.to_csv(out_dir / f"{name}.csv")
        print(f"\n### {name} ({args.split})\n")
        print(table.to_markdown(index=name != "main"))

    print(f"\nidentity volume error is exactly {IDENTITY_VOLUME_ERROR:+.0%} at every horizon.")
    print(f"tables written to {out_dir}")


if __name__ == "__main__":
    main()
