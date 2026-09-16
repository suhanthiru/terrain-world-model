"""Evaluate a checkpoint (or an untrained baseline) on every held-out split.

    python scripts/evaluate.py --run-id latentB_K5_z64_n2000_s0
    python scripts/evaluate.py --run-id identity --arch identity

Writes per-episode metrics as parquet plus an aggregated summary, and appends a row per
(run, split, horizon) to results/index.csv. Figures and tables are built only from those
files, never by re-running a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from wm.data.dataset import EpisodeStore  # noqa: E402
from wm.data.storage import read_manifest  # noqa: E402
from wm.eval.failures import AUDIT_HORIZONS, classify, summarise  # noqa: E402
from wm.eval.rollout import rollout_episodes, score_episode  # noqa: E402
from wm.models import build_model, count_parameters  # noqa: E402

EVAL_SPLITS = (
    "test_indist", "test_ood_terrain", "test_ood_soil", "test_ood_both", "test_ood_policy", "dev",
)
REPORT_HORIZONS = (1, 5, 10, 20, 50)

# Curves are summarised by the median with a 10-90 band rather than a mean with a
# standard deviation. The error distribution is heavy tailed because a handful of
# episodes diverge, and a mean tracks those rather than the typical case -- so the mean
# is reported too, in its own column, since the gap between them is itself the story.
PERCENTILES = (10, 50, 90)

ARRAY_METRICS = (
    "mae", "rmse", "max_ae", "masked_mae", "eps_net", "dv_pred", "dv_true",
    "repose_violation", "repose_violation_true", "hf_ratio", "checkerboard",
    "sign_alternation", "activity_ratio", "centroid_shift", "shift_corrected_ncc",
    "effective_swell", "displaced_error",
)


def load_model(run_dir: Path, arch: str | None, device: str):
    """An explicit --arch means an untrained baseline; otherwise restore the checkpoint."""
    if arch is not None:
        return build_model(arch).to(device), {"arch": arch, "untrained": True}

    payload = torch.load(run_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    cfg = payload["config"]
    model = build_model(cfg["arch"], latent_dim=cfg["latent_dim"], use_action=cfg["use_action"])
    model.load_state_dict(payload["model"])
    return model.to(device), cfg


def evaluate_split(model, store: EpisodeStore, swell: float, device: str,
                   protocol: str | None) -> tuple[pd.DataFrame, list[dict]]:
    predictions = rollout_episodes(model, store, device=device, protocol=protocol)

    rows, failures = [], []
    for e in range(len(store)):
        frames = store.episode_frames(e).astype(np.float64)
        tan_theta = float(np.tan(np.radians(store.meta[e]["theta_deg"])))
        scores = score_episode(
            predictions[e].astype(np.float64), frames, store.volumes["removed"][e],
            tan_theta, swell,
        )
        for k in range(predictions.shape[1]):
            rows.append({
                "episode": e, "episode_uid": store.meta[e]["episode_uid"],
                "terrain_family": store.meta[e]["terrain_family"],
                "theta_deg": store.meta[e]["theta_deg"], "horizon": k + 1,
                **{name: float(scores[name][k]) for name in ARRAY_METRICS},
            })
        for horizon in AUDIT_HORIZONS:
            failures.append({
                "episode": e, "episode_uid": store.meta[e]["episode_uid"],
                **classify(scores, predictions[e], horizon),
            })
    return pd.DataFrame(rows), failures


def summarise_split(frame: pd.DataFrame) -> dict:
    out = {}
    for horizon in REPORT_HORIZONS:
        at = frame[frame["horizon"] == horizon]
        if at.empty:
            continue
        entry = {}
        for name in ARRAY_METRICS:
            values = at[name].to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            p10, p50, p90 = np.percentile(finite, PERCENTILES)
            entry[name] = {"p10": float(p10), "median": float(p50), "p90": float(p90),
                           "mean": float(finite.mean())}
        out[str(horizon)] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arch", default=None, help="evaluate an untrained baseline instead")
    parser.add_argument("--protocol", default=None, help="override the rollout protocol")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "v1")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--splits", nargs="*", default=list(EVAL_SPLITS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.runs) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(run_dir, args.arch, args.device)
    swell = read_manifest(args.data_root)["swell"]

    index_rows = []
    for split in args.splits:
        started = time.perf_counter()
        store = EpisodeStore(args.data_root, split)
        frame, failures = evaluate_split(model, store, swell, args.device, args.protocol)

        out_dir = run_dir / "eval" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_dir / "per_episode.parquet", index=False)

        summary = summarise_split(frame)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "failures.json").write_text(json.dumps({
            "thresholds_frozen_on": "dev, baseline models only",
            "by_horizon": {
                str(h): summarise([f for f in failures if f["horizon"] == h])
                for h in AUDIT_HORIZONS
            },
            "episodes": failures,
        }, indent=2))

        for horizon, entry in summary.items():
            index_rows.append({
                "run_id": args.run_id, "eval_split": split, "horizon": int(horizon),
                "arch": cfg.get("arch"), "latent_dim": cfg.get("latent_dim"),
                "k_train": cfg.get("k_train"), "train_size": cfg.get("train_size"),
                "seed": cfg.get("seed"), "use_action": cfg.get("use_action", True),
                "loss": cfg.get("loss", "n/a"),
                # Read by make_tables.py, which refuses to mix these in one table.
                "uses_volume_loss": cfg.get("uses_volume_loss", False),
                "parameters": count_parameters(model),
                **{f"{name}_{stat}": entry[name][stat]
                   for name in entry for stat in ("median", "p10", "p90", "mean")},
            })
        print(f"  {split}: {len(store)} episodes in {time.perf_counter() - started:.0f}s  "
              f"median MAE@20 = {summary.get('20', {}).get('mae', {}).get('median', float('nan')) * 1000:.2f} mm  "
              f"eps_net@20 = {summary.get('20', {}).get('eps_net', {}).get('median', float('nan')) * 100:+.2f}%",
              flush=True)

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    index_path = results / "index.csv"
    frame = pd.DataFrame(index_rows)
    if index_path.exists():
        existing = pd.read_csv(index_path)
        keys = ["run_id", "eval_split", "horizon"]
        existing = existing.merge(frame[keys], on=keys, how="left", indicator=True)
        existing = existing[existing["_merge"] == "left_only"].drop(columns="_merge")
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(index_path, index=False)
    print(f"wrote {len(index_rows)} rows to {index_path}")


if __name__ == "__main__":
    main()
