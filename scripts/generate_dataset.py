"""Generate the episode dataset.

    python scripts/generate_dataset.py --root data/v1
    python scripts/generate_dataset.py --root data/smoke --splits dev --limit 8 --workers 4

Episodes are embarrassingly parallel, so this fans out over processes and writes each
shard as its episodes come back.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from terrain.action import BOUNDS, FIELDS, MODEL_DIM, MODEL_FIELDS, RAW_DIM  # noqa: E402
from terrain.grid import DEFAULT_GRID  # noqa: E402
from terrain.relax import ALPHA, TOL_FRAC, W_DIAG  # noqa: E402
from terrain.sim import CAPACITY_PER_WIDTH  # noqa: E402
from wm.data.generate import (  # noqa: E402
    MAX_RETRIES, N_STEPS, SPLITS, SWELL, TRAIN_SIZES, _worker, assert_seed_ranges_disjoint,
)
from wm.runlog import redirect_output  # noqa: E402
from wm.data.storage import (  # noqa: E402
    EPISODES_PER_SHARD, HEIGHT_LIMIT, HEIGHT_SCALE, SCHEMA_VERSION, ShardWriter, write_manifest,
)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


SHARD_FILES = ("heights.npy", "actions.npy", "volumes.npz", "meta.json")


def shard_is_complete(path: Path) -> bool:
    """A shard counts only when every file is present -- heights are written first."""
    return path.is_dir() and all((path / f).exists() for f in SHARD_FILES)


def generate_split(name: str, root: Path, workers: int, limit: int | None) -> dict:
    spec = SPLITS[name]
    plan = spec.episode_plan()
    if limit is not None:
        plan = plan[:limit]

    per_shard = min(EPISODES_PER_SHARD, len(plan))
    shards: list[dict] = []
    started = time.perf_counter()
    done = 0
    resumed = 0

    with mp.Pool(processes=workers) as pool:
        for shard_id, begin in enumerate(range(0, len(plan), per_shard)):
            chunk = plan[begin:begin + per_shard]
            shard_path = root / "shards" / f"{name}_{shard_id:03d}"

            # Resume: finished shards are kept, a half-written one is discarded and
            # redone. Episodes are deterministic in their seed, so a shard rebuilt now
            # is identical to the one that would have been written before the crash.
            if shard_is_complete(shard_path):
                shards.append({"name": shard_path.name, "n_episodes": len(chunk)})
                done += len(chunk)
                resumed += len(chunk)
                print(f"  {name}: {shard_path.name} already complete, skipping", flush=True)
                continue
            if shard_path.exists():
                for stale in shard_path.iterdir():
                    stale.unlink()

            writer = ShardWriter(
                path=shard_path,
                n_episodes=len(chunk),
                n_steps=N_STEPS,
                grid_shape=DEFAULT_GRID.shape,
                action_dim=RAW_DIM,
            )
            tasks = [(name, seed, family) for seed, family in chunk]
            for result in pool.imap(_worker, tasks, chunksize=1):
                writer.add(result["frames"], result["actions"], result["volumes"], result["meta"])
                done += 1
                if done % 25 == 0 or done == len(plan):
                    rate = (done - resumed) / max(time.perf_counter() - started, 1e-9)
                    remaining = (len(plan) - done) / max(rate, 1e-9)
                    print(f"  {name}: {done}/{len(plan)} episodes "
                          f"({rate * 60:.1f}/min, ~{remaining / 60:.1f} min left)", flush=True)
            shards.append(writer.close())

    elapsed = time.perf_counter() - started
    print(f"  {name}: done in {elapsed / 60:.1f} min", flush=True)
    return {
        "n_episodes": len(plan),
        "families": list(spec.families),
        "theta_deg": spec.theta_deg,
        "policy": spec.policy,
        "seed_base": spec.seed_base,
        "shards": shards,
        "generation_seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "v1")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--limit", type=int, default=None,
                        help="cap episodes per split, for smoke runs")
    parser.add_argument("--log-file", default=None,
                        help="write progress here, opened by this process (see wm.runlog)")
    args = parser.parse_args()
    redirect_output(args.log_file)

    assert_seed_ranges_disjoint()
    root = Path(args.root)
    print(f"writing to {root}  ({args.workers} workers)", flush=True)

    splits = {}
    for name in args.splits:
        if name not in SPLITS:
            raise SystemExit(f"unknown split {name!r}; expected one of {list(SPLITS)}")
        splits[name] = generate_split(name, root, args.workers, args.limit)

    write_manifest(root, {
        "schema_version": SCHEMA_VERSION,
        "sim_git_sha": git_sha(),
        "grid": list(DEFAULT_GRID.shape),
        "cell_size_m": DEFAULT_GRID.dx,
        "n_steps": N_STEPS,
        "swell": SWELL,
        "height_encoding": {
            "dtype": "int16", "scale_m": HEIGHT_SCALE, "limit_m": HEIGHT_LIMIT,
        },
        "sim_params": {
            "relax_alpha": ALPHA, "relax_w_diag": W_DIAG, "relax_tol_frac": TOL_FRAC,
            "capacity_per_width": CAPACITY_PER_WIDTH, "max_retries": MAX_RETRIES,
        },
        "action": {
            "raw_dim": RAW_DIM, "raw_fields": list(FIELDS),
            "model_dim": MODEL_DIM, "model_fields": list(MODEL_FIELDS),
            "bounds": {k: list(v) for k, v in BOUNDS.items()},
        },
        "train_sizes": list(TRAIN_SIZES),
        "splits": splits,
    })
    print(f"manifest written to {root / 'dataset.json'}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
