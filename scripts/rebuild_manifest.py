"""Rebuild a dataset manifest from whatever shards are complete on disk.

    python scripts/rebuild_manifest.py --root data/v1

Useful in two situations. If generation dies partway, this recovers a manifest for the
episodes that did finish instead of throwing them away. And while generation is still
running, it makes the completed prefix usable immediately -- episodes are deterministic
in their seed and shards are written in order, so a manifest over the first N shards
describes exactly the same data the full run will.

A shard counts only when all four of its files are present; heights are written first, so
a directory holding only heights.npy is one that was interrupted mid-write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from terrain.action import BOUNDS, FIELDS, MODEL_DIM, MODEL_FIELDS, RAW_DIM  # noqa: E402
from terrain.grid import DEFAULT_GRID  # noqa: E402
from terrain.relax import ALPHA, TOL_FRAC, W_DIAG  # noqa: E402
from terrain.sim import CAPACITY_PER_WIDTH  # noqa: E402
from wm.data.generate import MAX_RETRIES, N_STEPS, SPLITS, SWELL, TRAIN_SIZES  # noqa: E402
from wm.data.storage import (  # noqa: E402
    HEIGHT_LIMIT, HEIGHT_SCALE, SCHEMA_VERSION, write_manifest,
)

SHARD_FILES = ("heights.npy", "actions.npy", "volumes.npz", "meta.json")


def complete_shards(root: Path, split: str) -> list[dict]:
    """Shards of this split that finished writing, in order, stopping at the first gap."""
    found = []
    for index in range(1000):
        path = root / "shards" / f"{split}_{index:03d}"
        if not all((path / f).exists() for f in SHARD_FILES):
            break
        count = int(np.load(path / "heights.npy", mmap_mode="r").shape[0])
        found.append({"name": path.name, "n_episodes": count})
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "v1")
    parser.add_argument("--out", type=Path, default=None,
                        help="write here instead of <root>/dataset.json")
    args = parser.parse_args()

    root = Path(args.root)
    splits = {}
    for name, spec in SPLITS.items():
        shards = complete_shards(root, name)
        if not shards:
            continue
        total = sum(s["n_episodes"] for s in shards)
        splits[name] = {
            "n_episodes": total,
            "families": list(spec.families),
            "theta_deg": spec.theta_deg,
            "policy": spec.policy,
            "seed_base": spec.seed_base,
            "shards": shards,
            "generation_seconds": None,
            "partial": total < spec.n_episodes,
        }
        flag = f"  (partial: {spec.n_episodes} planned)" if splits[name]["partial"] else ""
        print(f"  {name:18s} {total:5d} episodes across {len(shards)} shard(s){flag}")

    if not splits:
        raise SystemExit(f"no complete shards under {root / 'shards'}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rebuilt_from_disk": True,
        "grid": list(DEFAULT_GRID.shape),
        "cell_size_m": DEFAULT_GRID.dx,
        "n_steps": N_STEPS,
        "swell": SWELL,
        "height_encoding": {"dtype": "int16", "scale_m": HEIGHT_SCALE, "limit_m": HEIGHT_LIMIT},
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
    }
    target = args.out or root / "dataset.json"
    write_manifest(target.parent, manifest) if target.name == "dataset.json" else \
        target.write_text(json.dumps(manifest, indent=2))
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
