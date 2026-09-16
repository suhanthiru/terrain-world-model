"""On-disk format for generated episodes.

Frames are stored as int16 fixed-point at 0.2 mm per count. That is the same two bytes
as float16 and deliberately not float16, because float16's error is not random and so
does not cancel. Excavation terrain is full of flat regions -- the initial flat family,
the cut floor, a trench bottom -- and every cell in a flat region sitting at an
unrepresentable height takes the *same* rounding error. Summed over a couple of thousand
touched cells those errors add coherently to roughly five litres, which is about five
percent of a single swing and the same order as the volume error we intend to report.
Fixed-point spreads its error uniformly and kills that mode.

The exact per-step volume bookkeeping is written separately in float64, so the headline
metric's ground truth never passes through the quantiser at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# Metres per int16 count. Range is +/- 6.5534 m; measured 50-cycle episodes stay inside
# about +/- 1.3 m, so there is roughly five times headroom.
HEIGHT_SCALE = 2e-4
HEIGHT_LIMIT = 6.0  # refuse to encode anything beyond this, rather than clip silently

EPISODES_PER_SHARD = 250

VOLUME_FIELDS = ("removed", "v_geom", "deposited", "total", "residual")


def encode_heights(h: np.ndarray) -> np.ndarray:
    """Metres -> int16 counts. Raises rather than clipping out-of-range material."""
    peak = float(np.abs(h).max())
    if peak > HEIGHT_LIMIT:
        raise ValueError(
            f"height {peak:.3f} m exceeds the {HEIGHT_LIMIT} m encoding limit; "
            "the episode has run away and should not be stored"
        )
    return np.rint(h / HEIGHT_SCALE).astype(np.int16)


def decode_heights(q: np.ndarray) -> np.ndarray:
    """int16 counts -> metres."""
    return q.astype(np.float32) * np.float32(HEIGHT_SCALE)


@dataclass
class ShardWriter:
    """Writes one shard: frames, actions, the float64 volume sidecar, and metadata.

    Frames are laid out (episode, time, y, x) so that every frame of an episode is
    contiguous. A K-step training window is then a single ~190 KiB read. Putting time on
    the outer axis instead would turn each window into six scattered reads.
    """

    path: Path
    n_episodes: int
    n_steps: int
    grid_shape: tuple[int, int]
    action_dim: int

    def __post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._heights = np.lib.format.open_memmap(
            self.path / "heights.npy", mode="w+", dtype=np.int16,
            shape=(self.n_episodes, self.n_steps + 1, *self.grid_shape),
        )
        self._actions = np.zeros((self.n_episodes, self.n_steps, self.action_dim), np.float32)
        self._volumes = {f: np.zeros((self.n_episodes, self.n_steps), np.float64)
                         for f in VOLUME_FIELDS if f != "total"}
        self._volumes["total"] = np.zeros((self.n_episodes, self.n_steps + 1), np.float64)
        self._meta: list[dict] = []
        self._written = 0

    def add(self, frames: np.ndarray, actions: np.ndarray, volumes: dict, meta: dict) -> None:
        i = self._written
        self._heights[i] = encode_heights(frames)
        self._actions[i] = actions
        for field, values in volumes.items():
            self._volumes[field][i] = values
        meta = dict(meta, index_in_shard=i, shard=self.path.name)
        self._meta.append(meta)
        self._written += 1

    def close(self) -> dict:
        if self._written != self.n_episodes:
            raise RuntimeError(f"shard {self.path.name}: expected {self.n_episodes}, got {self._written}")
        self._heights.flush()
        np.save(self.path / "actions.npy", self._actions)
        np.savez(self.path / "volumes.npz", **self._volumes)
        (self.path / "meta.json").write_text(json.dumps(self._meta, indent=2))
        return {"name": self.path.name, "n_episodes": self.n_episodes}


class ShardReader:
    """Random access to a shard, keeping the frames on disk until asked for."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.heights = np.load(self.path / "heights.npy", mmap_mode="r")
        self.actions = np.load(self.path / "actions.npy")
        with np.load(self.path / "volumes.npz") as z:
            self.volumes = {k: z[k] for k in z.files}
        self.meta = json.loads((self.path / "meta.json").read_text())

    def __len__(self) -> int:
        return self.heights.shape[0]

    @property
    def n_steps(self) -> int:
        return self.heights.shape[1] - 1

    def window(self, episode: int, start: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Frames [start, start+k] and the k actions that connect them."""
        frames = decode_heights(np.asarray(self.heights[episode, start:start + k + 1]))
        return frames, self.actions[episode, start:start + k]


def build_window_index(n_episodes: int, n_steps: int, k: int) -> np.ndarray:
    """Every (episode, start) pair admitting a k-step window, as int32."""
    starts = np.arange(n_steps - k + 1, dtype=np.int32)
    episodes = np.arange(n_episodes, dtype=np.int32)
    grid = np.stack(np.meshgrid(episodes, starts, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def write_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(root: Path) -> dict:
    return json.loads((Path(root) / "dataset.json").read_text())
