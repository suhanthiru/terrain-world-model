"""Loading episodes and serving K-step training windows.

No DataLoader workers. The arithmetic says they would not help: one window is a single
contiguous ~190 KiB read plus an int16 cast, and at batch 32 and ~8 steps/second the
pipeline needs a few hundred windows a second, which one Python process supplies
comfortably. Meanwhile Windows spawns processes by re-importing torch, costing seconds
per worker, and memmap handles do not survive that pickling cleanly. So the frames stay
in this process and the dequantise-and-normalise step happens on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from terrain.action import encode as encode_action
from terrain.grid import DEFAULT_GRID

from .normalize import HEIGHT_SCALE, encode_input, window_offset
from .storage import HEIGHT_SCALE as QUANT_SCALE
from .storage import ShardReader, read_manifest

# Above this, frames stay on disk and the OS page cache does the work.
EAGER_LOAD_LIMIT_GB = 6.0


@dataclass
class Batch:
    """One K-step training window, already on the target device.

    ``frames`` are offset-removed metres, so the model can work directly in this space
    and the frame-to-frame deltas are untouched by the normalisation.
    """

    frames: torch.Tensor   # (B, K+1, H, W) float32, offset-removed metres
    actions: torch.Tensor  # (B, K, 11) float32, normalised
    offset: torch.Tensor   # (B,) float32, the metres that were subtracted

    @property
    def k(self) -> int:
        return self.actions.shape[1]

    def encoder_input(self, step: int) -> torch.Tensor:
        return encode_input(self.frames[:, step]).unsqueeze(1)


class EpisodeStore:
    """Every episode of one split, with its frames, encoded actions and volume sidecar."""

    def __init__(self, root: Path, split: str, limit: int | None = None) -> None:
        self.root = Path(root)
        self.split = split
        manifest = read_manifest(self.root)
        if split not in manifest["splits"]:
            raise KeyError(f"split {split!r} not in {self.root/'dataset.json'}")

        shard_names = [s["name"] for s in manifest["splits"][split]["shards"]]
        readers = [ShardReader(self.root / "shards" / name) for name in shard_names]

        heights = [r.heights for r in readers]
        counts = [len(r) for r in readers]
        total = sum(counts)
        # Refuse to quietly hand back fewer episodes than asked for. Silently truncating
        # would let a run labelled n=6000 train on whatever happened to be generated,
        # which would corrupt the data-size sweep in a way nothing downstream could see.
        if limit is not None and limit > total:
            raise ValueError(
                f"split {split!r} holds {total} episodes but {limit} were requested; "
                "the dataset is incomplete for this run"
            )
        self.n_episodes = total if limit is None else limit
        self.n_steps = readers[0].n_steps

        nbytes = self.n_episodes * (self.n_steps + 1) * int(np.prod(DEFAULT_GRID.shape)) * 2
        self.eager = nbytes / 1024**3 <= EAGER_LOAD_LIMIT_GB

        # Episodes are taken in shard order, so a smaller `limit` is always a strict
        # prefix of a larger one. The data-size sweep then varies how much data there
        # is and never which data it is.
        kept, gathered = 0, []
        for block, count in zip(heights, counts):
            if kept >= self.n_episodes:
                break
            take = min(count, self.n_episodes - kept)
            gathered.append(np.asarray(block[:take]) if self.eager else block[:take])
            kept += take
        self._heights = np.concatenate(gathered, axis=0) if self.eager else gathered
        self._offsets = np.cumsum([0] + [len(g) for g in gathered])

        raw = np.concatenate([r.actions for r in readers], axis=0)[: self.n_episodes]
        self.actions = torch.from_numpy(encode_action(raw).astype(np.float32))

        self.volumes = {
            field: np.concatenate([r.volumes[field] for r in readers], axis=0)[: self.n_episodes]
            for field in readers[0].volumes
        }
        self.meta = [m for r in readers for m in r.meta][: self.n_episodes]

    def frames(self, episode: int, start: int, k: int) -> np.ndarray:
        """Raw int16 frames [start, start+k], without decoding."""
        if self.eager:
            return self._heights[episode, start:start + k + 1]
        block = int(np.searchsorted(self._offsets, episode, side="right")) - 1
        local = episode - self._offsets[block]
        return np.asarray(self._heights[block][local, start:start + k + 1])

    def episode_frames(self, episode: int) -> np.ndarray:
        """A whole episode decoded to metres, for evaluation rollouts."""
        return self.frames(episode, 0, self.n_steps).astype(np.float32) * np.float32(QUANT_SCALE)

    def __len__(self) -> int:
        return self.n_episodes


class WindowSampler:
    """Uniform random K-step windows, assembled on the CPU and normalised on the GPU."""

    def __init__(
        self,
        store: EpisodeStore,
        k: int,
        batch_size: int,
        device: torch.device | str = "cuda",
        seed: int = 0,
    ) -> None:
        self.store = store
        self.k = k
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        # Windows always span the widest horizon any run uses, so shortening K truncates
        # the loss rather than shifting which start states appear. Otherwise a K=1 stage
        # would see starts the K=5 stage never does, and the curriculum would quietly
        # change the data distribution as well as the objective.
        self.max_start = store.n_steps - k

    @property
    def windows_per_epoch(self) -> int:
        return len(self.store) * (self.max_start + 1)

    def steps_per_epoch(self) -> int:
        return self.windows_per_epoch // self.batch_size

    def sample(self) -> Batch:
        episodes = self.rng.integers(0, len(self.store), size=self.batch_size)
        starts = self.rng.integers(0, self.max_start + 1, size=self.batch_size)
        return self._assemble(episodes, starts)

    def _assemble(self, episodes: np.ndarray, starts: np.ndarray) -> Batch:
        quantised = np.stack([
            self.store.frames(int(e), int(s), self.k) for e, s in zip(episodes, starts)
        ])
        frames = torch.from_numpy(quantised).to(self.device, non_blocking=True)
        frames = frames.to(torch.float32) * QUANT_SCALE

        offset = frames[:, 0].mean(dim=(1, 2))
        frames = frames - offset[:, None, None, None]

        actions = torch.stack([
            self.store.actions[int(e), int(s):int(s) + self.k] for e, s in zip(episodes, starts)
        ]).to(self.device, non_blocking=True)
        return Batch(frames=frames, actions=actions, offset=offset)

    def fixed_batches(self, n_batches: int, seed: int = 1234):
        """A deterministic set of batches, for validation that does not move between epochs."""
        rng = np.random.default_rng(seed)
        for _ in range(n_batches):
            episodes = rng.integers(0, len(self.store), size=self.batch_size)
            starts = rng.integers(0, self.max_start + 1, size=self.batch_size)
            yield self._assemble(episodes, starts)


def cumulative_removed(store: EpisodeStore) -> np.ndarray:
    """Material actually excavated up to each step: the denominator of the volume metric."""
    return np.cumsum(store.volumes["removed"], axis=1)
