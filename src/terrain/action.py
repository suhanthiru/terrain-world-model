"""The action space: one dig-swing-dump cycle."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Declared bounds for every raw action component. These are part of the dataset
# contract and are written into the manifest. Normalisation uses *these*, never
# empirical statistics of the training set -- otherwise a held-out policy with a
# different action distribution would be silently rescaled at test time.
BOUNDS: dict[str, tuple[float, float]] = {
    "entry_y": (0.30, 6.10),
    "entry_x": (0.30, 6.10),
    "heading": (-math.pi, math.pi),
    "sweep_len": (0.60, 2.40),
    "depth_1": (0.00, 0.35),
    "depth_2": (0.00, 0.35),
    "depth_3": (0.00, 0.35),
    "width": (0.30, 0.90),
    "dump_y": (0.20, 6.20),
    "dump_x": (0.20, 6.20),
}

FIELDS = tuple(BOUNDS)
RAW_DIM = len(FIELDS)

# What the network sees. Heading becomes (cos, sin): a normalised angle has a
# discontinuity at the +/-pi wrap, and the model would otherwise have to learn that two
# numerically distant inputs describe the same physical swing.
MODEL_DIM = RAW_DIM + 1
MODEL_FIELDS = (
    "entry_y", "entry_x", "heading_cos", "heading_sin", "sweep_len",
    "depth_1", "depth_2", "depth_3", "width", "dump_y", "dump_x",
)

_LO = np.array([BOUNDS[f][0] for f in FIELDS], dtype=np.float64)
_HI = np.array([BOUNDS[f][1] for f in FIELDS], dtype=np.float64)


@dataclass(frozen=True)
class Action:
    entry_y: float
    entry_x: float
    heading: float
    sweep_len: float
    depth_1: float
    depth_2: float
    depth_3: float
    width: float
    dump_y: float
    dump_x: float

    @property
    def depths(self) -> tuple[float, float, float]:
        """Cut depth at s/L = 1/3, 2/3 and 1. Depth at the entry is pinned to zero."""
        return (self.depth_1, self.depth_2, self.depth_3)

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in FIELDS], dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Action":
        return cls(**{f: float(x) for f, x in zip(FIELDS, v)})

    def to_model_vector(self) -> np.ndarray:
        """Normalised to [-1, 1], with heading split into its cosine and sine."""
        return encode(self.to_vector())


def encode(raw: np.ndarray) -> np.ndarray:
    """Raw action(s) -> normalised model vector(s). Accepts (..., RAW_DIM)."""
    raw = np.asarray(raw, dtype=np.float64)
    scaled = 2.0 * (raw - _LO) / (_HI - _LO) - 1.0
    heading = raw[..., FIELDS.index("heading")]
    return np.concatenate(
        [
            scaled[..., :2],
            np.cos(heading)[..., None],
            np.sin(heading)[..., None],
            scaled[..., 3:],
        ],
        axis=-1,
    )


def sample(rng: np.random.Generator, *, policy: str = "heuristic") -> Action:
    """Draw an action.

    ``heuristic`` keeps the depth profile shallow-in, deep-middle, shallow-out, which is
    how a bucket is actually curled through a pass. ``uniform`` ignores that and samples
    the whole box, which is what the held-out policy split uses.
    """
    lo, hi = BOUNDS["depth_2"]
    if policy == "uniform":
        depths = rng.uniform(lo, hi, size=3)
    elif policy == "heuristic":
        peak = rng.uniform(0.35 * hi, hi)
        depths = np.array([
            peak * rng.uniform(0.45, 0.85),
            peak,
            peak * rng.uniform(0.10, 0.60),
        ])
    else:
        raise ValueError(f"unknown policy {policy!r}")

    return Action(
        entry_y=rng.uniform(*BOUNDS["entry_y"]),
        entry_x=rng.uniform(*BOUNDS["entry_x"]),
        heading=rng.uniform(*BOUNDS["heading"]),
        sweep_len=rng.uniform(*BOUNDS["sweep_len"]),
        depth_1=float(depths[0]),
        depth_2=float(depths[1]),
        depth_3=float(depths[2]),
        width=rng.uniform(*BOUNDS["width"]),
        dump_y=rng.uniform(*BOUNDS["dump_y"]),
        dump_x=rng.uniform(*BOUNDS["dump_x"]),
    )
