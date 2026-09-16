"""Domain geometry shared by every part of the simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CELLS = 128
CELL_SIZE = 0.05  # metres


@dataclass(frozen=True)
class Grid:
    """A square heightmap domain.

    Heights are in metres on an arbitrary datum -- the simulator never clamps them,
    so they may go negative. Imposing a floor would break mass conservation.
    """

    n: int = N_CELLS
    dx: float = CELL_SIZE

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def cell_area(self) -> float:
        return self.dx * self.dx

    @property
    def extent(self) -> float:
        """Domain width in metres."""
        return self.n * self.dx

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(y, x) coordinates of cell centres, in metres."""
        c = (np.arange(self.n, dtype=np.float64) + 0.5) * self.dx
        return np.meshgrid(c, c, indexing="ij")

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float64)

    def volume(self, h: np.ndarray) -> float:
        """Total volume under the surface, in cubic metres.

        Summed in float64 because the whole project rests on this number.
        """
        return float(np.sum(h, dtype=np.float64) * self.cell_area)

    def bilinear(self, h: np.ndarray, y: float, x: float) -> float:
        """Sample the surface at a continuous point, in metres.

        Used for the bucket's entry-height reference. Nearest-cell lookup would make
        the action's effect discontinuous: a one-cell move in the entry point could
        shift the reference by a full critical drop.
        """
        fy = np.clip(y / self.dx - 0.5, 0.0, self.n - 1)
        fx = np.clip(x / self.dx - 0.5, 0.0, self.n - 1)
        i0 = int(np.floor(fy))
        j0 = int(np.floor(fx))
        i1 = min(i0 + 1, self.n - 1)
        j1 = min(j0 + 1, self.n - 1)
        ty = fy - i0
        tx = fx - j0
        top = h[i0, j0] * (1.0 - tx) + h[i0, j1] * tx
        bot = h[i1, j0] * (1.0 - tx) + h[i1, j1] * tx
        return float(top * (1.0 - ty) + bot * ty)


DEFAULT_GRID = Grid()
