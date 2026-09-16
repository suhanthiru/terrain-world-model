"""Turning stored episodes into model inputs.

Two scales, both frozen constants rather than statistics re-estimated per split. A
data-dependent scaler would drift between the training set and the transfer sets --
slope and pile terrain genuinely has a different height variance from flat and trench --
and that difference is signal we want the model to see, not something to normalise away.
"""

from __future__ import annotations

import numpy as np

# Heights, after the per-window offset below, are order half a metre.
#
# Measured on the generated splits, the offset-removed height RMS is 0.16 m in the
# training distribution and 0.48 m on the hardest transfer split (slope and pile terrain
# at 40 degrees), with a peak of 2.1 m. Dividing by 0.5 puts encoder inputs at an RMS of
# 0.31 in-distribution and 0.96 out of it.
#
# That is deliberately not tuned to the training set. Estimating this from dev alone
# would give roughly 0.2, which would push transfer inputs to an RMS of 2.4 and a peak
# near 10 -- far outside anything the encoder saw in training, and it would confound "the
# model does not generalise" with "the input arrived at the wrong scale". The transfer
# splits genuinely have more height variance, and that is signal the model should see.
HEIGHT_SCALE = 0.5

# Per-step change is order centimetres. Measured across every split, the all-cell RMS
# delta is 2.7 to 3.2 cm, so dividing by 2 cm leaves the decoder predicting something of
# order one rather than order 0.01.
DELTA_SCALE = 0.02


def window_offset(first_frame: np.ndarray) -> np.ndarray:
    """The height datum for a window: the spatial mean of its *first* frame.

    One offset for the whole window, taken from frame zero. A per-frame mean would
    change the frame-to-frame differences and so destroy the very target the model is
    trained on.

    This is legitimate because the simulator has no bedrock and no boundary pinned to an
    absolute elevation, so adding a constant to the world just adds it to the answer
    (``tests/test_sim.py`` asserts this to 1e-11). It amounts to free translation
    augmentation. It also means the model cannot learn any absolute-height-dependent
    behaviour -- which is correct here only because there is none to learn.
    """
    axes = tuple(range(first_frame.ndim))[-2:]
    return first_frame.mean(axis=axes)


def to_model_frames(frames: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Stored metres -> offset-removed metres, the space the rollout runs in."""
    return frames - offset[..., None, None]


def encode_input(x: np.ndarray) -> np.ndarray:
    """Offset-removed metres -> encoder input."""
    return x / HEIGHT_SCALE


def decode_delta(delta_norm: np.ndarray) -> np.ndarray:
    """Decoder output -> metres."""
    return delta_norm * DELTA_SCALE
