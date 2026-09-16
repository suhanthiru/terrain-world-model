# Results

Sixteen trained models, two untrained baselines, six evaluation splits, fifty-step
rollouts. Every number below comes from `results/index.csv` and the per-run summaries,
and regenerates with `scripts/make_tables.py` and `scripts/make_figures.py` without
touching a GPU.

Unless stated otherwise: `test_indist` (200 held-out episodes, flat and trench terrain at
30°), median over episodes, error in millimetres, volume error as a percentage of the
material actually excavated.

---

## The headline

**A latent world model beats the do-nothing baseline on pixel error while inventing 60%
extra material from the very first step.**

| k | `latentB` MAE | identity MAE | `latentB` volume | identity volume |
|---:|---:|---:|---:|---:|
| 1 | **3.59 mm** | 4.64 mm | **+59.9%** | −25.0% |
| 5 | **16.3 mm** | 22.8 mm | **+63.6%** | −25.0% |
| 20 | **53.8 mm** | 63.7 mm | **+59.2%** | −25.0% |
| 50 | 193.6 mm | 103.7 mm | **+98.2%** | −25.0% |

Judged on the metric this literature reports, the model works: it is 23% better than
identity at one step and 16% better at twenty. Judged on whether its terrain contains the
right amount of dirt, it is more than twice as wrong as doing nothing, in the opposite
direction.

Two details matter more than the headline number.

**The error does not accumulate.** Volume error is essentially flat from k=1 to k=20
(+59.9%, +63.6%, +59.2%). This is not drift compounding through a rollout — it is a
systematic bias present in the very first prediction. "Errors grow with horizon" would be
the comfortable reading and it is the wrong one.

**Predicting no change is a strong baseline on this metric.** Identity scores exactly
−(swell−1) = −25.0% at every horizon, by construction. Every learned model in this study
is further from zero than that.

---

## What actually goes wrong: the model deposits but does not dig

The volume metric says material appears from nowhere. Decomposing the movement says why.

| | `latentB` @20 | `unet` @20 |
|---|---:|---:|
| material moved **down** vs truth | **−80.8%** | **−0.7%** |
| effective swell (simulator: **1.250**) | **5.305** | **1.379** |
| volume error | +59.2% | +10.0% |

The latent model reproduces the deposited cone and largely fails to reproduce the bucket
cut: it moves only a fifth of the material that should go downward, while depositing
roughly the right amount. Net, it manufactures dirt. Its effective swell factor is 5.3
where the simulator's is 1.25.

The reference is measured, not assumed. Applying the same estimator to the simulator's own
frames returns exactly **1.250** at every horizon — relaxation moves equal volumes up and
down, so it cancels in the ratio — which is what makes the 4.2× gap meaningful.

The blurring is the same failure seen spectrally: the latent model's high-frequency energy
ratio collapses to **0.11** at twenty steps. A sharp, localised, action-dependent cut is
exactly what a 64-number bottleneck cannot carry; a smooth cone is exactly what it can.

---

## It is the bottleneck, not learning

The pixel U-Net has the same parameter count to within 0.32% and sees the same data.

| | MAE@1 | MAE@20 | MAE@50 | volume@20 | repose@20 | beats identity to |
|---|---:|---:|---:|---:|---:|---:|
| `unet` K=5 | **1.54 mm** | **20.5 mm** | **34.6 mm** | **+10.0%** | 12.05% | **>50** |
| `latentB` K=5 | 3.59 mm | 53.8 mm | 193.6 mm | +59.2% | 11.29% | 49 |
| identity | 4.64 mm | 63.7 mm | 103.7 mm | −25.0% | 0% | — |

Routing detail around the bottleneck through skip connections recovers the excavation
almost exactly (−0.7% on downward movement) and brings mass conservation to ~10%. So the
mass hallucination is not a consequence of learning the dynamics; it is a consequence of
compressing the state.

**The comparison is parameter-matched and not compute-matched, and that favours the
U-Net.** It convolves at full 128×128 resolution throughout — 15.0 ms per rollout step
against the latent model's 4.1 — and its action reaches every block through FiLM, against
a single 64-d embedding entering one GRU. Both asymmetries are reported rather than
engineered away. The honest framing is not which architecture wins but what the bottleneck
costs: it buys a rollout that never re-reads a heightmap, and it costs mass conservation.

**Repose violations are essentially identical** (12.05% vs 11.29%). Skip connections fix
the mass failure and do nothing for the physics failure — these are separate problems.

---

## Failure modes

Classified per episode at k=20 and k=50, thresholds calibrated on `dev` against the
baselines only and then frozen.

| run | k=20 | k=50 |
|---|---|---|
| `unet` K=5 | 28.5% flagged — all `spatial_displacement` | 42.0% — all `spatial_displacement` |
| `latentB` K=5 | **100% `volume_explosion`** | **100% `volume_explosion`** |
| `latentB` K=1 | **100% `volume_explosion`** | **100% `volume_explosion`** |
| identity | 100% `identity_collapse` | 100% `identity_collapse` |

Not one of the two hundred held-out episodes escapes the volume failure for the latent
models. The U-Net never triggers it; its failures are spatial — a correctly shaped
excavation in the wrong place, which is what `spatial_displacement` is built to separate
from noise (large centroid shift *and* high shift-corrected correlation).

---

## Ablations

### Action conditioning is what makes any of this work

Feeding a zero action vector — same architecture, same parameter count, same optimiser —
collapses both architectures onto the identity baseline.

| | MAE@1 | MAE@20 | volume@20 |
|---|---:|---:|---:|
| identity | 4.642 mm | 63.66 mm | −25.00% |
| `latentB` no action | 4.652 mm | 63.73 mm | −25.23% |
| `unet` K=5 no action | 4.809 mm | 64.32 mm | −25.36% |

Architecture-independent, and the right answer: the action is the only thing that says
where the bucket goes.

### Multi-step loss is load-bearing

| | MAE@20 | MAE@50 | volume@20 | volume@50 | beats identity to |
|---|---:|---:|---:|---:|---:|
| `latentB` K=5 | 53.8 mm | 193.6 mm | +59.2% | +98.2% | 49 |
| `latentB` K=1 | 158.8 mm | 452.4 mm | **+198.3%** | **+288.2%** | **4** |

Training on single-step prediction produces a model that is trustworthy for four steps and
then diverges: three times the error at twenty steps and volume error rising without
bound. For the U-Net the effect is milder but the same sign (K=1 reaches +85.9% volume
error against K=5's +10.0%).

### Latent dimension is not monotonic

| z | MAE@20 | volume@20 |
|---:|---:|---:|
| 16 | 51.96 mm | +42.9% |
| 64 | 53.78 mm | +59.2% |
| 256 | 71.41 mm | **−86.9%** |

z=256 is worse on both axes and fails in the opposite direction — it destroys material
rather than inventing it. Note the parameter count is not held constant across this sweep
(894k / 1005k / 1448k), so capacity and bottleneck width are confounded; this is reported
rather than corrected, since equalising it would require changing encoder width and swap
one confound for another.

### Training-set size

Nested prefixes, so the sweep varies how much data there is and never which data it is.
Every run gets the same 30,000 optimiser steps, so it varies data diversity and not
training budget.

| episodes | 250 | 500 | 1000 | 2000 | 4000 | 6000 |
|---|---:|---:|---:|---:|---:|---:|
| MAE@1 (mm) | 3.815 | 3.629 | 3.617 | 3.586 | 3.581 | 3.570 |

Monotone and saturating: a 24× increase in data buys 6% at one step. Long-horizon error is
dominated by rollout dynamics rather than data volume and does not order cleanly.

### Plain L1 does not train this problem — reproduced at full scale

| | MAE@1 | MAE@20 | volume@20 |
|---|---:|---:|---:|
| identity | 4.642 mm | 63.657 mm | −25.00% |
| `latentB` trained with L1 | 4.642 mm | 63.658 mm | −25.00% |

Indistinguishable from predicting no change, to three decimal places. About 95% of cells
hold an exact zero delta and L1's gradient has constant magnitude however wrong a cell is,
so the static majority outvotes the rest and doing nothing is a fixed point the optimiser
converges to. Every other run here uses Huber, which is quadratic near zero and therefore
lets a nearly-correct static cell contribute a nearly-zero gradient. Reported error is
still absolute error throughout — this changes how the models train, not how they are
scored.

---

## Distribution shift

MAE@20, trained on flat and trench terrain at 30°.

| run | in-dist | terrain | soil (40°) | both | held-out policy |
|---|---:|---:|---:|---:|---:|
| `unet` K=5 | 20.5 | 24.6 (+20%) | **30.0 (+46%)** | 34.5 (+68%) | 22.8 (+11%) |
| `latentB` K=5 | 53.8 | 56.0 (+4%) | 59.9 (+11%) | 62.2 (+16%) | 53.6 (−0.3%) |
| identity | 63.7 | 60.9 | 68.0 | 65.6 | 61.3 |

**Soil transfer costs roughly twice what terrain transfer costs**, and the two are close to
additive (+20% and +46% separately, +68% together), so there is little interaction — which
is what the 2×2 factorial was for. The latent model degrades far less in relative terms
only because it is already far worse; in absolute terms the U-Net is better on every split.

---

## The quarantined arm

One run optimises the conservation residual directly. Its volume number is not evidence
about whether a generically trained model conserves mass — it optimised the thing being
measured — and `make_tables.py` refuses to put it in a table with the others. It is run to
answer a different question: does forcing volume to balance also fix the physics?

| | MAE@20 | volume@20 | repose@20 | effective swell |
|---|---:|---:|---:|---:|
| `latentB` | 53.8 mm | +59.2% | 11.29% | 5.305 |
| `latentB` + volume loss | 61.6 mm | **+7.8%** | **6.71%** | 2.789 |

I expected this to fix volume and leave the repose violations untouched. It partially fixed
both — repose violations nearly halved — at a cost in accuracy. But effective swell only
moves from 5.3 to 2.8 against a true 1.25, so the model is balancing its books by some
route other than learning to excavate.

---

## What would weaken these conclusions

- **One seed per configuration.** Optimisation variance is unmeasured. The large effects
  here (a 6× gap in effective swell, a 100% failure rate) are far outside any plausible
  seed noise, but the small ones — `latentB` at 11.29 mm versus `latentA` at 11.40 — are
  not, and no ordering should be read into them.
- **The simulator is mine.** The strongest available criticism. Mitigated with evidence
  rather than argument: the simulator's own mass-balance residual is at machine precision
  (≤7e-15 m³ over a 50-cycle episode, plotted in `results/figures/sim_conservation.png`),
  and its frames have exactly zero repose violations. Storage quantisation contributes
  0.009% of excavated material to the volume metric, against effects of tens of percent.
  This is a synthetic-dynamics study and makes no claim about real soil.
- **Training to five steps, evaluating to fifty.** A 10× extrapolation, marked on every
  figure. The identity and −25% reference lines keep k>5 interpretable regardless.
- **The learning-rate sweep is incomplete.** Two of six probes were lost to GPU contention
  from unrelated processes. 1e-4 won the latent sweep outright and the one U-Net probe that
  finished; the missing points are 3e-4 and 1e-3 for the U-Net.
- **Early stopping fires while masked error is still improving.** The criterion — dev
  5-step all-cell L1 — was fixed before any training and has not been changed. But all-cell
  error saturates because the model is already correct on the ~95% of cells that never
  move, so runs stop with headroom left on the metric that actually separates models.
