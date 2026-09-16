# Terrain world model

Learned world models for excavation, and whether they conserve mass.

Almost every heightmap prediction paper reports MSE. Very few report whether the model's
terrain still contains the right amount of dirt. A model can look sharp and plausible
while quietly inventing 12% extra soil over a 20-step rollout — useless for planning, and
pixel error will never tell you.

So the point of this repo is the evaluation harness, not the model. There is a small
NumPy excavation simulator with exact mass bookkeeping to generate ground truth, a few
deliberately small learned models, and a set of metrics built around volume conservation.

## The simulator

128x128 heightmap at 5 cm/cell — a 6.4 m patch. One action is one dig-swing-dump cycle:

1. **Cut** — subtract a swept bucket volume. The bucket follows a rigid trajectory
   relative to the entry height, so digging uphill removes more than digging downhill.
   Bucket capacity truncates the sweep where the bucket fills.
2. **Slump** — iterative angle-of-repose relaxation until no adjacent cell pair exceeds
   tan(theta).
3. **Deposit** — place `removed * swell` at the dump point as `max(h, z - tan(theta)*r)`,
   with `z` solved by water-filling so exactly that volume lands.
4. **Slump** again.

Mass bookkeeping is exact in float64. The invariant is not `sum(h) = const` — swell
deliberately creates height — but

```
V_after - V_before == (swell - 1) * removed
```

which holds to ~1e-15 m^3 over a 50-cycle episode. That is what makes a model's
conservation violation attributable to the model rather than to the simulator.

Terrain families vary the angle of repose and the initial surface -- flat, trench, slope,
existing pile -- and those become the train/test splits: models train on flat and trench
at 30 degrees and are tested on slope and pile, at 40 degrees, on both at once, and on a
held-out action policy.

Swell is held fixed at 1.25 across the dataset even though the simulator supports varying
it. A constant swell is what puts the identity baseline at exactly `-(swell - 1) = -25%`
on the volume metric at every horizon, which is a free and exact reference line on the
headline figure; varying it would smear that line without serving any axis the ablations
actually test.

## Layout

```
src/terrain/   simulator: relaxation, bucket cut, deposit, terrain families
src/wm/        data pipeline, models, training, evaluation
scripts/       generate_dataset, train, evaluate, run_matrix, make_tables, make_figures
tests/         conservation, physics, storage, metric and training invariants
```

```bash
python scripts/generate_dataset.py --root data/v1     # ~12 GB, resumable
python scripts/run_matrix.py --stage baselines core   # 13 training runs + 2 baselines
python scripts/make_tables.py && python scripts/make_figures.py
```

## The models

Deliberately small and matched at about a million parameters each:

- **latent** -- conv encoder to a 64-vector, action MLP, GRU transition, deconv decoder
  predicting a *delta* heightmap. Two rollout protocols: `B` encodes once and rolls the
  latent forward without ever looking at a heightmap again, which is the actual
  world-model claim; `A` re-encodes its own prediction each step.
- **U-Net** -- same job with skip connections instead of a bottleneck, action conditioning
  by FiLM. Having no latent to roll, it is structurally confined to protocol `A`, which
  is why a protocol-`A` latent model is trained alongside it. Comparing architectures
  across different protocols would license no conclusion.
- **identity** and **mean terrain** -- untrained references.

The comparison is matched on parameters and explicitly *not* on compute: the U-Net runs
at full resolution throughout and measures 15.0 ms per rollout step against 4.1 for the
latent model. That asymmetry favours the U-Net and is reported rather than hidden.

## What is measured

Error at 1 / 5 / 20 / 50 steps as a curve, not a scalar, with the training horizon marked
-- models train to five steps and are evaluated to fifty.

The headline is volume: `eps_net(k)`, the predicted change in total material minus the
true change, over the material actually excavated. It reads as *for every cubic metre the
machine moved, the model created this much out of nothing*. Predicting no change scores
exactly `-(swell - 1) = -25%` at every horizon, which is a free and exact reference line.

Alongside it: angle-of-repose violations, a high-frequency energy ratio that catches
blurring and checkerboarding in one number, and an automatic failure taxonomy.

**No physics metric is ever reported alone.** Predicting nothing has zero repose
violations and beats most models on volume, so a physics number without an accuracy
number beside it rewards doing nothing. The default rendering is a two-dimensional
scatter of accuracy against physics violation.

## Some things that turned out to matter

- **8-neighbour relaxation, not 4.** With 4 neighbours the equilibrium pile is a
  45-degree-rotated square pyramid, and the effective diagonal angle of repose is 36.5 deg
  when you asked for 30. That is a 6.5 deg error in the one parameter the terrain families
  are supposed to vary.
- **Deposit by `max`, not by adding a cone.** Slopes add: dropping a repose-slope cone onto
  repose-slope ground gives you `2*tan(theta)`, which the relaxation then has to spend
  hundreds of sweeps demolishing. `max` of two surfaces at repose is already at repose, so
  the relaxation after a deposit is a no-op.
- **Slicing, not `np.roll`.** `roll` wraps, which gives you periodic boundaries. Those are
  perfectly mass-conserving, so every conservation test passes while material teleports
  across the patch.
- **Plain L1 does not train this problem.** Only about 5% of cells move in a step, so 95%
  of targets are an exact zero. L1's gradient has the same magnitude however wrong a cell
  is, so the static majority outvotes the few that moved and predicting no change is a
  genuine optimisation attractor -- not merely a way to score well. Measured on a fixed
  batch, the predicted delta was driven to 4e-6 m against a target of 5.2e-3 m and sat
  exactly on the identity baseline at every learning rate tried. Huber fixes it by making
  a nearly-correct static cell contribute a nearly-zero gradient. Reported error is still
  absolute error; this changes how the models train, not how they are scored.
- **A zero-initialised output layer does not train either.** It is an appealing trick --
  the model starts bit-identical to the identity baseline -- but the gradient reaching
  every upstream layer is proportional to that layer's own weights, so at zero the
  encoder, GRU and action pathway get exactly nothing. The same bug in the U-Net's FiLM
  conditioning left its entire action encoder dead. Both now start near zero instead.
- **Window the FFT.** The domain has walls and does not wrap, so an unwindowed spectrum
  is dominated by the step discontinuity where the array edges meet. That leakage made
  the blur detector useless -- a heavy Gaussian blur left the apparent high-frequency
  energy unchanged. With a Hann window it reads 0.03.

## Storage

Frames are int16 fixed-point at 0.2 mm, with the exact per-step volume bookkeeping stored
separately in float64. Not float16: its error scales with magnitude and, worse, is
identical for every cell at the same unrepresentable height. Terrain is full of flat
regions, so those errors add coherently rather than cancelling -- about five litres over a
touched region, which is roughly 5% of a single swing and the same order as the effect
being reported. Measured, the fixed-point format contributes **0.009%** of the material
excavated to the volume metric.

## Status

Simulator, dataset, models, training and evaluation are in place and tested. Experiment
matrix running.
