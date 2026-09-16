# Terrain world model

Learned world models for excavation, and whether they conserve mass.

Almost every heightmap prediction paper reports MSE. Very few report whether the model's
terrain still contains the right amount of dirt. A model can look sharp and plausible
while quietly inventing 12% extra soil over a 20-step rollout — useless for planning, and
pixel error will never tell you.

So the point of this repo is the evaluation harness, not the model. There is a small
NumPy excavation simulator with exact mass bookkeeping to generate ground truth, a few
deliberately small learned models, and a set of metrics built around volume conservation.

## What's here

```
src/terrain/    the simulator: relaxation, bucket cut, deposit, terrain families
src/wm/         data pipeline, models, evaluation
scripts/        dataset generation, training, evaluation, figures
tests/          conservation and physics invariants
```

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

Terrain families vary the angle of repose (25-45 deg), the swell factor, and the initial
surface (flat, trench, slope, existing pile). These become the train/test splits.

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

## Status

Under construction. See the build order in the commit history.
