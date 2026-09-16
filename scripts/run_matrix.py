"""Run the experiment matrix.

    python scripts/run_matrix.py --stage core          # the 13-run minimum
    python scripts/run_matrix.py --stage seeds         # extra seeds on the headline configs
    python scripts/run_matrix.py --dry-run

Thirteen training runs cover every axis the project claims to measure. Everything else --
horizon curves, the volume metric, both rollout protocols, all four distribution shifts
and the failure taxonomy -- is evaluation-time work on those same checkpoints, so the
generalisation and soil-transfer results cost no additional training at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def run_id(**kw) -> str:
    parts = [kw["arch"], f"K{kw['k_train']}"]
    if kw["arch"].startswith("latent"):
        parts.append(f"z{kw['latent_dim']}")
    if not kw.get("use_action", True):
        parts.append("noact")
    if kw.get("loss", "huber") != "huber":
        parts.append(kw["loss"])
    if kw.get("uses_volume_loss"):
        parts.append("volloss")
    parts += [f"n{kw['train_size']}", f"s{kw['seed']}"]
    return "_".join(parts)


def spec(**kw) -> dict:
    base = dict(arch="latentB", latent_dim=64, k_train=5, train_size=2000,
                seed=0, use_action=True, loss="huber", uses_volume_loss=False)
    base.update(kw)
    return {"run_id": run_id(**base), **base}


# The minimum covering set: one seed each, every stated axis represented.
CORE = [
    spec(),                                              # primary world model
    spec(arch="latentA"),                                # protocol-matched partner
    spec(arch="unet", k_train=5),                        # protocol-matched pixel model
    spec(arch="unet", k_train=1),                        # stated baseline: one-step U-Net
    spec(k_train=1),                                     # stated baseline: latent, single step
    spec(use_action=False),                              # no action conditioning
    spec(latent_dim=16),
    spec(latent_dim=256),
    spec(train_size=250),
    spec(train_size=500),
    spec(train_size=1000),
    spec(train_size=4000),
    spec(train_size=6000),
]

# Extra seeds on the five headline configurations.
SEEDS = [
    spec(**{**{k: v for k, v in base.items() if k not in ("run_id",)}, "seed": seed})
    for base in [spec(), spec(arch="latentA"), spec(arch="unet", k_train=5),
                 spec(arch="unet", k_train=1), spec(k_train=1)]
    for seed in (1, 2)
]

# Runs that answer an obvious objection rather than a stated axis.
EXTRA = [
    spec(arch="unet", k_train=5, use_action=False),   # is the action result architecture-specific?
    spec(loss="l1"),                                  # reproduce the identity collapse
    # Quarantined: optimises the conservation residual directly, so its volume number is
    # not evidence about the headline claim. The interesting question it answers is
    # whether forcing volume to balance also fixes the repose violations.
    spec(uses_volume_loss=True),
]

# Short probes to pick a learning rate per architecture. Comparing a proposed model at
# its tuned learning rate against a baseline at someone else's is the standard way to
# make a baseline look worse than it is, so each architecture gets its own sweep and the
# numbers go in the appendix.
#
# Probed at a one-step horizon, deliberately. The five-step runs ramp their horizon over
# the first seven epochs, so a probe short enough to be affordable would never reach K=5
# and would be selecting a rate for an objective no real run uses. One step is a clean,
# common objective that every architecture here shares.
LR_PROBE_EPOCHS = 5
LR_PROBE = [
    dict(spec(arch=arch, k_train=1), lr=lr, run_id=f"lrprobe_{arch}_K1_lr{lr:g}")
    for arch in ("latentB", "unet")
    for lr in (1e-4, 3e-4, 1e-3)
]

# Untrained references. No training cost, and they anchor every physics metric.
BASELINES = ["identity", "mean_terrain"]

STAGES = {"lr": LR_PROBE, "core": CORE, "seeds": SEEDS, "extra": EXTRA}


# Abort after this many failures in a row. One failure is usually a transient -- an
# out-of-memory blip, a file still being written -- and losing the remaining runs to it
# would waste hours. Several in a row means something systematic, and grinding through
# the rest of the matrix producing nothing is worse than stopping.
CONSECUTIVE_FAILURE_LIMIT = 3


def launch(cmd: list[str], label: str, dry_run: bool) -> tuple[float, bool]:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0.0, True
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        print(f"!!! {label} failed with exit code {result.returncode}", flush=True)
        return elapsed, False
    return elapsed, True



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="*", default=["core"], choices=[*STAGES, "baselines"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", default="data/v1")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="learning rate for stages other than the probe")
    args = parser.parse_args()

    python = sys.executable
    timings: dict[str, float] = {}
    failed: list[str] = []
    consecutive = 0

    def record(label: str, ok: bool) -> None:
        nonlocal consecutive
        if ok:
            consecutive = 0
            return
        failed.append(label)
        consecutive += 1
        if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
            raise SystemExit(
                f"stopping: {consecutive} jobs failed in a row, most recently {label}"
            )

    if "baselines" in args.stage:
        for name in BASELINES:
            _, ok = launch([python, "scripts/evaluate.py", "--run-id", name, "--arch", name,
                            "--data-root", args.data_root], f"baseline {name}", args.dry_run)
            record(f"baseline {name}", ok)

    for stage in args.stage:
        if stage == "baselines":
            continue
        for cfg in STAGES[stage]:
            name = cfg["run_id"]
            if args.skip_existing and (ROOT / "runs" / name / "ckpt_best.pt").exists():
                print(f"\n=== {name}: already trained, skipping ===", flush=True)
            else:
                cmd = [python, "scripts/train.py", "--run-id", name,
                       "--arch", cfg["arch"], "--latent-dim", str(cfg["latent_dim"]),
                       "--k-train", str(cfg["k_train"]), "--train-size", str(cfg["train_size"]),
                       "--seed", str(cfg["seed"]), "--loss", cfg["loss"],
                       "--lr", str(cfg.get("lr", args.lr)),
                       "--data-root", args.data_root]
                cmd.append("--use-action" if cfg["use_action"] else "--no-use-action")
                cmd.append("--uses-volume-loss" if cfg.get("uses_volume_loss")
                           else "--no-uses-volume-loss")
                epochs = args.epochs or (LR_PROBE_EPOCHS if stage == "lr" else None)
                if epochs:
                    cmd += ["--epochs", str(epochs)]
                timings[name], ok = launch(cmd, f"train {name}", args.dry_run)
                record(f"train {name}", ok)
                if not ok:
                    continue

            # Probes are compared on dev five-step error alone; there is nothing to
            # learn from evaluating a deliberately under-trained model on held-out splits.
            if stage != "lr":
                _, ok = launch([python, "scripts/evaluate.py", "--run-id", name,
                                "--data-root", args.data_root], f"evaluate {name}", args.dry_run)
                record(f"evaluate {name}", ok)

    if timings and not args.dry_run:
        path = ROOT / "results" / "run_timings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update({k: round(v, 1) for k, v in timings.items()})
        path.write_text(json.dumps(existing, indent=2))

    total = len(BASELINES) if "baselines" in args.stage else 0
    total += sum(len(STAGES[s]) for s in args.stage if s != "baselines")
    print(f"\n{total} configurations in stages {args.stage}")
    if failed:
        print(f"{len(failed)} job(s) failed:")
        for label in failed:
            print(f"  - {label}")
        raise SystemExit(1)
    print("all jobs completed")


if __name__ == "__main__":
    main()
