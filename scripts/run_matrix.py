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
    parts += [f"n{kw['train_size']}", f"s{kw['seed']}"]
    return "_".join(parts)


def spec(**kw) -> dict:
    base = dict(arch="latentB", latent_dim=64, k_train=5, train_size=2000,
                seed=0, use_action=True, loss="huber")
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
]

# Untrained references. No training cost, and they anchor every physics metric.
BASELINES = ["identity", "mean_terrain"]

STAGES = {"core": CORE, "seeds": SEEDS, "extra": EXTRA}


def launch(cmd: list[str], label: str, dry_run: bool) -> float:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0.0
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="*", default=["core"], choices=[*STAGES, "baselines"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", default="data/v1")
    args = parser.parse_args()

    python = sys.executable
    timings: dict[str, float] = {}

    if "baselines" in args.stage:
        for name in BASELINES:
            launch([python, "scripts/evaluate.py", "--run-id", name, "--arch", name,
                    "--data-root", args.data_root], f"baseline {name}", args.dry_run)

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
                       "--data-root", args.data_root]
                cmd.append("--use-action" if cfg["use_action"] else "--no-use-action")
                if args.epochs:
                    cmd += ["--epochs", str(args.epochs)]
                timings[name] = launch(cmd, f"train {name}", args.dry_run)

            launch([python, "scripts/evaluate.py", "--run-id", name,
                    "--data-root", args.data_root], f"evaluate {name}", args.dry_run)

    if timings and not args.dry_run:
        path = ROOT / "results" / "run_timings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update({k: round(v, 1) for k, v in timings.items()})
        path.write_text(json.dumps(existing, indent=2))

    total = len(BASELINES) if "baselines" in args.stage else 0
    total += sum(len(STAGES[s]) for s in args.stage if s != "baselines")
    print(f"\n{total} jobs in stages {args.stage}")


if __name__ == "__main__":
    main()
