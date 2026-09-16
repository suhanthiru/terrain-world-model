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

from wm.runlog import redirect_output  # noqa: E402
from wm.data.storage import read_manifest  # noqa: E402


def available_episodes(data_root: str, split: str = "train") -> int:
    """How many episodes the manifest currently offers for a split."""
    try:
        return int(read_manifest(Path(data_root))["splits"][split]["n_episodes"])
    except (FileNotFoundError, KeyError):
        return 0


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


# Measured peak allocation per training step, plus about 0.7 GB for each process's own
# CUDA context. The five-step U-Net is the outlier at 8.7 GB and can only ever run alone.
GPU_GB = {
    ("latentB", 1): 1.2, ("latentB", 5): 2.3,
    ("latentA", 1): 1.2, ("latentA", 5): 2.5,
    ("unet", 1): 2.6, ("unet", 5): 9.4,
}
GPU_BUDGET_GB = 11.0

# How often to reap finished jobs. Negligible against runs that last an hour.
POLL_SECONDS = 5

# Each concurrent job holds its own copy of the training frames in RAM.
HOST_GB_PER_JOB = 3.4


def estimate_gpu_gb(cfg: dict) -> float:
    return GPU_GB.get((cfg["arch"], cfg["k_train"]), 3.0)


def run_pool(jobs: list[dict], max_jobs: int, dry_run: bool) -> tuple[dict, list[str], list[str]]:
    """Run training jobs concurrently, within a GPU memory budget.

    These models are small enough that a single one leaves the card mostly idle -- it
    spends much of its time launching kernels rather than executing them -- so several
    train at close to full speed side by side. The budget is what stops that from turning
    into an out-of-memory failure eleven hours in: jobs are admitted only while their
    estimated peak allocations still fit, which naturally serialises the five-step U-Net.
    """
    timings: dict[str, float] = {}
    failed: list[str] = []
    # Names of runs that actually produced a checkpoint. Reported explicitly rather than
    # inferred as "everything that did not fail" -- a job the pool abandoned never ran at
    # all, and treating it as trained would send evaluation after a checkpoint that does
    # not exist.
    succeeded: list[str] = []
    pending = list(jobs)
    running: list[dict] = []

    while pending or running:
        # Stop admitting once several jobs have failed and none has succeeded. If the
        # failures are slow ones -- running out of memory half an hour in, say -- letting
        # the whole stage play out would burn hours to learn nothing.
        if not running and failed and len(failed) >= CONSECUTIVE_FAILURE_LIMIT and not succeeded:
            print(f"!!! abandoning {len(pending)} unstarted job(s): "
                  f"{len(failed)} failed, none succeeded", flush=True)
            break

        while pending:
            candidate = pending[0]
            used = sum(j["gpu_gb"] for j in running)
            if running and (len(running) >= max_jobs
                            or used + candidate["gpu_gb"] > GPU_BUDGET_GB):
                break
            if len(failed) >= CONSECUTIVE_FAILURE_LIMIT and not succeeded:
                break
            pending.pop(0)
            print(f"\n=== start {candidate['label']} "
                  f"({candidate['gpu_gb']:.1f} GB, {len(running) + 1} running) ===", flush=True)
            print("$ " + " ".join(candidate["cmd"]), flush=True)
            if dry_run:
                timings[candidate["label"]] = 0.0
                succeeded.append(candidate["name"])
                continue
            log = ROOT / "runs" / candidate["name"]
            log.mkdir(parents=True, exist_ok=True)
            handle = (log / "train_stdout.log").open("w")
            candidate["proc"] = subprocess.Popen(
                candidate["cmd"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
            )
            candidate["handle"] = handle
            candidate["started"] = time.perf_counter()
            running.append(candidate)

        if not running:
            continue

        time.sleep(POLL_SECONDS)
        for job in list(running):
            if job["proc"].poll() is None:
                continue
            running.remove(job)
            job["handle"].close()
            elapsed = time.perf_counter() - job["started"]
            timings[job["label"]] = elapsed
            ok = job["proc"].returncode == 0
            if ok:
                succeeded.append(job["name"])
            status = "done" if ok else f"FAILED ({job['proc'].returncode})"
            print(f"=== {status}: {job['label']} after {elapsed / 60:.1f} min ===", flush=True)
            if not ok:
                failed.append(job["label"])
                print(f"    see runs/{job['name']}/train_stdout.log", flush=True)

    return timings, failed, succeeded



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="*", default=["core"], choices=[*STAGES, "baselines"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", default="data/v1")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="learning rate for stages other than the probe")
    parser.add_argument("--jobs", type=int, default=1,
                        help="train this many runs concurrently, within the GPU budget")
    parser.add_argument("--log-file", default=None,
                        help="write progress here, opened by this process (see wm.runlog)")
    args = parser.parse_args()
    redirect_output(args.log_file)

    if args.jobs > 1:
        print(f"training up to {args.jobs} runs at once "
              f"(GPU budget {GPU_BUDGET_GB:.0f} GB, ~{HOST_GB_PER_JOB:.1f} GB host RAM each)",
              flush=True)

    python = sys.executable
    episodes_available = available_episodes(args.data_root)
    print(f"train split currently offers {episodes_available} episodes", flush=True)
    timings: dict[str, float] = {}
    failed: list[str] = []
    all_deferred: list[str] = []
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

    def train_command(cfg: dict, stage: str) -> list[str]:
        cmd = [python, "scripts/train.py", "--run-id", cfg["run_id"],
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
        return cmd

    for stage in args.stage:
        if stage == "baselines":
            continue

        pending, trained, deferred = [], [], []
        for cfg in STAGES[stage]:
            name = cfg["run_id"]
            # summary.json is written only when fit() returns, so it marks a run that
            # actually finished. A checkpoint is not a completion marker -- one is saved
            # on every improving epoch, so a run killed after epoch one leaves a
            # perfectly valid-looking ckpt_best.pt behind.
            if args.skip_existing and (ROOT / "runs" / name / "summary.json").exists():
                print(f"\n=== {name}: already trained, skipping ===", flush=True)
                trained.append(name)
                continue
            # Not a failure: the dataset is simply still being generated. Running this
            # command again later picks the run up, and --skip-existing leaves the
            # finished ones alone.
            if cfg["train_size"] > episodes_available:
                print(f"\n=== {name}: deferred, needs {cfg['train_size']} episodes but "
                      f"only {episodes_available} are generated ===", flush=True)
                deferred.append(name)
                continue
            pending.append({"name": name, "label": f"train {name}",
                            "cmd": train_command(cfg, stage),
                            "gpu_gb": estimate_gpu_gb(cfg)})

        all_deferred.extend(deferred)

        if args.jobs > 1 and pending:
            pool_timings, pool_failed, pool_trained = run_pool(pending, args.jobs, args.dry_run)
            timings.update(pool_timings)
            failed.extend(pool_failed)
            trained += pool_trained
            # The consecutive-failure rule does not apply to jobs that ran side by side,
            # so apply the equivalent check: several failing and none succeeding is
            # systematic, and there is nothing to gain from starting the next stage.
            if len(pool_failed) >= CONSECUTIVE_FAILURE_LIMIT and not trained:
                raise SystemExit(
                    f"stopping: {len(pool_failed)} jobs in stage {stage!r} failed and none succeeded"
                )
        else:
            for job in pending:
                timings[job["name"]], ok = launch(job["cmd"], job["label"], args.dry_run)
                record(job["label"], ok)
                if ok:
                    trained.append(job["name"])

        # Evaluation stays serial: it is a small share of the wall clock and holds a
        # whole split of frames plus its predictions in host memory.
        # Probes are compared on dev five-step error alone, so there is nothing to learn
        # from evaluating a deliberately under-trained model on held-out splits.
        if stage != "lr":
            for name in trained:
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
