"""Wait for the dataset and for the running matrix to finish, then train what is left.

    python scripts/run_when_ready.py --log-file followup.log

Two of the core runs need more training episodes than have been generated so far, so
run_matrix defers them. This waits for those episodes to exist and for the current sweep
to release the GPU, then runs the remaining stages. It is safe to start at any time: every
stage uses --skip-existing, so finished runs are left alone.

Launch it detached and let it sit. It polls rather than chaining off a process handle,
because a handle owned by whatever shell started it is exactly the kind of thing that
dies and takes the job with it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wm.data.generate import SPLITS  # noqa: E402
from wm.data.storage import read_manifest  # noqa: E402
from wm.runlog import redirect_output  # noqa: E402

POLL_SECONDS = 120
PATIENCE_HOURS = 24


def episodes_ready(data_root: Path, split: str = "train") -> int:
    try:
        return int(read_manifest(data_root)["splits"][split]["n_episodes"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0


def matrix_running() -> bool:
    """Whether another run_matrix or train.py process holds the GPU."""
    query = (
        "@(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
        "Where-Object { $_.CommandLine -match 'run_matrix|train\\.py' }).Count"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                             capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip() or 0) > 0
    except Exception as exc:  # a failed probe must not end the wait
        print(f"  could not probe for running jobs ({exc}); assuming busy", flush=True)
        return True


def wait_for(label: str, predicate, deadline: float) -> bool:
    print(f"waiting for {label}...", flush=True)
    while time.time() < deadline:
        if predicate():
            print(f"  {label}: ready", flush=True)
            return True
        time.sleep(POLL_SECONDS)
    print(f"!!! gave up waiting for {label}", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/v1")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--stages", nargs="*", default=["core", "extra"])
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--skip-wait", action="store_true",
                        help="run immediately instead of waiting")
    args = parser.parse_args()
    redirect_output(args.log_file)

    data_root = Path(args.data_root)
    needed = SPLITS["train"].n_episodes
    deadline = time.time() + PATIENCE_HOURS * 3600

    if not args.skip_wait:
        ready = wait_for(
            f"{needed} training episodes",
            lambda: episodes_ready(data_root) >= needed,
            deadline,
        )
        if not ready:
            # Not a reason to stop: whatever exists is still worth training on, and
            # run_matrix defers anything it cannot supply rather than failing it.
            print("proceeding with the episodes that exist", flush=True)

        # A moment's grace so the generator finishes writing its manifest.
        time.sleep(30)
        wait_for("the GPU to be free", lambda: not matrix_running(), deadline)

    print(f"\ntraining episodes available: {episodes_ready(data_root)}", flush=True)
    failures = []
    for stage in args.stages:
        cmd = [sys.executable, "scripts/run_matrix.py", "--stage", stage,
               "--jobs", str(args.jobs), "--lr", args.lr,
               "--data-root", args.data_root]
        print(f"\n### stage {stage}\n$ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"!!! stage {stage} exited with {result.returncode}", flush=True)
            failures.append(stage)

    print(f"\nfinished; {len(failures)} stage(s) reported failures: {failures}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
