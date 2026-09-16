"""Summarise the learning-rate probes and name the best rate per architecture.

    python scripts/report_lr_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rows = []
    for summary in sorted((ROOT / "runs").glob("lrprobe_*/summary.json")):
        payload = json.loads(summary.read_text())
        config = json.loads((summary.parent / "config.json").read_text())
        rows.append({
            "run_id": payload["run_id"], "arch": config["arch"], "lr": config["lr"],
            "dev_mae": payload["best_dev_loss"], "identity": payload["identity_dev_loss"],
        })
    if not rows:
        raise SystemExit("no probes found -- run scripts/run_matrix.py --stage lr first")

    print(f"{'architecture':12s} {'lr':>8s} {'dev 5-step MAE':>15s} {'vs identity':>12s}")
    best: dict[str, dict] = {}
    for row in sorted(rows, key=lambda r: (r["arch"], r["lr"])):
        ratio = row["dev_mae"] / row["identity"]
        print(f"{row['arch']:12s} {row['lr']:8.0e} {row['dev_mae']:15.4f} {ratio:11.3f}x")
        if row["arch"] not in best or row["dev_mae"] < best[row["arch"]]["dev_mae"]:
            best[row["arch"]] = row

    print("\nbest per architecture:")
    for arch, row in best.items():
        print(f"  {arch}: {row['lr']:.0e}")

    out = ROOT / "results" / "lr_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"probes": rows, "best": {a: r["lr"] for a, r in best.items()}}, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
