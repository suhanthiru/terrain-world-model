"""Train one model.

    python scripts/train.py --run-id latentB_K5_z64_n2000_s0 --arch latentB
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wm.train import RunConfig, Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    for f in fields(RunConfig):
        if f.name == "run_id":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool" or isinstance(f.default, bool):
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction,
                                default=f.default)
        elif f.name == "curriculum":
            continue
        else:
            parser.add_argument(flag, dest=f.name, type=type(f.default), default=f.default)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    known = {f.name for f in fields(RunConfig)}
    cfg = RunConfig(**{k: v for k, v in vars(args).items() if k in known})

    out_dir = Path(args.out) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=list))

    print(f"[{cfg.run_id}] arch={cfg.arch} z={cfg.latent_dim} K={cfg.k_train} "
          f"n={cfg.train_size} seed={cfg.seed}", flush=True)
    summary = Trainer(cfg, out_dir).fit()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
