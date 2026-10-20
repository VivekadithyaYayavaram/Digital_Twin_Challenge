"""Command line: python -m swasthtwin.cli {data|train|evaluate|all}"""
from __future__ import annotations

import argparse
import sys

from . import pipeline as P


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="swasthtwin", description=__doc__)
    ap.add_argument("step", choices=["data", "train", "evaluate", "all"])
    args = ap.parse_args(argv)
    try:
        if args.step in ("data", "all"):
            P.step_data()
        if args.step in ("train", "all"):
            P.step_train()
        if args.step in ("evaluate", "all"):
            P.step_evaluate()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
