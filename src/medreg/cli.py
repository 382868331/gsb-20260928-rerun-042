from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import BatchItem, run_batch
from .synthetic import synthetic_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigid medical image registration batch runner")
    parser.add_argument("output", type=Path, help="batch output directory")
    parser.add_argument("--parameters", type=Path, help="JSON parameter file")
    parser.add_argument("--shape", type=int, nargs=3, default=(48, 56, 44))
    args = parser.parse_args()
    fixed, moving, _ = synthetic_pair(tuple(args.shape))
    item = BatchItem("synthetic_rigid", levels=(3, 2, 1), max_iterations=80)
    report = run_batch([(item, fixed, moving)], args.output, args.parameters)
    print(json.dumps(report["items"][0]["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
