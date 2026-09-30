from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import BatchItem, run_batch
from .synthetic import synthetic_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigid medical image registration batch runner")
    parser.add_argument("output", type=Path, help="batch output directory (each run adds versions/)")
    parser.add_argument("--parameters", type=Path, help="JSON parameter file")
    parser.add_argument("--shape", type=int, nargs=3, default=(40, 44, 36))
    args = parser.parse_args()
    fixed, moving, _ = synthetic_pair(tuple(args.shape))
    item = BatchItem("synthetic_rigid", levels=(3, 2, 1), max_iterations=50)
    report = run_batch([(item, fixed, moving)], args.output, args.parameters)
    item_report = report["items"][0]
    summary = item_report["result"]
    print(json.dumps(
        {
            "version": report["version"],
            "selected_candidate_id": item_report["selected_candidate_id"],
            "status": summary["status"],
            "uncertainty": summary["uncertainty"],
            "candidate_count": len(summary["candidates"]),
            "outputs": item_report["outputs"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
