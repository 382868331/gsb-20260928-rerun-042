from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import BatchItem, run_batch
from .series_batch import SeriesItem, run_series_batch
from .synthetic import synthetic_pair, synthetic_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigid medical image registration batch runner")
    parser.add_argument("output", type=Path, help="batch output directory")
    parser.add_argument("--parameters", type=Path, help="JSON parameter file")
    parser.add_argument("--shape", type=int, nargs=3, default=(48, 56, 44))
    parser.add_argument(
        "--series",
        action="store_true",
        help="run a synthetic time-series registration instead of a single pair",
    )
    args = parser.parse_args()
    if args.series:
        images, _ = synthetic_series(
            tuple(args.shape),
            time_ids=("t0", "t1", "t2", "t3"),
            jump_time_id="t2",
        )
        item = SeriesItem(
            "synthetic_series",
            reference_time_id="t0",
            time_order=("t0", "t1", "t2", "t3"),
            initial_parameter_sets=[[0.0] * 6, [0.05, -0.03, 0.03, 4.0, -2.0, 2.0]],
            levels=(3, 2, 1),
            max_iterations=80,
        )
        report = run_series_batch([(item, images)], args.output, args.parameters)
        item_report = report["items"][0]
        print(
            json.dumps(
                {
                    "version_id": item_report["version_id"],
                    "status": item_report["status"],
                    "jump_warnings": item_report["jump_warnings"],
                    "time_points": [
                        {
                            "time_id": point["time_id"],
                            "status": point["status"],
                            "rejection_reasons": point["rejection_reasons"],
                        }
                        for point in item_report["series"]["time_points"]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    fixed, moving, _ = synthetic_pair(tuple(args.shape))
    item = BatchItem("synthetic_rigid", levels=(3, 2, 1), max_iterations=80)
    report = run_batch([(item, fixed, moving)], args.output, args.parameters)
    item_report = report["items"][0]
    print(
        json.dumps(
            {
                "version_id": item_report["version_id"],
                "selection": item_report["selection"],
                "selected": item_report["result"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
