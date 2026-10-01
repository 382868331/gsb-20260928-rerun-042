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
        type=int,
        metavar="N",
        help="run a temporal series of N synthetic timepoints instead of a single pair",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="timepoint index used as the registration reference in series mode",
    )
    args = parser.parse_args()

    if args.series:
        images, _ = synthetic_series(n_timepoints=args.series, shape=tuple(args.shape))
        item = SeriesItem(
            "synthetic_series",
            reference_index=args.reference_index,
            levels=(3, 2, 1),
            max_iterations=80,
        )
        report = run_series_batch([(item, images)], args.output, args.parameters)
        entry = report["series"][0]
        print(
            json.dumps(
                {
                    "version_id": entry["version_id"],
                    "status": entry["status"],
                    "adjacency": entry["adjacency"],
                    "timepoints": [
                        {"label": tp["label"], "status": tp["status"]}
                        for tp in entry["timepoints"]
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

