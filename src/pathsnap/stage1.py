"""Command-line entry point for training-only Stage 1 diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import AlignmentConfig
from .diagnostics import run_training_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/source/dataset/Student folder"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/stage1"))
    parser.add_argument("--candidate-radius-m", type=float, default=80.0)
    parser.add_argument("--timestamp-gap-threshold-s", type=float, default=5.0)
    parser.add_argument("--map-count", type=int, default=3)
    args = parser.parse_args()
    if not args.data_root.exists():
        raise SystemExit(f"Dataset directory not found: {args.data_root}")
    config = AlignmentConfig(
        candidate_radius_m=args.candidate_radius_m,
        timestamp_gap_threshold_s=args.timestamp_gap_threshold_s,
    )
    rows = run_training_diagnostics(args.data_root, args.output_root, config, args.map_count)
    coverage = sum(row["candidate_coverage_pct"] for row in rows) / len(rows)
    print(f"Wrote diagnostics for {len(rows)} training trips; unweighted mean coverage={coverage:.2f}%.")


if __name__ == "__main__":
    main()
