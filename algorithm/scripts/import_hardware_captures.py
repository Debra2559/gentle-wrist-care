#!/usr/bin/env python3
"""Convert and audit unlabeled wired SheWrist hardware captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.hardware_capture import import_capture_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets",
        help="Read-only directory containing imu_pressure_*.csv and wrist_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/hardware_capture_import",
        help="Directory for canonical copies and audit_report.json.",
    )
    parser.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=50.0,
        help="Expected formal capture rate used only for audit warnings; data are not resampled.",
    )
    parser.add_argument(
        "--adc-max",
        type=float,
        default=4095.0,
        help="ADC full-scale value used only to flag raw FSR saturation.",
    )
    args = parser.parse_args()
    if args.target_sample_rate_hz <= 0.0:
        parser.error("--target-sample-rate-hz must be positive")
    if args.adc_max <= 0.0:
        parser.error("--adc-max must be positive")

    report = import_capture_directory(
        args.source_dir,
        args.output_dir,
        target_sample_rate_hz=args.target_sample_rate_hz,
        adc_max=args.adc_max,
    )
    output = {
        "status": "PASS" if not report["rejected_files"] else "PARTIAL",
        "dataset_classification": report["dataset_classification"],
        "summary": report["summary"],
        "analysis_eligibility": report["analysis_eligibility"],
        "audit_report": report["audit_report_path"],
        "standardized_directory": str(args.output_dir.resolve() / "standardized"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    if report["rejected_files"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
