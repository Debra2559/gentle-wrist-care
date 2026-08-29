#!/usr/bin/env python3
"""Evaluate the public reference toolbox's aligned wrist IMU/OPTO angles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import load_config, write_json
from shewrist.dataset_registry import DatasetRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-config", type=Path, default=PROJECT_ROOT / "config/datasets.json")
    parser.add_argument("--dataset-id", default="comparison_imu_optotrak")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--aligned-pickle", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/opto_reference_validation.json")
    args = parser.parse_args()
    config = load_config(args.config)
    registry = DatasetRegistry.from_config(args.registry_config, PROJECT_ROOT)
    payload = registry.evaluate_angle_reference(
        args.dataset_id,
        config,
        root_override=args.dataset_root,
        aligned_pickle_override=args.aligned_pickle,
    )
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()