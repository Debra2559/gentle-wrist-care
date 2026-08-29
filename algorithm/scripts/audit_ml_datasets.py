#!/usr/bin/env python3
"""Audit local ML datasets, adapters, expert readiness, and evidence boundaries."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-config", type=Path, default=PROJECT_ROOT / "config/datasets.json")
    parser.add_argument("--algorithm-config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/datasets/readiness_report.json")
    parser.add_argument("--skip-deep", action="store_true", help="Skip building ready activity datasets and angle metrics.")
    args = parser.parse_args()

    registry = DatasetRegistry.from_config(args.registry_config, PROJECT_ROOT)
    report = registry.readiness_summary()
    report["registry_config"] = str(args.registry_config.resolve())
    report["expert_interfaces"] = {
        descriptor.expert_id: {
            "dataset_ids": [
                item.dataset_id
                for item in (registry.descriptor(dataset_id) for dataset_id in registry.ids())
                if item.expert_id == descriptor.expert_id
            ],
            "operating_mode": "shadow",
            "control_authority": "none",
        }
        for descriptor in (registry.descriptor(dataset_id) for dataset_id in registry.ids())
    }
    if not args.skip_deep:
        algorithm_config = load_config(args.algorithm_config)
        ml_config = load_config(args.ml_config)
        activity = {}
        for dataset_id in report["ready_activity_dataset_ids"]:
            dataset = registry.build_activity_dataset(dataset_id, algorithm_config, ml_config)
            activity[dataset_id] = {
                "status": "ready",
                "window_count": len(dataset),
                "participant_count": len(set(dataset.subject_ids.tolist())),
                "session_count": len(set(dataset.session_ids.tolist())),
                "label_counts": dataset.label_counts(),
                "feature_names": list(dataset.feature_names),
                "missing_window_count": int((dataset.missing_fraction > 0.0).sum()),
            }
        report["activity_adapter_results"] = activity
        references = {}
        for dataset_id in report["usable_angle_reference_dataset_ids"]:
            references[dataset_id] = registry.evaluate_angle_reference(dataset_id, algorithm_config)
        report["angle_reference_results"] = references
    write_json(args.output, report)
    print(json.dumps({
        "status": "passed",
        "ready_activity_dataset_ids": report["ready_activity_dataset_ids"],
        "cross_dataset_activity_evaluation": report["cross_dataset_activity_evaluation"],
        "fusion_policy": report["fusion_policy"],
        "report": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
