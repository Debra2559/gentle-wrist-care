#!/usr/bin/env python3
"""Run leave-one-dataset-out evaluation for compatible activity datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import load_config, write_json
from shewrist.dataset_registry import DatasetRegistry
from shewrist.ml_evaluation import (
    ParticipantSplit,
    evaluate_prediction,
    leave_one_dataset_out_splits,
    train_split_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-config", type=Path, default=PROJECT_ROOT / "config/datasets.json")
    parser.add_argument("--dataset-id", action="append", default=[], help="Compatible activity dataset ID; repeatable.")
    parser.add_argument("--algorithm-config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/datasets/cross_dataset_activity.json")
    args = parser.parse_args()

    registry = DatasetRegistry.from_config(args.registry_config, PROJECT_ROOT)
    readiness = registry.readiness_summary()
    selected = tuple(dict.fromkeys(args.dataset_id or readiness["ready_activity_dataset_ids"]))
    report: dict[str, object] = {
        "schema_version": 1,
        "protocol": "leave_one_dataset_out_activity_evaluation",
        "dataset_ids": list(selected),
        "operating_mode": "shadow",
        "control_authority": "none",
        "evidence_limit": "Activity transfer only; not wrist-angle truth, strain, pain, disease, pressure, or safety validation.",
    }
    if len(selected) < 2:
        report.update(
            {
                "status": "not_evaluable",
                "reason": "at_least_two_compatible_labeled_activity_datasets_are_required",
                "folds": [],
            }
        )
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "reason": report["reason"], "report": str(args.output.resolve())}, ensure_ascii=False))
        return

    algorithm_config = load_config(args.algorithm_config)
    ml_config = load_config(args.ml_config)
    dataset = registry.build_activity_datasets(selected, algorithm_config, ml_config)
    folds = []
    for fold_index, holdout in enumerate(leave_one_dataset_out_splits(dataset.dataset_ids), start=1):
        train_pool = np.flatnonzero(np.isin(dataset.dataset_ids, holdout.train_dataset_ids))
        candidate_subjects = sorted(set(dataset.subject_ids[train_pool].tolist()))
        if len(candidate_subjects) < 2:
            raise SystemExit(f"{holdout.test_dataset_id}: training sources need at least two participants")
        validation_subject = candidate_subjects[-1]
        training_subjects = tuple(value for value in candidate_subjects if value != validation_subject)
        split = ParticipantSplit(training_subjects, validation_subject, f"dataset:{holdout.test_dataset_id}")
        pipeline, training = train_split_pipeline(
            dataset,
            split,
            ml_config,
            args.epochs,
            seed_override=int(ml_config["model"].get("seed", 0)) + fold_index,
        )
        test_indices = np.flatnonzero(dataset.dataset_ids == holdout.test_dataset_id)
        test_data = dataset.subset(test_indices)
        metrics = evaluate_prediction(test_data, pipeline.predict(test_data))
        folds.append(
            {
                "held_out_dataset_id": holdout.test_dataset_id,
                "training_dataset_ids": list(holdout.train_dataset_ids),
                "training_subject_count": len(training_subjects),
                "validation_subject": validation_subject,
                "test_window_count": len(test_data),
                "training": training,
                "metrics": metrics,
            }
        )
    report.update({"status": "evaluated", "folds": folds})
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "fold_count": len(folds), "report": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
