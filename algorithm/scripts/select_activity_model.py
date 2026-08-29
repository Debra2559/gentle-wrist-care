#!/usr/bin/env python3
"""Limited activity-model selection with one untouched participant test set."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import load_config, write_json
from shewrist.dataset_registry import DatasetRegistry
from shewrist.hmm import TemporalHMM
from shewrist.ml import ShadowActivityPipeline, label_sequences
from shewrist.ml_evaluation import ParticipantSplit, evaluate_prediction, train_split_pipeline


def compact_metrics(metrics: dict[str, object]) -> dict[str, float | None]:
    return {
        "raw_macro_f1": float(metrics["raw_window"]["macro_f1"]),
        "hmm_macro_f1": float(metrics["hmm_window"]["macro_f1"]),
        "accepted_macro_f1": float(metrics["accepted_window"]["macro_f1"]),
        "accepted_selective_accuracy": None if metrics["accepted_window"]["selective_accuracy"] is None else float(metrics["accepted_window"]["selective_accuracy"]),
        "coverage": float(metrics["accepted_window"]["coverage"]),
        "event_f1": float(metrics["event"]["f1"]),
        "calibration_error": float(metrics["calibration_error"]),
    }


def selection_key(row: dict[str, object]) -> tuple[float, float, float]:
    metrics = row["validation_metrics"]
    return (
        float(metrics["accepted_macro_f1"]),
        float(metrics["event_f1"]),
        float(metrics["coverage"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-config", type=Path, default=PROJECT_ROOT / "config/datasets.json")
    parser.add_argument("--dataset-id", default="upper_body_movements")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional registered dataset root override.")
    parser.add_argument("--algorithm-config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--validation-subject", default="subject10")
    parser.add_argument("--test-subject", default="subject11")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/model_selection")
    args = parser.parse_args()

    started = perf_counter()
    algorithm_config = load_config(args.algorithm_config)
    base_config = load_config(args.ml_config)
    registry = DatasetRegistry.from_config(args.registry_config, PROJECT_ROOT)
    subjects = registry.activity_subjects(args.dataset_id, args.dataset)
    if args.validation_subject == args.test_subject:
        raise SystemExit("validation and test subjects must differ")
    if args.validation_subject not in subjects or args.test_subject not in subjects:
        raise SystemExit("validation or test subject is absent from the dataset")
    training_subjects = tuple(value for value in subjects if value not in {args.validation_subject, args.test_subject})
    selection_subjects = (*training_subjects, args.validation_subject)
    dataset = registry.build_activity_dataset(
        args.dataset_id,
        algorithm_config,
        base_config,
        subjects=selection_subjects,
        root_override=args.dataset,
    )
    split = ParticipantSplit(training_subjects, args.validation_subject, args.test_subject)
    train_full = np.flatnonzero(dataset.subject_mask(split.train_subjects))
    validation_data = dataset.subset(np.flatnonzero(dataset.subject_ids == split.validation_subject))
    candidates: list[dict[str, object]] = []
    candidate_pipelines: list[ShadowActivityPipeline] = []
    architecture_training: dict[str, object] = {}
    pooling_options = ("mean", "mean_max")
    threshold_options = (0.45, 0.55, 0.65)
    self_transition_options = (10.0, 20.0, 40.0)
    for pooling in pooling_options:
        config = copy.deepcopy(base_config)
        config["model"]["pooling"] = pooling
        fixed_selection_seed = int(base_config["model"]["seed"]) + pooling_options.index(pooling)
        base_pipeline, training = train_split_pipeline(
            dataset,
            split,
            config,
            args.epochs,
            seed_override=fixed_selection_seed,
        )
        training["selection_seed"] = fixed_selection_seed
        architecture_training[pooling] = training
        for self_transition in self_transition_options:
            hmm = TemporalHMM.fit(
                label_sequences(dataset, train_full),
                len(dataset.class_names),
                float(config["hmm"].get("laplace", 0.5)),
                self_transition,
                float(config["hmm"].get("background_transition_prior", 2.0)),
            )
            for threshold in threshold_options:
                pipeline = ShadowActivityPipeline(
                    base_pipeline.cnn,
                    hmm,
                    dataset.class_names,
                    dataset.feature_names,
                    threshold,
                    float(config["window"]["min_window_quality"]),
                    {
                        **base_pipeline.metadata,
                        "selection_protocol": "validation_only_locked_test",
                        "pooling": pooling,
                        "confidence_threshold": threshold,
                        "hmm_self_transition_prior": self_transition,
                        "training_dataset_ids": [args.dataset_id],
                    },
                    float(config["window"].get("max_missing_fraction", 1.0)),
                )
                validation_metrics = compact_metrics(evaluate_prediction(validation_data, pipeline.predict(validation_data)))
                candidates.append(
                    {
                        "pooling": pooling,
                        "confidence_threshold": threshold,
                        "hmm_self_transition_prior": self_transition,
                        "validation_metrics": validation_metrics,
                    }
                )
                candidate_pipelines.append(pipeline)
    selected_index = max(range(len(candidates)), key=lambda index: selection_key(candidates[index]))
    selected = candidates[selected_index]
    selected_pipeline = candidate_pipelines[selected_index]
    test_data = registry.build_activity_dataset(
        args.dataset_id,
        algorithm_config,
        base_config,
        subjects=(args.test_subject,),
        root_override=args.dataset,
    )
    locked_test_metrics = compact_metrics(evaluate_prediction(test_data, selected_pipeline.predict(test_data)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "activity_cnn_hmm_locked_split.npz"
    selected_pipeline.save(model_path)
    selected_config = copy.deepcopy(base_config)
    selected_config["model"]["pooling"] = selected["pooling"]
    selected_config["model"]["confidence_threshold"] = selected["confidence_threshold"]
    selected_config["hmm"]["self_transition_prior"] = selected["hmm_self_transition_prior"]
    selected_config["selection"] = {
        "protocol": "validation_only_locked_test",
        "training_subjects": list(training_subjects),
        "validation_subject": args.validation_subject,
        "test_subject": args.test_subject,
        "objective_order": ["accepted_macro_f1", "event_f1", "coverage"],
        "test_access": "once_after_configuration_lock",
    }
    write_json(args.output_dir / "selected_ml_activity.json", selected_config)
    report = {
        "schema_version": 1,
        "task": base_config["task"],
        "selection_scope": "limited engineering model selection",
        "data_split": {
            "dataset_id": args.dataset_id,
            "training_subjects": list(training_subjects),
            "validation_subject": args.validation_subject,
            "test_subject": args.test_subject,
            "test_used_for_selection": False,
            "test_evaluation_count": 1,
        },
        "search_space": {
            "pooling": list(pooling_options),
            "confidence_threshold": list(threshold_options),
            "hmm_self_transition_prior": list(self_transition_options),
            "candidate_count": len(candidates),
            "cnn_training_count": len(pooling_options),
        },
        "selection_rule": "Lexicographic maximum of validation accepted macro-F1, event F1, then coverage.",
        "architecture_training": architecture_training,
        "validation_candidates": candidates,
        "selected_configuration": {key: selected[key] for key in ("pooling", "confidence_threshold", "hmm_self_transition_prior")},
        "selected_validation_metrics": selected["validation_metrics"],
        "locked_test_metrics": locked_test_metrics,
        "model_path": str(model_path),
        "elapsed_seconds": perf_counter() - started,
        "interpretation": [
            "The test participant was not used to choose pooling, threshold, HMM prior, early stopping, or temperature.",
            "This one locked split is a limited engineering comparison, not a replacement for nested cross-validation.",
            "The selected model remains shadow-only and has no alarm or mechanical-control authority.",
        ],
    }
    write_json(args.output_dir / "selection_report.json", report)
    print(json.dumps({
        "status": "passed",
        "selected": report["selected_configuration"],
        "validation": report["selected_validation_metrics"],
        "locked_test": report["locked_test_metrics"],
        "report": str(args.output_dir / "selection_report.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()