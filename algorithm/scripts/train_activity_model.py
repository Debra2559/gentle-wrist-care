#!/usr/bin/env python3
"""Train and evaluate the public wrist-motion shadow model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import load_config, write_json
from shewrist.dataset_registry import DatasetRegistry
from shewrist.hmm import TemporalHMM
from shewrist.ml import NumpyTemporalCNN, ShadowActivityPipeline, label_sequences
from shewrist.ml_evaluation import (
    ParticipantSplit,
    balanced_indices,
    evaluate_prediction,
    evaluate_prediction_by_dataset,
    loso_splits,
    robustness_curve,
    train_split_pipeline,
)
from shewrist.tokens import build_inertial_tokens


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset_id",
        "subject_id",
        "session_id",
        "sequence_id",
        "start_s",
        "end_s",
        "true_label",
        "raw_prediction",
        "hmm_prediction",
        "accepted_prediction",
        "confidence",
        "mean_quality",
        "missing_fraction",
        "rejection_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(folds: list[dict[str, object]]) -> dict[str, object]:
    paths = {
        "raw_accuracy": ("metrics", "raw_window", "accuracy"),
        "raw_macro_f1": ("metrics", "raw_window", "macro_f1"),
        "hmm_accuracy": ("metrics", "hmm_window", "accuracy"),
        "hmm_macro_f1": ("metrics", "hmm_window", "macro_f1"),
        "accepted_selective_accuracy": ("metrics", "accepted_window", "selective_accuracy"),
        "accepted_macro_f1": ("metrics", "accepted_window", "macro_f1"),
        "coverage": ("metrics", "accepted_window", "coverage"),
        "event_f1": ("metrics", "event", "f1"),
        "false_positive_events_per_hour": ("metrics", "event", "false_positive_events_per_hour"),
        "calibration_error": ("metrics", "calibration_error"),
        "training_seconds": ("training", "training_seconds"),
    }
    summary: dict[str, object] = {}
    for name, path in paths.items():
        values = []
        for fold in folds:
            value: object = fold
            for key in path:
                value = value[key]  # type: ignore[index]
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    return summary


def _aggregate_robustness(folds: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[float, float], list[dict[str, object]]] = {}
    for fold in folds:
        for row in fold["robustness"]:  # type: ignore[index]
            key = (float(row["noise_fraction_of_feature_std"]), float(row["missing_sample_fraction"]))
            buckets.setdefault(key, []).append(row)
    results = []
    for (noise, missing), rows in sorted(buckets.items()):
        results.append(
            {
                "noise_fraction_of_feature_std": noise,
                "missing_sample_fraction": missing,
                "macro_f1_mean": float(np.mean([float(row["macro_f1"]) for row in rows])),
                "coverage_mean": float(np.mean([float(row["coverage"]) for row in rows])),
                "event_f1_mean": float(np.mean([float(row["event_f1"]) for row in rows])),
            }
        )
    return results


def _train_final(dataset, config, epochs: int, temperature: float, output: Path) -> ShadowActivityPipeline:
    all_indices = np.arange(len(dataset), dtype=int)
    balanced = balanced_indices(dataset.labels, all_indices, seed=int(config["model"]["seed"]) + 1000)
    cnn = NumpyTemporalCNN(
        len(dataset.feature_names),
        len(dataset.class_names),
        int(config["model"]["filters"]),
        int(config["model"]["kernel_size"]),
        int(config["model"]["seed"]) + 1000,
        pooling=str(config["model"].get("pooling", "mean")),
    )
    cnn.fit(
        dataset.windows[balanced],
        dataset.labels[balanced],
        None,
        None,
        dataset.feature_names,
        config["model"],
        config["augmentation"],
        epochs,
    )
    cnn.temperature = float(temperature)
    hmm_config = config["hmm"]
    hmm = TemporalHMM.fit(
        label_sequences(dataset, all_indices),
        len(dataset.class_names),
        float(hmm_config["laplace"]),
        float(hmm_config["self_transition_prior"]),
        float(hmm_config["background_transition_prior"]),
    )
    pipeline = ShadowActivityPipeline(
        cnn,
        hmm,
        dataset.class_names,
        dataset.feature_names,
        float(config["model"]["confidence_threshold"]),
        float(config["window"]["min_window_quality"]),
        {
            "task": config["task"],
            "training_scope": "all_public_participants_after_loso_model_selection",
            "training_subjects": sorted(set(dataset.subject_ids.tolist())),
            "training_dataset_ids": sorted(set(dataset.dataset_ids.tolist())),
            "epochs": epochs,
            "temperature_source": "median of participant-held-out validation calibrations",
            "evidence_limit": "Public healthy-participant wrist-motion labels only; not a strain, pain, disease, pressure, or clinical model.",
        },
        float(config["window"].get("max_missing_fraction", 1.0)),
    )
    pipeline.save(output)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-config",
        type=Path,
        default=PROJECT_ROOT / "config/datasets.json",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Registered compatible activity dataset ID; repeatable.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Legacy root override; valid only when one --dataset-id is selected.",
    )
    parser.add_argument("--algorithm-config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/ml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for each LOSO fold.")
    parser.add_argument("--fold", action="append", default=[], help="Run only the named test subject; repeatable.")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()

    algorithm_config = load_config(args.algorithm_config)
    ml_config = load_config(args.ml_config)
    registry = DatasetRegistry.from_config(args.registry_config, PROJECT_ROOT)
    dataset_ids = tuple(dict.fromkeys(args.dataset_id or ["upper_body_movements"]))
    if args.dataset is not None and len(dataset_ids) != 1:
        raise SystemExit("--dataset can override the root only when one --dataset-id is selected")
    root_overrides = {} if args.dataset is None else {dataset_ids[0]: args.dataset}
    started = perf_counter()
    dataset = registry.build_activity_datasets(
        dataset_ids,
        algorithm_config,
        ml_config,
        root_overrides=root_overrides,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_folds = set(args.fold)
    folds = [split for split in loso_splits(dataset.subject_ids) if not selected_folds or split.test_subject in selected_folds]
    if not folds:
        raise SystemExit("no requested folds match available subjects")
    fold_reports: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    for fold_number, split in enumerate(folds, start=1):
        print(f"[{fold_number}/{len(folds)}] test={split.test_subject} validation={split.validation_subject}", flush=True)
        pipeline, training = train_split_pipeline(dataset, split, ml_config, args.epochs)
        test_indices = np.flatnonzero(dataset.subject_ids == split.test_subject)
        test_data = dataset.subset(test_indices)
        prediction = pipeline.predict(test_data)
        metrics = evaluate_prediction(test_data, prediction)
        metrics_by_dataset = evaluate_prediction_by_dataset(test_data, prediction)
        robustness = [] if args.skip_robustness else robustness_curve(
            pipeline,
            test_data,
            seed=int(ml_config["model"]["seed"]) + fold_number,
        )
        fold_reports.append(
            {
                "test_subject": split.test_subject,
                "validation_subject": split.validation_subject,
                "train_subjects": list(split.train_subjects),
                "training": training,
                "metrics": metrics,
                "metrics_by_dataset": metrics_by_dataset,
                "robustness": robustness,
            }
        )
        for local_index in range(len(test_data)):
            def label_name(value: int) -> str:
                return "unknown" if value < 0 else test_data.class_names[value]
            prediction_rows.append(
                {
                    "dataset_id": str(test_data.dataset_ids[local_index]),
                    "subject_id": str(test_data.subject_ids[local_index]),
                    "session_id": str(test_data.session_ids[local_index]),
                    "sequence_id": str(test_data.sequence_ids[local_index]),
                    "start_s": float(test_data.start_s[local_index]),
                    "end_s": float(test_data.end_s[local_index]),
                    "true_label": label_name(int(test_data.labels[local_index])),
                    "raw_prediction": label_name(int(prediction.raw_labels[local_index])),
                    "hmm_prediction": label_name(int(prediction.smoothed_labels[local_index])),
                    "accepted_prediction": label_name(int(prediction.accepted_labels[local_index])),
                    "confidence": float(prediction.confidence[local_index]),
                    "mean_quality": float(test_data.mean_quality[local_index]),
                    "missing_fraction": float(test_data.missing_fraction[local_index]),
                    "rejection_reason": str(prediction.rejection_reason[local_index]),
                }
            )
        token_rows.extend(
            token.to_dict()
            for token in build_inertial_tokens(
                prediction.accepted_labels,
                prediction.confidence,
                test_data.start_s,
                test_data.end_s,
                test_data.mean_quality,
                test_data.windows,
                test_data.class_names,
                test_data.feature_names,
                f"{split.test_subject}-set2",
                sequence_ids=test_data.sequence_ids,
            )
        )
        print(
            json.dumps(
                {
                    "accepted_macro_f1": metrics["accepted_window"]["macro_f1"],
                    "coverage": metrics["accepted_window"]["coverage"],
                    "event_f1": metrics["event"]["f1"],
                    "seconds": training["training_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    best_epochs = [int(fold["training"]["best_epoch"]) for fold in fold_reports]
    final_epochs = max(1, int(round(median(best_epochs))))
    final_temperature = float(median(float(fold["training"]["temperature"]) for fold in fold_reports))
    model_path = args.output_dir / "activity_cnn_hmm_shadow.npz"
    if len(folds) == len(set(dataset.subject_ids.tolist())):
        print(f"Training final all-participant model for {final_epochs} epochs", flush=True)
        _train_final(dataset, ml_config, final_epochs, final_temperature, model_path)
    report = {
        "schema_version": 1,
        "task": ml_config["task"],
        "operating_mode": "shadow",
        "safety": ml_config["safety"],
        "dataset": {
            "name": "Upper-body movements: precise tracking of human motion using inertial sensors" if dataset_ids == ("upper_body_movements",) else "Registered compatible activity datasets",
            "dataset_ids": list(dataset_ids),
            "dataset_counts": dataset.dataset_counts(),
            "sources": [registry.descriptor(dataset_id).to_dict() for dataset_id in dataset_ids],
            "subjects": len(set(dataset.subject_ids.tolist())),
            "windows": len(dataset),
            "window_shape": list(dataset.windows.shape[1:]),
            "label_counts": dataset.label_counts(),
            "calibration_assistance": "Dataset adapters must isolate calibration repetitions from classifier labels.",
            "ground_truth_limit": "Registered activity labels only; no tissue strain, pain, pressure, disease, or clinical ground truth.",
        },
        "evaluation_design": "Participant-disjoint LOSO within the registered compatible activity pool; every test fold also reports metrics by dataset source.",
        "cross_dataset_evaluation": {
            "status": "ready_via_scripts/evaluate_cross_dataset_activity.py" if len(dataset_ids) >= 2 else "not_evaluable",
            "reason": None if len(dataset_ids) >= 2 else "only_one_compatible_labeled_activity_dataset_selected",
            "dataset_ids": list(dataset_ids),
        },
        "folds": fold_reports,
        "aggregate": _aggregate(fold_reports),
        "robustness_aggregate": _aggregate_robustness(fold_reports),
        "selected_final_epochs": final_epochs,
        "selected_final_temperature": final_temperature,
        "model_path": str(model_path) if model_path.exists() else None,
        "elapsed_seconds": perf_counter() - started,
        "interpretation": [
            "Metrics describe public-dataset wrist-motion classification, not strain or disease prediction.",
            "Rejected predictions have no alarm or mechanical effect.",
            "The deterministic angle/pressure state machine remains the only alarm and release path.",
            "A production activity model requires target-hardware, cross-session, and independently labeled data.",
        ],
    }
    write_json(args.output_dir / "loso_report.json", report)
    write_json(
        args.output_dir / "summary.json",
        {
            "schema_version": report["schema_version"],
            "task": report["task"],
            "operating_mode": report["operating_mode"],
            "safety": report["safety"],
            "dataset": report["dataset"],
            "evaluation_design": report["evaluation_design"],
            "cross_dataset_evaluation": report["cross_dataset_evaluation"],
            "aggregate": report["aggregate"],
            "robustness_aggregate": report["robustness_aggregate"],
            "selected_final_epochs": report["selected_final_epochs"],
            "selected_final_temperature": report["selected_final_temperature"],
            "model_path": report["model_path"],
            "elapsed_seconds": report["elapsed_seconds"],
            "interpretation": report["interpretation"],
        },
    )
    write_json(args.output_dir / "oof_tokens.json", {"operating_mode": "shadow", "tokens": token_rows})
    _write_predictions(args.output_dir / "oof_predictions.csv", prediction_rows)
    print(json.dumps({"report": str(args.output_dir / "loso_report.json"), "model": report["model_path"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()