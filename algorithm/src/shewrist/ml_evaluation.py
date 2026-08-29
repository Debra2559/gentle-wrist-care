"""Participant-wise evaluation and robustness checks for the shadow model."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .hmm import TemporalHMM
from .ml import NumpyTemporalCNN, ShadowActivityPipeline, ShadowPrediction, classification_metrics, label_sequences
from .ml_data import WindowDataset


@dataclass(frozen=True)
class ParticipantSplit:
    train_subjects: tuple[str, ...]
    validation_subject: str
    test_subject: str


@dataclass(frozen=True)
class DatasetHoldoutSplit:
    train_dataset_ids: tuple[str, ...]
    test_dataset_id: str


def leave_one_dataset_out_splits(dataset_ids: np.ndarray) -> list[DatasetHoldoutSplit]:
    datasets = tuple(sorted(str(value) for value in set(np.asarray(dataset_ids, dtype=object))))
    if len(datasets) < 2:
        raise ValueError("cross-dataset evaluation requires at least two compatible datasets")
    return [
        DatasetHoldoutSplit(
            tuple(dataset_id for dataset_id in datasets if dataset_id != test_dataset_id),
            test_dataset_id,
        )
        for test_dataset_id in datasets
    ]


def loso_splits(subject_ids: np.ndarray) -> list[ParticipantSplit]:
    subjects = tuple(sorted(str(value) for value in set(np.asarray(subject_ids, dtype=object))))
    if len(subjects) < 3:
        raise ValueError("LOSO evaluation requires at least three participants")
    splits = []
    for test_index, test_subject in enumerate(subjects):
        validation_subject = subjects[(test_index + 1) % len(subjects)]
        train = tuple(subject for subject in subjects if subject not in {test_subject, validation_subject})
        splits.append(ParticipantSplit(train, validation_subject, test_subject))
    return splits


def balanced_indices(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    background_ratio: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Keep all movement windows and cap background windows deterministically."""
    y = np.asarray(labels, dtype=int)
    candidate = np.asarray(candidate_indices, dtype=int)
    foreground = candidate[y[candidate] > 0]
    background = candidate[y[candidate] == 0]
    if len(foreground) == 0 or len(background) == 0:
        return np.sort(candidate)
    counts = np.bincount(y[foreground])
    largest_class = int(np.max(counts[1:])) if len(counts) > 1 else len(foreground)
    target_background = min(len(background), max(1, int(round(background_ratio * largest_class))))
    rng = np.random.default_rng(seed)
    selected_background = rng.choice(background, size=target_background, replace=False)
    return np.sort(np.concatenate((foreground, selected_background)))


def expected_calibration_error(true: np.ndarray, predicted: np.ndarray, confidence: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(true, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    conf = np.asarray(confidence, dtype=float)
    if not (len(y) == len(pred) == len(conf)) or bins < 1:
        raise ValueError("invalid calibration inputs")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (conf >= edges[index]) & (conf <= edges[index + 1])
        else:
            selected = (conf >= edges[index]) & (conf < edges[index + 1])
        if np.any(selected):
            accuracy = np.mean(y[selected] == pred[selected])
            error += np.mean(selected) * abs(float(accuracy) - float(np.mean(conf[selected])))
    return float(error)


def _runs(dataset: WindowDataset, labels: np.ndarray, foreground_only: bool) -> list[dict[str, object]]:
    values = np.asarray(labels, dtype=int)
    events: list[dict[str, object]] = []
    for sequence in dict.fromkeys(dataset.sequence_ids.tolist()):
        indices = np.flatnonzero(dataset.sequence_ids == sequence)
        position = 0
        while position < len(indices):
            first = position
            label = int(values[indices[position]])
            position += 1
            while position < len(indices) and int(values[indices[position]]) == label:
                position += 1
            if foreground_only and label <= 0:
                continue
            events.append(
                {
                    "sequence_id": str(sequence),
                    "label": label,
                    "start_s": float(dataset.start_s[indices[first]]),
                    "end_s": float(dataset.end_s[indices[position - 1]]),
                }
            )
    return events


def event_metrics(dataset: WindowDataset, predicted: np.ndarray, min_iou: float = 0.1) -> dict[str, object]:
    true_events = _runs(dataset, dataset.labels, foreground_only=True)
    predicted_events = _runs(dataset, predicted, foreground_only=True)
    candidates: list[tuple[float, int, int]] = []
    for true_index, truth in enumerate(true_events):
        for predicted_index, estimate in enumerate(predicted_events):
            if truth["sequence_id"] != estimate["sequence_id"] or truth["label"] != estimate["label"]:
                continue
            overlap = max(0.0, min(float(truth["end_s"]), float(estimate["end_s"])) - max(float(truth["start_s"]), float(estimate["start_s"])))
            union = max(float(truth["end_s"]), float(estimate["end_s"])) - min(float(truth["start_s"]), float(estimate["start_s"]))
            iou = overlap / union if union > 0.0 else 0.0
            if iou >= min_iou:
                candidates.append((iou, true_index, predicted_index))
    matched_true: set[int] = set()
    matched_predicted: set[int] = set()
    latency = []
    for _, true_index, predicted_index in sorted(candidates, reverse=True):
        if true_index in matched_true or predicted_index in matched_predicted:
            continue
        matched_true.add(true_index)
        matched_predicted.add(predicted_index)
        latency.append(float(predicted_events[predicted_index]["start_s"]) - float(true_events[true_index]["start_s"]))
    tp = len(matched_true)
    fp = len(predicted_events) - tp
    fn = len(true_events) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_duration_s = 0.0
    for sequence in dict.fromkeys(dataset.sequence_ids.tolist()):
        indices = np.flatnonzero(dataset.sequence_ids == sequence)
        total_duration_s += float(dataset.end_s[indices[-1]] - dataset.start_s[indices[0]])
    return {
        "true_events": len(true_events),
        "predicted_events": len(predicted_events),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_events_per_hour": float(fp / max(total_duration_s / 3600.0, 1e-12)),
        "matched_start_latency_mean_s": float(np.mean(latency)) if latency else None,
        "matched_start_latency_abs_mean_s": float(np.mean(np.abs(latency))) if latency else None,
    }


def evaluate_prediction(dataset: WindowDataset, prediction: ShadowPrediction) -> dict[str, object]:
    raw = classification_metrics(dataset.labels, prediction.raw_labels, dataset.class_names)
    smoothed = classification_metrics(dataset.labels, prediction.smoothed_labels, dataset.class_names)
    accepted = classification_metrics(dataset.labels, prediction.accepted_labels, dataset.class_names)
    return {
        "raw_window": raw,
        "hmm_window": smoothed,
        "accepted_window": accepted,
        "event": event_metrics(dataset, prediction.accepted_labels),
        "calibration_error": expected_calibration_error(dataset.labels, prediction.smoothed_labels, prediction.confidence),
        "mean_confidence": float(np.mean(prediction.confidence)),
        "mean_quality": float(np.mean(dataset.mean_quality)),
        "rejection_reasons": {
            str(reason): int(np.count_nonzero(prediction.rejection_reason == reason))
            for reason in sorted(set(prediction.rejection_reason.tolist()))
        },
    }


def _subset_prediction(prediction: ShadowPrediction, indices: np.ndarray) -> ShadowPrediction:
    return ShadowPrediction(
        raw_labels=prediction.raw_labels[indices],
        smoothed_labels=prediction.smoothed_labels[indices],
        accepted_labels=prediction.accepted_labels[indices],
        confidence=prediction.confidence[indices],
        probabilities=prediction.probabilities[indices],
        rejection_reason=prediction.rejection_reason[indices],
    )


def evaluate_prediction_by_dataset(
    dataset: WindowDataset,
    prediction: ShadowPrediction,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for dataset_id in sorted(set(dataset.dataset_ids.tolist())):
        indices = np.flatnonzero(dataset.dataset_ids == dataset_id)
        results[str(dataset_id)] = evaluate_prediction(
            dataset.subset(indices),
            _subset_prediction(prediction, indices),
        )
    return results


def corrupt_dataset(
    dataset: WindowDataset,
    noise_fraction: float = 0.0,
    missing_fraction: float = 0.0,
    seed: int = 0,
) -> WindowDataset:
    if not (0.0 <= noise_fraction <= 2.0 and 0.0 <= missing_fraction < 1.0):
        raise ValueError("corruption fractions are out of range")
    rng = np.random.default_rng(seed)
    windows = dataset.windows.copy()
    signal_indices = [index for index, name in enumerate(dataset.feature_names) if name != "quality"]
    quality_index = dataset.feature_names.index("quality") if "quality" in dataset.feature_names else None
    missing_by_window = dataset.missing_fraction.copy()
    if noise_fraction > 0.0:
        spread = np.std(windows[:, :, signal_indices], axis=(0, 1), keepdims=True)
        noise = rng.normal(size=(len(windows), windows.shape[1], len(signal_indices))).astype(np.float32)
        windows[:, :, signal_indices] += noise_fraction * noise * np.maximum(spread, 1e-3)
    if missing_fraction > 0.0:
        missing = rng.random((len(windows), windows.shape[1])) < missing_fraction
        for channel in signal_indices:
            windows[:, :, channel][missing] = 0.0
        if quality_index is not None:
            windows[:, :, quality_index][missing] = 0.0
        injected = np.mean(missing, axis=1)
        missing_by_window = 1.0 - (1.0 - missing_by_window) * (1.0 - injected)
    quality = np.mean(windows[:, :, quality_index], axis=1) if quality_index is not None else dataset.mean_quality.copy()
    return WindowDataset(
        windows=windows,
        labels=dataset.labels.copy(),
        subject_ids=dataset.subject_ids.copy(),
        session_ids=dataset.session_ids.copy(),
        sequence_ids=dataset.sequence_ids.copy(),
        start_s=dataset.start_s.copy(),
        end_s=dataset.end_s.copy(),
        mean_quality=quality,
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
        dataset_ids=dataset.dataset_ids.copy(),
        missing_fraction=missing_by_window,
    )


def robustness_curve(
    pipeline: ShadowActivityPipeline,
    dataset: WindowDataset,
    noise_levels: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
    missing_levels: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
    seed: int = 0,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for noise in noise_levels:
        for missing in missing_levels:
            if noise > 0.0 and missing > 0.0:
                continue
            corrupted = corrupt_dataset(dataset, noise, missing, seed + int(1000 * noise) + int(10000 * missing))
            metrics = evaluate_prediction(corrupted, pipeline.predict(corrupted))
            results.append(
                {
                    "noise_fraction_of_feature_std": float(noise),
                    "missing_sample_fraction": float(missing),
                    "macro_f1": metrics["accepted_window"]["macro_f1"],
                    "coverage": metrics["accepted_window"]["coverage"],
                    "event_f1": metrics["event"]["f1"],
                }
            )
    return results


def train_split_pipeline(
    dataset: WindowDataset,
    split: ParticipantSplit,
    ml_config: Mapping[str, object],
    epochs: int | None = None,
    seed_override: int | None = None,
) -> tuple[ShadowActivityPipeline, dict[str, object]]:
    subject_suffix = split.test_subject.rsplit("subject", 1)[-1]
    try:
        subject_seed = int(subject_suffix)
    except ValueError:
        subject_seed = sum((index + 1) * ord(character) for index, character in enumerate(split.test_subject))
    seed = (
        int(seed_override)
        if seed_override is not None
        else int(ml_config["model"].get("seed", 0)) + subject_seed
    )
    train_full = np.flatnonzero(dataset.subject_mask(split.train_subjects))
    validation_full = np.flatnonzero(dataset.subject_ids == split.validation_subject)
    train_indices = balanced_indices(dataset.labels, train_full, seed=seed)
    validation_indices = balanced_indices(dataset.labels, validation_full, seed=seed + 1)
    cnn = NumpyTemporalCNN(
        len(dataset.feature_names),
        len(dataset.class_names),
        int(ml_config["model"]["filters"]),
        int(ml_config["model"]["kernel_size"]),
        seed,
        pooling=str(ml_config["model"].get("pooling", "mean")),
    )
    started = perf_counter()
    history = cnn.fit(
        dataset.windows[train_indices],
        dataset.labels[train_indices],
        dataset.windows[validation_indices],
        dataset.labels[validation_indices],
        dataset.feature_names,
        ml_config["model"],
        ml_config["augmentation"],
        epochs,
    )
    cnn.calibrate_temperature(dataset.windows[validation_full], dataset.labels[validation_full])
    hmm_config = ml_config["hmm"]
    hmm = TemporalHMM.fit(
        label_sequences(dataset, train_full),
        len(dataset.class_names),
        float(hmm_config.get("laplace", 0.5)),
        float(hmm_config.get("self_transition_prior", 20.0)),
        float(hmm_config.get("background_transition_prior", 2.0)),
    )
    pipeline = ShadowActivityPipeline(
        cnn,
        hmm,
        dataset.class_names,
        dataset.feature_names,
        float(ml_config["model"]["confidence_threshold"]),
        float(ml_config["window"]["min_window_quality"]),
        {
            "task": ml_config.get("task"),
            "training_subjects": list(split.train_subjects),
            "validation_subject": split.validation_subject,
            "test_subject": split.test_subject,
            "training_dataset_ids": sorted(set(dataset.dataset_ids[train_full].tolist())),
        },
        float(ml_config["window"].get("max_missing_fraction", 1.0)),
    )
    details = {
        "train_windows_full": int(len(train_full)),
        "train_windows_balanced": int(len(train_indices)),
        "validation_windows_full": int(len(validation_full)),
        "validation_windows_balanced": int(len(validation_indices)),
        "best_epoch": int(cnn.best_epoch),
        "epochs_completed": len(history),
        "temperature": float(cnn.temperature),
        "training_seconds": perf_counter() - started,
        "history": history,
    }
    return pipeline, details