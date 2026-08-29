"""Windowed datasets for calibration-assisted wrist-motion classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .calibration import interval_mask
from .data import find_neutral_interval, load_annotations, load_public_trial
from .kinematics import compute_wrist_kinematics
from .quality import sample_quality


CLASS_NAMES = (
    "background",
    "extension",
    "flexion",
    "radial_deviation",
    "ulnar_deviation",
)
LABEL_BY_ANNOTATION = {
    "Extension": 1,
    "Flexion": 2,
    "Radial Deviation": 3,
    "Ulnar Deviation": 4,
}
FEATURE_NAMES = (
    "theta_fe_deg",
    "theta_rud_deg",
    "dtheta_fe_deg_s",
    "dtheta_rud_deg_s",
    "angular_velocity_deg_s",
    "quality",
)


@dataclass(frozen=True)
class WindowDataset:
    windows: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    sequence_ids: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray
    mean_quality: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    class_names: tuple[str, ...] = CLASS_NAMES
    dataset_ids: np.ndarray | None = None
    missing_fraction: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.windows)
        n = len(x)
        if x.ndim != 3 or x.shape[2] != len(self.feature_names):
            raise ValueError("windows must have shape (n, time, features)")
        dataset_ids = (
            np.full(n, "unspecified", dtype=object)
            if self.dataset_ids is None
            else np.asarray(self.dataset_ids, dtype=object)
        )
        missing_fraction = (
            np.zeros(n, dtype=float)
            if self.missing_fraction is None
            else np.asarray(self.missing_fraction, dtype=float)
        )
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "missing_fraction", missing_fraction)
        arrays = (
            self.labels,
            self.subject_ids,
            self.session_ids,
            self.sequence_ids,
            self.start_s,
            self.end_s,
            self.mean_quality,
            dataset_ids,
            missing_fraction,
        )
        if any(len(np.asarray(values)) != n for values in arrays):
            raise ValueError("all window metadata arrays must have equal length")
        if np.any(~np.isfinite(missing_fraction)) or np.any((missing_fraction < 0.0) | (missing_fraction > 1.0)):
            raise ValueError("missing_fraction must be finite and within 0..1")

    def __len__(self) -> int:
        return int(len(self.windows))

    def subset(self, indices: np.ndarray | Sequence[int]) -> "WindowDataset":
        index = np.asarray(indices)
        return WindowDataset(
            windows=self.windows[index],
            labels=self.labels[index],
            subject_ids=self.subject_ids[index],
            session_ids=self.session_ids[index],
            sequence_ids=self.sequence_ids[index],
            start_s=self.start_s[index],
            end_s=self.end_s[index],
            mean_quality=self.mean_quality[index],
            feature_names=self.feature_names,
            class_names=self.class_names,
            dataset_ids=self.dataset_ids[index],
            missing_fraction=self.missing_fraction[index],
        )

    def subject_mask(self, subjects: Sequence[str]) -> np.ndarray:
        return np.isin(self.subject_ids, np.asarray(subjects, dtype=object))

    def label_counts(self) -> dict[str, int]:
        return {
            name: int(np.count_nonzero(self.labels == index))
            for index, name in enumerate(self.class_names)
        }

    def dataset_counts(self) -> dict[str, int]:
        return {
            str(dataset_id): int(np.count_nonzero(self.dataset_ids == dataset_id))
            for dataset_id in sorted(set(self.dataset_ids.tolist()))
        }


def split_functional_repeats(
    rows: Sequence[Mapping[str, object]],
    source_labels: Sequence[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    labels = tuple(LABEL_BY_ANNOTATION) if source_labels is None else tuple(str(value) for value in source_labels)
    grouped: dict[str, list[dict]] = {label: [] for label in labels}
    for source in rows:
        row = dict(source)
        label = str(row.get("Type", ""))
        duration = float(row.get("End", 0.0)) - float(row.get("Init", 0.0))
        if row.get("Segment") == "wrist" and label in grouped and 0.0 < duration <= 15.0:
            grouped[label].append(row)
    calibration: list[dict] = []
    validation: list[dict] = []
    for label in labels:
        repeats = sorted(grouped[label], key=lambda row: float(row["Init"]))
        if not repeats:
            raise ValueError(f"missing functional interval: {label}")
        calibration.append(repeats[0])
        validation.extend(repeats[1:])
    return calibration, validation


def _feature_matrix(timestamp_s: np.ndarray, theta_fe: np.ndarray, theta_rud: np.ndarray, quality: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamp_s, dtype=float)
    fe = np.asarray(theta_fe, dtype=float)
    rud = np.asarray(theta_rud, dtype=float)
    q = np.asarray(quality, dtype=float)
    dfe = np.gradient(fe, t)
    drud = np.gradient(rud, t)
    speed = np.sqrt(dfe * dfe + drud * drud)
    features = np.column_stack((fe, rud, dfe, drud, speed, q))
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _assign_sample_labels(
    timestamp_s: np.ndarray,
    validation_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    label_by_annotation: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mapping = LABEL_BY_ANNOTATION if label_by_annotation is None else label_by_annotation
    labels = np.zeros(len(timestamp_s), dtype=int)
    excluded = np.zeros(len(timestamp_s), dtype=bool)
    for row in validation_rows:
        label = int(mapping[str(row["Type"])])
        labels[interval_mask(timestamp_s, float(row["Init"]), float(row["End"]))] = label
    for row in calibration_rows:
        excluded |= interval_mask(timestamp_s, float(row["Init"]), float(row["End"]))
    return labels, excluded


def extract_windows(
    timestamp_s: np.ndarray,
    features: np.ndarray,
    sample_labels: np.ndarray | None,
    excluded: np.ndarray | None,
    subject_id: str,
    session_id: str,
    window_config: Mapping[str, object],
    dataset_id: str = "unspecified",
    sample_missing: np.ndarray | None = None,
) -> WindowDataset:
    t = np.asarray(timestamp_s, dtype=float)
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or len(x) != len(t):
        raise ValueError("features must have shape (samples, features)")
    labels = np.full(len(t), -1, dtype=int) if sample_labels is None else np.asarray(sample_labels, dtype=int)
    excluded_values = np.zeros(len(t), dtype=bool) if excluded is None else np.asarray(excluded, dtype=bool)
    missing_values = np.zeros(len(t), dtype=bool) if sample_missing is None else np.asarray(sample_missing, dtype=bool)
    if len(labels) != len(t) or len(excluded_values) != len(t) or len(missing_values) != len(t):
        raise ValueError("sample labels and masks must match timestamps")
    rate = float(window_config["sample_rate_hz"])
    window_samples = int(round(float(window_config["window_seconds"]) * rate))
    step_samples = int(round(float(window_config["step_seconds"]) * rate))
    label_strategy = str(window_config.get("label_strategy", "center_sample"))
    if label_strategy != "center_sample":
        raise ValueError(f"unsupported label strategy: {label_strategy}")
    if window_samples < 2 or step_samples < 1:
        raise ValueError("invalid window or step length")
    output_x: list[np.ndarray] = []
    output_y: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    quality: list[float] = []
    missing_fractions: list[float] = []
    sequences: list[str] = []
    sequence_number = -1
    last_start_index: int | None = None
    quality_index = FEATURE_NAMES.index("quality")
    for start in range(0, len(t) - window_samples + 1, step_samples):
        stop = start + window_samples
        if np.any(excluded_values[start:stop]):
            continue
        window_labels = labels[start:stop]
        if sample_labels is None:
            label = -1
        else:
            label = int(window_labels[window_samples // 2])
        if last_start_index is None or start - last_start_index > int(round(1.5 * step_samples)):
            sequence_number += 1
        last_start_index = start
        output_x.append(x[start:stop])
        output_y.append(label)
        starts.append(float(t[start]))
        ends.append(float(t[stop - 1] + 1.0 / rate))
        quality.append(float(np.mean(x[start:stop, quality_index])))
        missing_fractions.append(float(np.mean(missing_values[start:stop])))
        sequences.append(f"{session_id}-seq{sequence_number:03d}")
    if not output_x:
        raise ValueError(f"no valid windows extracted for {session_id}")
    count = len(output_x)
    return WindowDataset(
        windows=np.stack(output_x).astype(np.float32),
        labels=np.asarray(output_y, dtype=int),
        subject_ids=np.full(count, subject_id, dtype=object),
        session_ids=np.full(count, session_id, dtype=object),
        sequence_ids=np.asarray(sequences, dtype=object),
        start_s=np.asarray(starts, dtype=float),
        end_s=np.asarray(ends, dtype=float),
        mean_quality=np.asarray(quality, dtype=float),
        dataset_ids=np.full(count, str(dataset_id), dtype=object),
        missing_fraction=np.asarray(missing_fractions, dtype=float),
    )


def concatenate_datasets(datasets: Sequence[WindowDataset]) -> WindowDataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    feature_names = datasets[0].feature_names
    class_names = datasets[0].class_names
    if any(item.feature_names != feature_names or item.class_names != class_names for item in datasets):
        raise ValueError("dataset schemas do not match")
    return WindowDataset(
        windows=np.concatenate([item.windows for item in datasets], axis=0),
        labels=np.concatenate([item.labels for item in datasets]),
        subject_ids=np.concatenate([item.subject_ids for item in datasets]),
        session_ids=np.concatenate([item.session_ids for item in datasets]),
        sequence_ids=np.concatenate([item.sequence_ids for item in datasets]),
        start_s=np.concatenate([item.start_s for item in datasets]),
        end_s=np.concatenate([item.end_s for item in datasets]),
        mean_quality=np.concatenate([item.mean_quality for item in datasets]),
        feature_names=feature_names,
        class_names=class_names,
        dataset_ids=np.concatenate([item.dataset_ids for item in datasets]),
        missing_fraction=np.concatenate([item.missing_fraction for item in datasets]),
    )


def build_public_activity_dataset(
    dataset_root: str | Path,
    algorithm_config: Mapping[str, object],
    ml_config: Mapping[str, object],
    subjects: Sequence[str] | None = None,
    dataset_id: str = "upper_body_movements",
    source_label_map: Mapping[str, str] | None = None,
) -> WindowDataset:
    """Build calibration-assisted windows from held-out repetitions.

    The first repetition of each movement is used only for per-subject functional
    axis calibration and is excluded from classifier windows.  Later repetitions
    provide the movement labels.  All model evaluation must still split by
    participant. ``subjects`` can isolate a locked test participant on disk.
    """
    root = Path(dataset_root)
    annotations_path = root / "annotations.csv"
    available = sorted(path.name for path in root.glob("subject*") if path.is_dir())
    selected = available if subjects is None else [str(subject) for subject in subjects]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"subjects are absent from {root}: {missing}")
    subjects = selected
    if not subjects:
        raise ValueError(f"no subject directories under {root}")
    rate = float(ml_config["window"]["sample_rate_hz"])
    configured_label_map = (
        {source: CLASS_NAMES[index] for source, index in LABEL_BY_ANNOTATION.items()}
        if source_label_map is None
        else {str(source): str(target) for source, target in source_label_map.items()}
    )
    movement_label_map = {
        source: CLASS_NAMES.index(target)
        for source, target in configured_label_map.items()
        if target != "background"
    }
    if not movement_label_map:
        raise ValueError("source_label_map must include at least one non-background activity")
    datasets: list[WindowDataset] = []
    for subject in subjects:
        rows = load_annotations(annotations_path, subject, "set2")
        calibration_rows, validation_rows = split_functional_repeats(rows, tuple(movement_label_map))
        trial = load_public_trial(root, subject, "set2", rate)
        timestamp = np.asarray(trial["timestamp_s"], dtype=float)
        forearm_quality, _ = sample_quality(
            timestamp,
            np.asarray(trial["forearm_accel"]),
            np.asarray(trial["forearm_gyro"]),
            None,
            algorithm_config["quality"],
        )
        hand_quality, _ = sample_quality(
            timestamp,
            np.asarray(trial["hand_accel"]),
            np.asarray(trial["hand_gyro"]),
            None,
            algorithm_config["quality"],
        )
        result = compute_wrist_kinematics(
            timestamp,
            np.asarray(trial["forearm_accel"]),
            np.asarray(trial["forearm_gyro"]),
            np.asarray(trial["hand_accel"]),
            np.asarray(trial["hand_gyro"]),
            find_neutral_interval(rows),
            calibration_rows,
            algorithm=str(algorithm_config["fusion"].get("default_algorithm", "madgwick")),
            fusion_config=algorithm_config["fusion"],
            quality_config=algorithm_config["quality"],
        )
        quality = np.minimum.reduce((forearm_quality, hand_quality, result.quality))
        features = _feature_matrix(timestamp, result.theta_fe_deg, result.theta_rud_deg, quality)
        sample_labels, excluded = _assign_sample_labels(
            timestamp,
            validation_rows,
            calibration_rows,
            movement_label_map,
        )
        datasets.append(
            extract_windows(
                timestamp,
                features,
                sample_labels,
                excluded,
                subject,
                f"{subject}-set2",
                ml_config["window"],
                dataset_id=dataset_id,
            )
        )
    return concatenate_datasets(datasets)


def build_joint_state_windows(
    joint_state: Mapping[str, np.ndarray],
    ml_config: Mapping[str, object],
    session_id: str,
) -> WindowDataset:
    source_t = np.asarray(joint_state["timestamp_ms"], dtype=float) / 1000.0
    rate = float(ml_config["window"]["sample_rate_hz"])
    target_t = np.arange(source_t[0], source_t[-1] + 0.5 / rate, 1.0 / rate)
    source_fe = np.asarray(joint_state["theta_FE"], dtype=float)
    source_rud = np.asarray(joint_state["theta_RUD"], dtype=float)
    source_quality = (
        np.asarray(joint_state["quality"], dtype=float)
        if "quality" in joint_state
        else np.ones(len(source_t), dtype=float)
    )
    source_missing = ~(np.isfinite(source_fe) & np.isfinite(source_rud) & np.isfinite(source_quality))
    fe = np.interp(target_t, source_t, source_fe)
    rud = np.interp(target_t, source_t, source_rud)
    quality = np.interp(target_t, source_t, source_quality)
    target_missing = np.interp(
        target_t,
        source_t,
        source_missing.astype(float),
        left=1.0,
        right=1.0,
    ) > 0.0
    features = _feature_matrix(target_t, fe, rud, quality)
    return extract_windows(
        target_t,
        features,
        None,
        None,
        session_id,
        session_id,
        ml_config["window"],
        dataset_id="runtime_joint_state",
        sample_missing=target_missing,
    )