"""Neutral-pose and functional-axis calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .quaternion import average, conjugate, multiply, normalize, to_rotation_vector


@dataclass(frozen=True)
class FunctionalAxes:
    flexion_extension: np.ndarray
    radial_ulnar: np.ndarray
    pronation_supination: np.ndarray
    convention: str = "FE positive extension; RUD positive ulnar deviation"


def interval_mask(timestamp_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    t = np.asarray(timestamp_s, dtype=float)
    return (t >= float(start_s)) & (t <= float(end_s))


def estimate_gyro_bias(timestamp_s: np.ndarray, gyro: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    mask = interval_mask(timestamp_s, start_s, end_s)
    if np.count_nonzero(mask) < 3:
        raise ValueError("neutral interval must contain at least three gyro samples")
    return np.median(np.asarray(gyro, dtype=float)[mask], axis=0)


def relative_quaternion(forearm_q: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
    forearm_q = normalize(forearm_q)
    hand_q = normalize(hand_q)
    if forearm_q.shape != hand_q.shape:
        raise ValueError("forearm and hand quaternion arrays must have matching shapes")
    return normalize(multiply(conjugate(forearm_q), hand_q))


def neutral_zero(relative_q: np.ndarray, neutral_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relative_q = normalize(relative_q)
    mask = np.asarray(neutral_mask, dtype=bool)
    if len(mask) != len(relative_q) or np.count_nonzero(mask) < 3:
        raise ValueError("neutral mask must select at least three quaternion samples")
    neutral_q = average(relative_q[mask])
    zeroed = normalize(multiply(conjugate(neutral_q), relative_q))
    return zeroed, neutral_q


def _principal_axis(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) < 10:
        raise ValueError("functional calibration needs at least ten valid samples")
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] < 1e-9:
        raise ValueError("functional calibration has insufficient angular variation")
    axis = vh[0]
    return axis / np.linalg.norm(axis)


def _collect_intervals(
    vectors: np.ndarray,
    timestamp_s: np.ndarray,
    annotations: Iterable[Mapping[str, object]],
    labels: set[str],
    max_interval_s: float,
) -> np.ndarray:
    chunks = []
    for row in annotations:
        label = str(row.get("Type", ""))
        start = float(row.get("Init", 0.0))
        end = float(row.get("End", 0.0))
        if label in labels and 0.0 < end - start <= max_interval_s:
            mask = interval_mask(timestamp_s, start, end)
            if np.count_nonzero(mask) > 0:
                chunks.append(vectors[mask])
    if not chunks:
        raise ValueError(f"no usable annotation intervals for {sorted(labels)}")
    return np.concatenate(chunks, axis=0)


def _mean_projection(
    vectors: np.ndarray,
    timestamp_s: np.ndarray,
    annotations: Iterable[Mapping[str, object]],
    label: str,
    axis: np.ndarray,
    max_interval_s: float,
) -> float:
    data = _collect_intervals(vectors, timestamp_s, annotations, {label}, max_interval_s)
    return float(np.nanmean(np.sum(data * axis, axis=1)))


def _mean_interval_vector(
    vectors: np.ndarray,
    timestamp_s: np.ndarray,
    annotations: Iterable[Mapping[str, object]],
    label: str,
    max_interval_s: float,
) -> np.ndarray | None:
    matching = [row for row in annotations if str(row.get("Type", "")) == label]
    if not matching:
        return None
    samples = _collect_intervals(vectors, timestamp_s, matching, {label}, max_interval_s)
    finite = samples[np.all(np.isfinite(samples), axis=1)]
    if not len(finite):
        return None
    return np.median(finite, axis=0)


def _signed_axis(
    vectors: np.ndarray,
    timestamp_s: np.ndarray,
    annotations: list[Mapping[str, object]],
    positive_label: str,
    negative_label: str,
    max_interval_s: float,
) -> np.ndarray:
    positive = _mean_interval_vector(vectors, timestamp_s, annotations, positive_label, max_interval_s)
    negative = _mean_interval_vector(vectors, timestamp_s, annotations, negative_label, max_interval_s)
    if positive is None and negative is None:
        raise ValueError(f"functional calibration needs {positive_label} or {negative_label}")
    if positive is not None and negative is not None:
        samples = _collect_intervals(
            vectors,
            timestamp_s,
            annotations,
            {positive_label, negative_label},
            max_interval_s,
        )
        direction = _principal_axis(samples)
        if _mean_projection(vectors, timestamp_s, annotations, positive_label, direction, max_interval_s) < _mean_projection(
            vectors, timestamp_s, annotations, negative_label, direction, max_interval_s
        ):
            direction *= -1.0
        return direction
    direction = positive if positive is not None else -negative
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise ValueError(f"functional calibration has insufficient {positive_label}/{negative_label} separation")
    return direction / norm


def estimate_functional_axes(
    zeroed_relative_q: np.ndarray,
    timestamp_s: np.ndarray,
    annotations: Iterable[Mapping[str, object]],
    max_interval_s: float = 15.0,
) -> FunctionalAxes:
    annotations = list(annotations)
    vectors = to_rotation_vector(zeroed_relative_q)
    fe_axis = _signed_axis(vectors, timestamp_s, annotations, "Extension", "Flexion", max_interval_s)
    rud_axis = _signed_axis(vectors, timestamp_s, annotations, "Ulnar Deviation", "Radial Deviation", max_interval_s)
    rud_axis = rud_axis - np.dot(rud_axis, fe_axis) * fe_axis
    rud_norm = np.linalg.norm(rud_axis)
    if rud_norm < 1e-6:
        raise ValueError("functional flexion and deviation axes are nearly collinear")
    rud_axis /= rud_norm
    ps_axis = np.cross(fe_axis, rud_axis)
    ps_axis /= np.linalg.norm(ps_axis)
    rud_axis = np.cross(ps_axis, fe_axis)
    rud_axis /= np.linalg.norm(rud_axis)
    return FunctionalAxes(fe_axis, rud_axis, ps_axis)


def project_angles(zeroed_relative_q: np.ndarray, axes: FunctionalAxes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors_deg = np.degrees(to_rotation_vector(zeroed_relative_q))
    fe = np.sum(vectors_deg * axes.flexion_extension, axis=1)
    rud = np.sum(vectors_deg * axes.radial_ulnar, axis=1)
    ps = np.sum(vectors_deg * axes.pronation_supination, axis=1)
    return fe, rud, ps