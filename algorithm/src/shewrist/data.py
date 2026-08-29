"""Data loading, synchronization, and CSV helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


SENSOR_FILES = {
    "accel": "Accelerometer.txt",
    "gyro": "Gyroscope.txt",
    "mag": "Magnetometer.txt",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sensor_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(Path(path), delimiter=",")
    if data.ndim != 2 or data.shape[1] != 4 or len(data) < 2:
        raise ValueError(f"expected four-column sensor file: {path}")
    if not np.all(np.isfinite(data)) or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"invalid or non-monotonic sensor data: {path}")
    return data[:, 0], data[:, 1:]


def resample_xyz(time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target_time, dtype=float)
    if values.shape != (len(time), 3):
        raise ValueError("values must have shape (n, 3)")
    return np.column_stack([np.interp(target, time, values[:, axis]) for axis in range(3)])


def load_public_trial(
    dataset_root: str | Path,
    subject: str,
    set_name: str = "set2",
    sample_rate_hz: float = 100.0,
) -> dict:
    root = Path(dataset_root) / subject / set_name
    raw: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for segment in ("forearm", "hand"):
        raw[segment] = {}
        for modality, filename in SENSOR_FILES.items():
            raw[segment][modality] = load_sensor_file(root / segment / filename)
    starts = [series[0][0] for segment in raw.values() for series in segment.values()]
    ends = [series[0][-1] for segment in raw.values() for series in segment.values()]
    start, end = max(starts), min(ends)
    if end <= start:
        raise ValueError(f"no common time range for {subject}/{set_name}")
    count = int(np.floor((end - start) * sample_rate_hz)) + 1
    timeline = start + np.arange(count) / sample_rate_hz
    result: dict[str, object] = {"timestamp_s": timeline, "subject": subject, "set": set_name}
    for segment, modalities in raw.items():
        for modality, (time, values) in modalities.items():
            result[f"{segment}_{modality}"] = resample_xyz(time, values, timeline)
    return result


def load_annotations(path: str | Path, subject: str | None = None, set_name: str | None = None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if subject is not None and row["Subject"] != subject:
                continue
            if set_name is not None and row["Set"] != set_name:
                continue
            row["Init"] = float(row["Init"])
            row["End"] = float(row["End"])
            rows.append(row)
    return rows


def find_neutral_interval(annotations: Iterable[Mapping[str, object]]) -> tuple[float, float]:
    candidates = [
        row
        for row in annotations
        if str(row.get("Category")) == "Relative"
        and str(row.get("Segment")) == "wrist"
        and str(row.get("Type")) == "AnatomicalPos"
    ]
    if not candidates:
        raise ValueError("no relative wrist AnatomicalPos interval found")
    selected = max(candidates, key=lambda row: float(row["End"]) - float(row["Init"]))
    return float(selected["Init"]), float(selected["End"])


def load_joint_state_csv(path: str | Path) -> dict[str, np.ndarray]:
    required = {"timestamp_ms", "theta_FE", "theta_RUD"}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("joint state CSV is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    numeric = {"timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "quality"}
    output: dict[str, np.ndarray] = {}
    for key in rows[0]:
        if key in numeric:
            output[key] = np.array([float(row[key]) if row.get(key, "") not in {"", None} else np.nan for row in rows])
        else:
            output[key] = np.array([row.get(key, "") for row in rows], dtype=object)
    order = np.argsort(output["timestamp_ms"])
    output = {key: value[order] for key, value in output.items()}
    if np.any(np.diff(output["timestamp_ms"]) <= 0.0):
        raise ValueError("timestamp_ms must be unique and strictly increasing")
    return output


def load_mechanical_csv(path: str | Path) -> dict[str, np.ndarray]:
    numeric = {
        "timestamp_ms",
        "device_ms",
        "dial_level",
        "support_level",
        "cable_tension_N",
        "p_radial_kPa",
        "p_dorsal_kPa",
        "p_ulnar_kPa",
        "fsr_raw",
        "fsr_raw_adc",
        "fsr_normalized_pct",
        "discomfort",
        "discomfort_nrs",
        "safety_symptom_flag",
        "user_continues",
    }
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("mechanical CSV is empty")
    timestamp_key = "timestamp_ms" if "timestamp_ms" in rows[0] else "device_ms" if "device_ms" in rows[0] else None
    if timestamp_key is None:
        raise ValueError("mechanical CSV must include timestamp_ms or device_ms")
    if "fsr_raw" in rows[0] and "fsr_raw_adc" in rows[0]:
        raise ValueError("use only one of fsr_raw or fsr_raw_adc")
    output = {
        key: np.array([float(row[key]) if row.get(key, "") not in {"", None} else np.nan for row in rows])
        for key in rows[0]
        if key in numeric
    }
    if timestamp_key == "device_ms":
        device_time = output.pop("device_ms")
        output["timestamp_ms"] = device_time - device_time[0]
    if "fsr_raw" in output:
        output["fsr_raw_adc"] = output.pop("fsr_raw")
    for key in ("discomfort", "safety_symptom_flag", "user_continues"):
        if key in output:
            finite = output[key][np.isfinite(output[key])]
            if len(finite) != len(output[key]) or np.any(~np.isin(finite, [0.0, 1.0])):
                raise ValueError(f"{key} must contain only 0 or 1")
    if "discomfort_nrs" in output:
        finite = output["discomfort_nrs"][np.isfinite(output["discomfort_nrs"])]
        if np.any((finite < 0.0) | (finite > 10.0)):
            raise ValueError("discomfort_nrs must be within 0..10")
    if "fsr_raw_adc" in output:
        finite = output["fsr_raw_adc"][np.isfinite(output["fsr_raw_adc"])]
        if np.any(finite < 0.0):
            raise ValueError("fsr_raw_adc must be non-negative")
    if "fsr_normalized_pct" in output:
        finite = output["fsr_normalized_pct"][np.isfinite(output["fsr_normalized_pct"])]
        if np.any((finite < 0.0) | (finite > 100.0)):
            raise ValueError("fsr_normalized_pct must be within 0..100")
    return output


def align_mechanical_to_joint(joint_timestamp_ms: np.ndarray, mechanical: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    source_time = np.asarray(mechanical["timestamp_ms"], dtype=float)
    target_time = np.asarray(joint_timestamp_ms, dtype=float)
    aligned = {"timestamp_ms": target_time.copy()}
    for key, values in mechanical.items():
        if key == "timestamp_ms":
            continue
        source_values = np.asarray(values, dtype=float)
        if key in {"discomfort", "safety_symptom_flag", "user_continues", "support_level", "dial_level", "discomfort_nrs"}:
            indices = np.searchsorted(source_time, target_time, side="right") - 1
            valid = (indices >= 0) & (target_time <= source_time[-1])
            output = np.full(len(target_time), np.nan, dtype=float)
            output[valid] = source_values[indices[valid]]
            aligned[key] = output
        else:
            aligned[key] = np.interp(target_time, source_time, source_values, left=np.nan, right=np.nan)
    return aligned


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")