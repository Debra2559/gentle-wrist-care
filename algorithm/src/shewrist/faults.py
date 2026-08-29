"""Deterministic fault injection for dual-IMU offline replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


FAULT_KINDS = {
    "dropout",
    "out_of_order",
    "timestamp_offset",
    "silence",
    "saturation",
    "gyro_bias",
    "mounting_rotation",
    "slip",
}


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    target: str = "hand"
    start_fraction: float = 0.60
    duration_fraction: float = 0.10
    magnitude: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unsupported fault kind: {self.kind}")
        if self.target not in {"forearm", "hand", "both"}:
            raise ValueError("fault target must be forearm, hand, or both")
        if not 0.0 <= self.start_fraction <= 1.0:
            raise ValueError("start_fraction must be in 0..1")
        if not 0.0 <= self.duration_fraction <= 1.0:
            raise ValueError("duration_fraction must be in 0..1")


def _targets(target: str) -> tuple[str, ...]:
    return ("forearm", "hand") if target == "both" else (target,)


def _span(length: int, spec: FaultSpec) -> tuple[int, int]:
    start = min(length - 1, max(0, int(round(spec.start_fraction * max(0, length - 1)))))
    count = max(1, int(round(spec.duration_fraction * length)))
    return start, min(length, start + count)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.radians(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _rotate_rows(values: np.ndarray, degrees: np.ndarray | float) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    output = data.copy()
    angles = np.full(len(data), float(degrees)) if np.isscalar(degrees) else np.asarray(degrees, dtype=float)
    if len(angles) != len(data):
        raise ValueError("rotation angles must match sensor rows")
    radians = np.radians(angles)
    cosine, sine = np.cos(radians), np.sin(radians)
    output[:, 0] = cosine * data[:, 0] - sine * data[:, 1]
    output[:, 1] = sine * data[:, 0] + cosine * data[:, 1]
    output[:, 2] = data[:, 2]
    return output


def inject_faults(
    raw: Mapping[str, np.ndarray],
    specs: Sequence[FaultSpec],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]]]:
    """Return corrupted streams, known-invalid masks, and an audit log.

    Expected raw keys are ``<node>_timestamp_s``, ``<node>_accel`` and
    ``<node>_gyro`` for forearm and hand. Optional magnetometer arrays are
    transformed with mounting faults but are otherwise untouched.
    """
    output = {key: np.asarray(value).copy() for key, value in raw.items()}
    for node in ("forearm", "hand"):
        required = (f"{node}_timestamp_s", f"{node}_accel", f"{node}_gyro")
        if any(key not in output for key in required):
            raise ValueError(f"missing raw {node} stream")
        length = len(output[required[0]])
        if output[required[1]].shape != (length, 3) or output[required[2]].shape != (length, 3):
            raise ValueError(f"invalid raw {node} shape")
    invalid = {
        "forearm": np.zeros(len(output["forearm_timestamp_s"]), dtype=bool),
        "hand": np.zeros(len(output["hand_timestamp_s"]), dtype=bool),
    }
    audit: list[dict[str, object]] = []
    for spec in specs:
        record = asdict(spec)
        record["affected_samples"] = {}
        record["expected_response"] = {
            "out_of_order": "input_rejected",
            "timestamp_offset": "synchronization_quality_rejected",
            "mounting_rotation": "functional_calibration_compensation_or_explicit_failure",
        }.get(spec.kind, "affected_samples_rejected_or_degraded")
        for node in _targets(spec.target):
            timestamp_key = f"{node}_timestamp_s"
            accel_key = f"{node}_accel"
            gyro_key = f"{node}_gyro"
            mag_key = f"{node}_mag"
            length = len(output[timestamp_key])
            start, stop = _span(length, spec)
            if spec.kind == "dropout":
                keep = np.ones(length, dtype=bool)
                keep[start:stop] = False
                for key in (timestamp_key, accel_key, gyro_key, mag_key):
                    if key in output:
                        output[key] = output[key][keep]
                invalid[node] = invalid[node][keep]
            elif spec.kind == "out_of_order":
                second = min(length - 1, start + 1)
                output[timestamp_key][start], output[timestamp_key][second] = (
                    output[timestamp_key][second],
                    output[timestamp_key][start],
                )
                invalid[node][start : second + 1] = True
            elif spec.kind == "timestamp_offset":
                offset_s = float(spec.magnitude if spec.magnitude else 50.0) / 1000.0
                output[timestamp_key] = output[timestamp_key] + offset_s
                invalid[node][:] = True
            elif spec.kind == "silence":
                anchor = max(0, start - 1)
                output[accel_key][start:stop] = output[accel_key][anchor]
                output[gyro_key][start:stop] = output[gyro_key][anchor]
                if mag_key in output:
                    output[mag_key][start:stop] = output[mag_key][anchor]
                invalid[node][start:stop] = True
            elif spec.kind == "saturation":
                multiplier = abs(float(spec.magnitude)) if spec.magnitude else 1.0
                output[accel_key][start:stop] = 200.0 * multiplier
                output[gyro_key][start:stop] = 50.0 * multiplier
                invalid[node][start:stop] = True
            elif spec.kind == "gyro_bias":
                magnitude = float(spec.magnitude if spec.magnitude else 0.35)
                output[gyro_key][start:stop] += np.array([magnitude, -0.5 * magnitude, 0.25 * magnitude])
                invalid[node][start:stop] = True
            elif spec.kind == "mounting_rotation":
                degrees = float(spec.magnitude if spec.magnitude else 20.0)
                for key in (accel_key, gyro_key, mag_key):
                    if key in output:
                        output[key] = _rotate_rows(output[key], degrees)
            elif spec.kind == "slip":
                degrees = float(spec.magnitude if spec.magnitude else 20.0)
                angles = np.linspace(0.0, degrees, max(1, stop - start))
                for key in (accel_key, gyro_key, mag_key):
                    if key in output:
                        output[key][start:stop] = _rotate_rows(output[key][start:stop], angles)
                invalid[node][start:stop] = True
            record["affected_samples"][node] = int(stop - start)
        audit.append(record)
    return output, invalid, audit


def default_fault_suite() -> dict[str, list[FaultSpec]]:
    return {
        "baseline": [],
        "dropout": [FaultSpec("dropout", duration_fraction=0.03)],
        "out_of_order": [FaultSpec("out_of_order", duration_fraction=0.0)],
        "timestamp_offset_50ms": [FaultSpec("timestamp_offset", magnitude=50.0)],
        "silence": [FaultSpec("silence", duration_fraction=0.08)],
        "saturation": [FaultSpec("saturation", duration_fraction=0.02)],
        "gyro_bias": [FaultSpec("gyro_bias", duration_fraction=0.15, magnitude=0.35)],
        "mounting_rotation_20deg": [FaultSpec("mounting_rotation", start_fraction=0.0, duration_fraction=1.0, magnitude=20.0)],
        "slip_20deg": [FaultSpec("slip", duration_fraction=0.20, magnitude=20.0)],
    }
