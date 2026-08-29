"""Signal quality, packet-gap, synchronization, and stationarity checks."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def timestamp_quality(timestamp_s: np.ndarray, gap_factor: float = 1.5) -> dict[str, float | int]:
    t = np.asarray(timestamp_s, dtype=float)
    if len(t) < 2 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    dt = np.diff(t)
    nominal = float(np.median(dt))
    gaps = dt > gap_factor * nominal
    estimated_missing = np.maximum(np.rint(dt[gaps] / nominal).astype(int) - 1, 0)
    return {
        "sample_count": int(len(t)),
        "nominal_rate_hz": 1.0 / nominal,
        "median_interval_ms": 1000.0 * nominal,
        "max_interval_ms": 1000.0 * float(np.max(dt)),
        "gap_count": int(np.count_nonzero(gaps)),
        "estimated_missing_samples": int(np.sum(estimated_missing)),
    }


def nearest_sync_error_ms(reference_s: np.ndarray, other_s: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference_s, dtype=float)
    other = np.asarray(other_s, dtype=float)
    if len(reference) == 0 or len(other) == 0:
        raise ValueError("both timestamp arrays must be non-empty")
    indices = np.searchsorted(other, reference)
    right = np.clip(indices, 0, len(other) - 1)
    left = np.clip(indices - 1, 0, len(other) - 1)
    errors = np.minimum(np.abs(reference - other[left]), np.abs(reference - other[right])) * 1000.0
    return {
        "median_sync_error_ms": float(np.median(errors)),
        "p95_sync_error_ms": float(np.percentile(errors, 95)),
        "max_sync_error_ms": float(np.max(errors)),
    }


def detect_stationary(
    accel_mps2: np.ndarray,
    gyro_rad_s: np.ndarray,
    gravity: float = 9.80665,
    accel_tolerance: float = 0.35,
    gyro_threshold_rad_s: float = 0.08,
) -> np.ndarray:
    accel = np.asarray(accel_mps2, dtype=float)
    gyro = np.asarray(gyro_rad_s, dtype=float)
    if accel.shape != gyro.shape or accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError("accel and gyro must have matching shape (n, 3)")
    return (np.abs(np.linalg.norm(accel, axis=1) - gravity) <= accel_tolerance) & (
        np.linalg.norm(gyro, axis=1) <= gyro_threshold_rad_s
    )


def sensor_fault_quality(
    accel_mps2: np.ndarray,
    gyro_rad_s: np.ndarray,
    accel_saturation_mps2: float = 100.0,
    gyro_saturation_rad_s: float = 20.0,
    silence_min_samples: int = 10,
    silence_tolerance: float = 1e-12,
) -> tuple[np.ndarray, list[list[str]]]:
    """Detect observable saturation and exactly frozen sensor runs.

    Gyro bias and rigid mounting rotation are intentionally not inferred here:
    without an external reference they can be indistinguishable from motion or
    a valid coordinate change.
    """
    accel = np.asarray(accel_mps2, dtype=float)
    gyro = np.asarray(gyro_rad_s, dtype=float)
    if accel.shape != gyro.shape or accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError("accel and gyro must have matching shape (n, 3)")
    if silence_min_samples < 2:
        raise ValueError("silence_min_samples must be at least two")
    quality = np.ones(len(accel), dtype=float)
    reasons: list[list[str]] = [[] for _ in range(len(accel))]
    saturated = (np.max(np.abs(accel), axis=1) >= accel_saturation_mps2) | (
        np.max(np.abs(gyro), axis=1) >= gyro_saturation_rad_s
    )
    quality[saturated] = 0.0
    for index in np.flatnonzero(saturated):
        reasons[index].append("sensor_saturation")
    if len(accel) >= silence_min_samples:
        unchanged = (np.max(np.abs(np.diff(accel, axis=0)), axis=1) <= silence_tolerance) & (
            np.max(np.abs(np.diff(gyro, axis=0)), axis=1) <= silence_tolerance
        )
        run_start = 0
        while run_start < len(unchanged):
            if not unchanged[run_start]:
                run_start += 1
                continue
            run_end = run_start + 1
            while run_end < len(unchanged) and unchanged[run_end]:
                run_end += 1
            sample_start = run_start
            sample_stop = run_end + 1
            if sample_stop - sample_start >= silence_min_samples:
                quality[sample_start:sample_stop] = 0.0
                for index in range(sample_start, sample_stop):
                    reasons[index].append("sensor_silence")
            run_start = run_end
    return quality, reasons


def sample_quality(
    timestamp_s: np.ndarray,
    accel_mps2: np.ndarray,
    gyro_rad_s: np.ndarray,
    magnetometer_uT: np.ndarray | None = None,
    config: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, list[list[str]]]:
    config = {} if config is None else dict(config)
    t = np.asarray(timestamp_s, dtype=float)
    accel = np.asarray(accel_mps2, dtype=float)
    gyro = np.asarray(gyro_rad_s, dtype=float)
    if accel.shape != gyro.shape or accel.shape != (len(t), 3):
        raise ValueError("aligned time, accel and gyro arrays are required")
    gravity = float(config.get("gravity_mps2", 9.80665))
    accel_tolerance = float(config.get("acceleration_norm_tolerance_mps2", 3.0))
    gap_factor = float(config.get("gap_factor", 1.5))
    quality = np.ones(len(t), dtype=float)
    reasons: list[list[str]] = [[] for _ in range(len(t))]
    finite = np.all(np.isfinite(accel), axis=1) & np.all(np.isfinite(gyro), axis=1) & np.isfinite(t)
    quality[~finite] = 0.0
    for index in np.flatnonzero(~finite):
        reasons[index].append("non_finite")
    accel_error = np.abs(np.linalg.norm(accel, axis=1) - gravity)
    dynamic = accel_error > accel_tolerance
    quality[dynamic] *= np.clip(1.0 - (accel_error[dynamic] - accel_tolerance) / max(2.0 * accel_tolerance, 1e-12), 0.25, 1.0)
    for index in np.flatnonzero(dynamic):
        reasons[index].append("dynamic_acceleration")
    if len(t) > 1:
        dt = np.diff(t)
        nominal = np.median(dt)
        gaps = np.flatnonzero(dt > gap_factor * nominal) + 1
        quality[gaps] = 0.0
        for index in gaps:
            reasons[index].append("timestamp_gap")
    if magnetometer_uT is not None:
        mag = np.asarray(magnetometer_uT, dtype=float)
        if mag.shape != accel.shape:
            raise ValueError("magnetometer must match accel shape")
        norm = np.linalg.norm(mag, axis=1)
        low = float(config.get("magnetic_field_min_uT", 15.0))
        high = float(config.get("magnetic_field_max_uT", 80.0))
        disturbed = (~np.all(np.isfinite(mag), axis=1)) | (norm < low) | (norm > high)
        quality[disturbed] *= 0.7
        for index in np.flatnonzero(disturbed):
            reasons[index].append("magnetic_disturbance")
    return np.clip(quality, 0.0, 1.0), reasons