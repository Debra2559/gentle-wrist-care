"""Explainable SheWrist exposure and intervention metrics."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def sample_durations(timestamp_s: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamp_s, dtype=float)
    if len(t) < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("at least two strictly increasing timestamps are required")
    diffs = np.diff(t)
    nominal = float(np.median(diffs))
    return np.concatenate((diffs, [nominal]))


def longest_continuous_duration(mask: np.ndarray, duration_s: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    duration = np.asarray(duration_s, dtype=float)
    if len(mask) != len(duration):
        raise ValueError("mask and duration must have equal length")
    best = current = 0.0
    for active, dt in zip(mask, duration):
        current = current + float(dt) if active else 0.0
        best = max(best, current)
    return best


def count_complete_cycles(angle_deg: np.ndarray, amplitude_deg: float = 8.0) -> float:
    angle = np.asarray(angle_deg, dtype=float)
    angle = angle[np.isfinite(angle)]
    if len(angle) < 3 or amplitude_deg <= 0.0:
        return 0.0
    excursion = 2.0 * float(amplitude_deg)
    running_min = running_max = float(angle[0])
    direction = 0
    half_cycles = 0
    for value in angle[1:]:
        value = float(value)
        if direction <= 0:
            running_min = min(running_min, value)
            if value - running_min >= excursion:
                half_cycles += 1
                direction = 1
                running_max = value
        if direction >= 0:
            running_max = max(running_max, value)
            if running_max - value >= excursion:
                half_cycles += 1
                direction = -1
                running_min = value
    return float(half_cycles // 2)


def intervention_efficiency(baseline_dose: float, support_dose: float) -> float | None:
    baseline = float(baseline_dose)
    if not np.isfinite(baseline) or baseline <= 0.0:
        return None
    return 100.0 * (baseline - float(support_dose)) / baseline


def exposure_metrics(
    timestamp_s: np.ndarray,
    theta_fe_deg: np.ndarray,
    theta_rud_deg: np.ndarray,
    config: Mapping[str, object],
    zones: Sequence[str] | None = None,
    quality: np.ndarray | None = None,
    pressure_kpa: np.ndarray | None = None,
    cable_tension_n: np.ndarray | None = None,
    lever_arm_m: np.ndarray | float | None = None,
) -> dict[str, float | int | None]:
    t = np.asarray(timestamp_s, dtype=float)
    fe = np.asarray(theta_fe_deg, dtype=float)
    rud = np.asarray(theta_rud_deg, dtype=float)
    if not (len(t) == len(fe) == len(rud)):
        raise ValueError("timestamps and angles must have equal length")
    dt = sample_durations(t)
    valid = np.isfinite(fe) & np.isfinite(rud)
    if quality is not None:
        valid &= np.asarray(quality, dtype=float) >= 0.2
    if zones is None:
        fe_yellow = float(config["angle_degrees"]["flexion_extension"]["yellow_abs"])
        rud_yellow = float(config["angle_degrees"]["radial_ulnar"]["yellow_abs"])
        high = (np.abs(fe) >= fe_yellow) | (np.abs(rud) >= rud_yellow)
    else:
        if len(zones) != len(t):
            raise ValueError("zones must match timestamps")
        high = np.array([zone in {"yellow", "red"} for zone in zones], dtype=bool)
    high &= valid
    valid_time = float(np.sum(dt[valid]))
    fe_threshold = float(config["angle_degrees"]["flexion_extension"]["yellow_abs"])
    rud_threshold = float(config["angle_degrees"]["radial_ulnar"]["yellow_abs"])
    dose_fe = float(np.sum(np.maximum(np.abs(fe[valid]) - fe_threshold, 0.0) * dt[valid]))
    dose_rud = float(np.sum(np.maximum(np.abs(rud[valid]) - rud_threshold, 0.0) * dt[valid]))
    duration_minutes = valid_time / 60.0
    cycle_amplitude = float(config["cycles_per_minute"]["cycle_amplitude_deg"])
    fe_cycles = count_complete_cycles(fe[valid], cycle_amplitude) if np.any(valid) else 0.0
    rud_cycles = count_complete_cycles(rud[valid], cycle_amplitude) if np.any(valid) else 0.0
    result: dict[str, float | int | None] = {
        "task_duration_s": valid_time,
        "valid_sample_pct": 100.0 * float(np.count_nonzero(valid)) / max(len(valid), 1),
        "P_high_pct": 100.0 * float(np.sum(dt[high])) / valid_time if valid_time > 0 else None,
        "D_FE_deg_s": dose_fe,
        "D_RUD_deg_s": dose_rud,
        "D_total_deg_s": dose_fe + dose_rud,
        "L_max_s": longest_continuous_duration(high, dt),
        "FE_cycles_per_min": fe_cycles / duration_minutes if duration_minutes > 0 else None,
        "RUD_cycles_per_min": rud_cycles / duration_minutes if duration_minutes > 0 else None,
        "max_abs_FE_deg": float(np.nanmax(np.abs(fe[valid]))) if np.any(valid) else None,
        "max_abs_RUD_deg": float(np.nanmax(np.abs(rud[valid]))) if np.any(valid) else None,
    }
    if pressure_kpa is not None:
        pressure = np.asarray(pressure_kpa, dtype=float)
        if len(pressure) != len(t):
            raise ValueError("pressure must match timestamps")
        finite_pressure = np.isfinite(pressure)
        red = float(config["pressure_kpa"]["red"])
        result["max_pressure_kPa"] = float(np.max(pressure[finite_pressure])) if np.any(finite_pressure) else None
        result["pressure_over_screening_s"] = float(np.sum(dt[finite_pressure & (pressure > red)]))
    if cable_tension_n is not None and lever_arm_m is not None:
        tension = np.asarray(cable_tension_n, dtype=float)
        lever = np.asarray(lever_arm_m, dtype=float)
        torque = tension * lever
        result["mean_external_assist_torque_Nm"] = float(np.nanmean(torque))
        result["max_external_assist_torque_Nm"] = float(np.nanmax(torque))
    return result