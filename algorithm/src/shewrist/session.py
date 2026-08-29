"""Auditable end-to-end offline sessions for public or normalized replay data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .calibration import interval_mask
from .data import load_sensor_file, resample_xyz
from .explanation import explain_analysis
from .faults import FaultSpec, inject_faults
from .kinematics import compute_wrist_kinematics, compute_wrist_kinematics_from_profile
from .ml_data import split_functional_repeats
from .quality import detect_stationary, nearest_sync_error_ms, sample_quality, sensor_fault_quality, timestamp_quality
from .replay import canonical_fingerprint, stream_exposure_states, verify_chunked_replay


PUBLIC_SENSOR_FILES = {
    "accel": "Accelerometer.txt",
    "gyro": "Gyroscope.txt",
    "mag": "Magnetometer.txt",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_public_raw_trial(dataset_root: str | Path, subject: str, set_name: str = "set2") -> tuple[dict[str, np.ndarray], list[Path]]:
    root = Path(dataset_root) / subject / set_name
    raw: dict[str, np.ndarray] = {}
    files: list[Path] = []
    for node in ("forearm", "hand"):
        streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for modality, filename in PUBLIC_SENSOR_FILES.items():
            path = root / node / filename
            streams[modality] = load_sensor_file(path)
            files.append(path)
        node_time = streams["accel"][0]
        raw[f"{node}_timestamp_s"] = node_time.copy()
        for modality, (time, values) in streams.items():
            raw[f"{node}_{modality}"] = values.copy() if np.array_equal(time, node_time) else resample_xyz(time, values, node_time)
    return raw, files


def _gap_mask(source_time: np.ndarray, target_time: np.ndarray, gap_factor: float) -> np.ndarray:
    source = np.asarray(source_time, dtype=float)
    target = np.asarray(target_time, dtype=float)
    mask = np.zeros(len(target), dtype=bool)
    if len(source) < 2:
        return np.ones(len(target), dtype=bool)
    dt = np.diff(source)
    nominal = float(np.median(dt))
    for index in np.flatnonzero(dt > gap_factor * nominal):
        mask |= (target > source[index]) & (target < source[index + 1])
    return mask


def _merge_reason_counts(*reason_lists: Sequence[Sequence[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for reasons in reason_lists:
        for sample_reasons in reasons:
            counts.update(str(reason) for reason in sample_reasons)
    return dict(sorted(counts.items()))


def neutral_calibration_quality(
    aligned: Mapping[str, np.ndarray],
    neutral_interval_s: tuple[float, float],
    quality_config: Mapping[str, object],
    acceptance_config: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    timestamp = np.asarray(aligned["timestamp_s"], dtype=float)
    interval = interval_mask(timestamp, *neutral_interval_s)
    interval_count = int(np.count_nonzero(interval))
    gravity = float(quality_config.get("gravity_mps2", 9.80665))
    accel_tolerance = float(quality_config.get("neutral_accel_tolerance_mps2", 1.5))
    gyro_threshold = float(quality_config.get("neutral_gyro_threshold_rad_s", 0.15))
    minimum_samples = int(acceptance_config.get("neutral_stationary_samples_min", 30))
    minimum_pct = float(acceptance_config.get("neutral_stationary_sample_pct_min", 70.0))
    node_masks: dict[str, np.ndarray] = {}
    node_audit: dict[str, dict[str, object]] = {}
    for node in ("forearm", "hand"):
        stationary = detect_stationary(
            np.asarray(aligned[f"{node}_accel"], dtype=float),
            np.asarray(aligned[f"{node}_gyro"], dtype=float),
            gravity=gravity,
            accel_tolerance=accel_tolerance,
            gyro_threshold_rad_s=gyro_threshold,
        )
        usable = stationary & (np.asarray(aligned[f"{node}_quality"], dtype=float) >= 0.2)
        node_masks[node] = usable
        selected_count = int(np.count_nonzero(interval & usable))
        node_audit[node] = {
            "stationary_sample_count": selected_count,
            "stationary_sample_pct": 100.0 * selected_count / max(interval_count, 1),
        }
    selected = interval & node_masks["forearm"] & node_masks["hand"]
    selected_count = int(np.count_nonzero(selected))
    selected_pct = 100.0 * selected_count / max(interval_count, 1)
    reasons = []
    if interval_count < minimum_samples:
        reasons.append("neutral_interval_too_short")
    if selected_count < minimum_samples:
        reasons.append("insufficient_joint_stationary_samples")
    if selected_pct < minimum_pct:
        reasons.append("neutral_stationary_fraction_below_minimum")
    passed = not reasons
    return selected, {
        "quality_gate_passed": passed,
        "quality_reasons": reasons,
        "neutral_interval_sample_count": interval_count,
        "neutral_stationary_sample_count": selected_count,
        "neutral_stationary_sample_pct": selected_pct,
        "neutral_stationary_sample_pct_min": minimum_pct,
        "neutral_stationary_samples_min": minimum_samples,
        "stationarity_thresholds": {
            "accel_norm_tolerance_mps2": accel_tolerance,
            "gyro_norm_max_rad_s": gyro_threshold,
        },
        "node_stationarity": node_audit,
    }


def synchronize_dual_imu(
    raw: Mapping[str, np.ndarray],
    sample_rate_hz: float,
    quality_config: Mapping[str, object],
    sync_limit_ms: float = 20.0,
    use_magnetometer: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    source_quality = {}
    for node in ("forearm", "hand"):
        time = np.asarray(raw[f"{node}_timestamp_s"], dtype=float)
        accel = np.asarray(raw[f"{node}_accel"], dtype=float)
        gyro = np.asarray(raw[f"{node}_gyro"], dtype=float)
        timestamp_quality(time, float(quality_config.get("gap_factor", 1.5)))
        if accel.shape != (len(time), 3) or gyro.shape != (len(time), 3):
            raise ValueError(f"{node} arrays must match timestamps with shape (n, 3)")
        source_quality[node] = timestamp_quality(time, float(quality_config.get("gap_factor", 1.5)))
    forearm_time = np.asarray(raw["forearm_timestamp_s"], dtype=float)
    hand_time = np.asarray(raw["hand_timestamp_s"], dtype=float)
    start_offset_ms = 1000.0 * abs(float(forearm_time[0] - hand_time[0]))
    end_offset_ms = 1000.0 * abs(float(forearm_time[-1] - hand_time[-1]))
    forearm_to_hand = nearest_sync_error_ms(forearm_time, hand_time)
    hand_to_forearm = nearest_sync_error_ms(hand_time, forearm_time)
    sync = {
        "median_sync_error_ms": max(forearm_to_hand["median_sync_error_ms"], hand_to_forearm["median_sync_error_ms"]),
        "p95_sync_error_ms": max(forearm_to_hand["p95_sync_error_ms"], hand_to_forearm["p95_sync_error_ms"]),
        "max_sync_error_ms": max(forearm_to_hand["max_sync_error_ms"], hand_to_forearm["max_sync_error_ms"]),
        "forearm_to_hand": forearm_to_hand,
        "hand_to_forearm": hand_to_forearm,
    }
    boundary_offset_ms = max(start_offset_ms, end_offset_ms)
    sync_gate_passed = bool(
        boundary_offset_ms <= sync_limit_ms
        and sync["p95_sync_error_ms"] <= sync_limit_ms
        and sync["max_sync_error_ms"] <= sync_limit_ms
    )
    start = max(float(forearm_time[0]), float(hand_time[0]))
    end = min(float(forearm_time[-1]), float(hand_time[-1]))
    if end <= start:
        raise ValueError("dual IMU streams have no common time range")
    count = int(np.floor((end - start) * sample_rate_hz)) + 1
    target = start + np.arange(count, dtype=float) / sample_rate_hz
    aligned: dict[str, np.ndarray] = {"timestamp_s": target}
    quality_by_node: dict[str, np.ndarray] = {}
    reason_counts: dict[str, dict[str, int]] = {}
    for node, source_time in (("forearm", forearm_time), ("hand", hand_time)):
        accel = resample_xyz(source_time, np.asarray(raw[f"{node}_accel"]), target)
        gyro = resample_xyz(source_time, np.asarray(raw[f"{node}_gyro"]), target)
        mag = resample_xyz(source_time, np.asarray(raw[f"{node}_mag"]), target) if f"{node}_mag" in raw else None
        aligned[f"{node}_accel"] = accel
        aligned[f"{node}_gyro"] = gyro
        if mag is not None:
            aligned[f"{node}_mag"] = mag
        base_quality, base_reasons = sample_quality(
            target,
            accel,
            gyro,
            mag if use_magnetometer else None,
            quality_config,
        )
        fault_quality, fault_reasons = sensor_fault_quality(
            accel,
            gyro,
            float(quality_config.get("accel_saturation_mps2", 100.0)),
            float(quality_config.get("gyro_saturation_rad_s", 20.0)),
            int(quality_config.get("silence_min_samples", 10)),
        )
        gap = _gap_mask(source_time, target, float(quality_config.get("gap_factor", 1.5)))
        observable_fault = gap | (fault_quality <= 0.0)
        combined = np.minimum(base_quality, fault_quality)
        combined[gap] = 0.0
        first_fault = np.flatnonzero(observable_fault)
        if len(first_fault):
            combined[first_fault[0] :] = 0.0
        if not sync_gate_passed:
            combined[:] = 0.0
        quality_by_node[node] = combined
        gap_reasons = [["source_timestamp_gap"] if value else [] for value in gap]
        counts = _merge_reason_counts(base_reasons, fault_reasons, gap_reasons)
        if len(first_fault):
            counts["recalibration_required_after_fault"] = len(target) - int(first_fault[0])
        if not sync_gate_passed:
            counts["node_clock_sync_error"] = len(target)
        reason_counts[node] = counts
    aligned["forearm_quality"] = quality_by_node["forearm"]
    aligned["hand_quality"] = quality_by_node["hand"]
    audit = {
        "sample_rate_hz": float(sample_rate_hz),
        "common_sample_count": len(target),
        "common_start_s": float(target[0]),
        "common_end_s": float(target[-1]),
        "sync_limit_ms": float(sync_limit_ms),
        "start_offset_ms": start_offset_ms,
        "end_offset_ms": end_offset_ms,
        "boundary_offset_ms": boundary_offset_ms,
        "nearest_sync": sync,
        "sync_gate_basis": "boundary_offset_and_bidirectional_p95_and_max",
        "sync_gate_passed": sync_gate_passed,
        "source_timestamp_quality": source_quality,
        "quality_reason_counts": reason_counts,
    }
    return aligned, audit


def prepare_joint_state_from_raw(
    raw: Mapping[str, np.ndarray],
    neutral_interval_s: tuple[float, float],
    functional_annotations: Sequence[Mapping[str, object]],
    algorithm_config: Mapping[str, object],
    calibration_id: str,
    sample_rate_hz: float = 100.0,
    initialize_from_accel: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    quality_config = algorithm_config["quality"]
    sync_limit = float(algorithm_config.get("acceptance", {}).get("node_sync_ms_max", 20.0))
    use_magnetometer = bool(algorithm_config["fusion"].get("use_magnetometer", False))
    aligned, sync_audit = synchronize_dual_imu(
        raw,
        sample_rate_hz,
        quality_config,
        sync_limit,
        use_magnetometer,
    )
    calibration_mask, calibration_quality = neutral_calibration_quality(
        aligned,
        neutral_interval_s,
        quality_config,
        algorithm_config.get("acceptance", {}),
    )
    algorithm = str(algorithm_config["fusion"].get("default_algorithm", "madgwick"))
    result = compute_wrist_kinematics(
        aligned["timestamp_s"],
        aligned["forearm_accel"],
        aligned["forearm_gyro"],
        aligned["hand_accel"],
        aligned["hand_gyro"],
        neutral_interval_s,
        functional_annotations,
        aligned.get("forearm_mag") if use_magnetometer else None,
        aligned.get("hand_mag") if use_magnetometer else None,
        algorithm,
        algorithm_config["fusion"],
        quality_config,
        neutral_sample_mask=calibration_mask if np.count_nonzero(calibration_mask) >= 3 else None,
        initialize_from_accel=initialize_from_accel,
    )
    quality = np.minimum.reduce((aligned["forearm_quality"], aligned["hand_quality"], result.quality))
    if not calibration_quality["quality_gate_passed"]:
        quality[:] = 0.0
    joint = {
        "timestamp_ms": result.timestamp_s * 1000.0,
        "theta_FE": result.theta_fe_deg,
        "theta_RUD": result.theta_rud_deg,
        "theta_thumb": np.full(len(result.timestamp_s), np.nan),
        "angular_velocity": result.angular_velocity_deg_s,
        "calibration_id": np.full(len(result.timestamp_s), calibration_id, dtype=object),
        "quality": quality,
    }
    neutral_mask = interval_mask(result.timestamp_s, *neutral_interval_s)
    calibration = {
        "status": "passed" if calibration_quality["quality_gate_passed"] else "rejected",
        **calibration_quality,
        "calibration_id": calibration_id,
        "algorithm": result.algorithm,
        "neutral_interval_s": list(neutral_interval_s),
        "neutral_sample_count": int(np.count_nonzero(neutral_mask)),
        "neutral_samples_used_for_bias_and_zero": int(np.count_nonzero(calibration_mask)) if np.count_nonzero(calibration_mask) >= 3 else int(np.count_nonzero(neutral_mask)),
        "flexion_extension_axis": result.axes.flexion_extension.tolist(),
        "radial_ulnar_axis": result.axes.radial_ulnar.tolist(),
        "pronation_supination_axis": result.axes.pronation_supination.tolist(),
        "neutral_quaternion": result.neutral_quaternion.tolist(),
        "forearm_gyro_bias_rad_s": result.forearm_gyro_bias_rad_s.tolist(),
        "hand_gyro_bias_rad_s": result.hand_gyro_bias_rad_s.tolist(),
        "orientation_initialization": "accelerometer_tilt_yaw_zero" if initialize_from_accel else "identity",
    }
    return joint, {"synchronization": sync_audit, "calibration": calibration}


def prepare_joint_state_from_calibration_profile(
    raw: Mapping[str, np.ndarray],
    calibration_profile: Mapping[str, object],
    algorithm_config: Mapping[str, object],
    calibration_id: str,
    sample_rate_hz: float = 100.0,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Apply a frozen calibration profile to a separately recorded task file."""
    quality_config = algorithm_config["quality"]
    sync_limit = float(algorithm_config.get("acceptance", {}).get("node_sync_ms_max", 20.0))
    use_magnetometer = bool(algorithm_config["fusion"].get("use_magnetometer", False))
    aligned, sync_audit = synchronize_dual_imu(
        raw,
        sample_rate_hz,
        quality_config,
        sync_limit,
        use_magnetometer,
    )
    algorithm = str(algorithm_config["fusion"].get("default_algorithm", "madgwick"))
    result = compute_wrist_kinematics_from_profile(
        aligned["timestamp_s"],
        aligned["forearm_accel"],
        aligned["forearm_gyro"],
        aligned["hand_accel"],
        aligned["hand_gyro"],
        calibration_profile,
        aligned.get("forearm_mag") if use_magnetometer else None,
        aligned.get("hand_mag") if use_magnetometer else None,
        algorithm,
        algorithm_config["fusion"],
        quality_config,
    )
    quality = np.minimum.reduce((aligned["forearm_quality"], aligned["hand_quality"], result.quality))
    quality_gate_passed = calibration_profile.get("quality_gate_passed") is not False
    if not quality_gate_passed:
        quality[:] = 0.0
    joint = {
        "timestamp_ms": result.timestamp_s * 1000.0,
        "theta_FE": result.theta_fe_deg,
        "theta_RUD": result.theta_rud_deg,
        "theta_thumb": np.full(len(result.timestamp_s), np.nan),
        "angular_velocity": result.angular_velocity_deg_s,
        "calibration_id": np.full(len(result.timestamp_s), calibration_id, dtype=object),
        "quality": quality,
    }
    calibration = {
        "status": "passed" if quality_gate_passed else "rejected",
        "quality_gate_passed": quality_gate_passed,
        "quality_reasons": [] if quality_gate_passed else ["source_calibration_quality_gate_failed"],
        "calibration_id": calibration_id,
        "algorithm": calibration_profile.get("algorithm", algorithm),
        "application_mode": "separate_calibration_file",
        "task_axes_reestimated": False,
        "task_neutral_reestimated": False,
        "reference_application": "stored_calibration_neutral_quaternion",
        "orientation_initialization": "accelerometer_tilt_yaw_zero",
        "flexion_extension_axis": list(calibration_profile["flexion_extension_axis"]),
        "radial_ulnar_axis": list(calibration_profile["radial_ulnar_axis"]),
        "pronation_supination_axis": list(calibration_profile["pronation_supination_axis"]),
        "neutral_quaternion": list(calibration_profile["neutral_quaternion"]),
        "forearm_gyro_bias_rad_s": list(calibration_profile["forearm_gyro_bias_rad_s"]),
        "hand_gyro_bias_rad_s": list(calibration_profile["hand_gyro_bias_rad_s"]),
    }
    return joint, {"synchronization": sync_audit, "calibration": calibration}


def prepare_public_joint_state(
    dataset_root: str | Path,
    annotations: Sequence[Mapping[str, object]],
    subject: str,
    set_name: str,
    algorithm_config: Mapping[str, object],
    faults: Sequence[FaultSpec] = (),
    sample_rate_hz: float = 100.0,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[Path]]:
    raw, raw_files = load_public_raw_trial(dataset_root, subject, set_name)
    corrupted, oracle_invalid, fault_audit = inject_faults(raw, faults)
    calibration_rows, validation_rows = split_functional_repeats(annotations)
    neutral_candidates = [
        row for row in annotations
        if str(row.get("Category")) == "Relative"
        and str(row.get("Segment")) == "wrist"
        and str(row.get("Type")) == "AnatomicalPos"
    ]
    if not neutral_candidates:
        raise ValueError("no relative wrist AnatomicalPos interval found")
    selected_neutral = max(neutral_candidates, key=lambda row: float(row["End"]) - float(row["Init"]))
    neutral = (float(selected_neutral["Init"]), float(selected_neutral["End"]))
    calibration_seed = {
        "subject": subject,
        "set": set_name,
        "neutral": neutral,
        "functional_annotations": calibration_rows,
        "algorithm": algorithm_config["fusion"],
    }
    calibration_id = f"{subject}-{set_name}-{canonical_fingerprint(calibration_seed)[:12]}"
    joint, audit = prepare_joint_state_from_raw(
        corrupted,
        neutral,
        calibration_rows,
        algorithm_config,
        calibration_id,
        sample_rate_hz,
    )
    audit["fault_injection"] = {
        "enabled": bool(faults),
        "specs": fault_audit,
        "oracle_invalid_samples": {key: int(np.count_nonzero(value)) for key, value in oracle_invalid.items()},
        "note": "Oracle masks document injected locations and are not supplied to the inference algorithm.",
    }
    audit["validation_intervals"] = [dict(row) for row in validation_rows]
    return joint, audit, raw_files


def timeline_rows(
    joint_state: Mapping[str, np.ndarray],
    analysis: Mapping[str, object],
    algorithm_config: Mapping[str, object],
    chunk_size: int,
    pressure_kpa: np.ndarray | None = None,
    discomfort: np.ndarray | None = None,
    user_continues: np.ndarray | None = None,
    angle_alerts_enabled: bool = True,
    mechanical_recommendations_enabled: bool = True,
) -> list[dict[str, object]]:
    states = stream_exposure_states(
        joint_state,
        algorithm_config,
        chunk_size,
        pressure_kpa,
        discomfort,
        user_continues,
        angle_alerts_enabled,
        mechanical_recommendations_enabled,
    )
    timestamp = np.asarray(joint_state["timestamp_ms"], dtype=float)
    activity = np.full(len(timestamp), "background_or_rejected", dtype=object)
    ml = analysis.get("ml_shadow", {})
    tokens = ml.get("tokens", []) if isinstance(ml, Mapping) else []
    for token in tokens:
        if isinstance(token, Mapping):
            selected = (timestamp >= float(token["start_ms"])) & (timestamp <= float(token["end_ms"]))
            activity[selected] = str(token["event_type"])
    rows = []
    quality = np.asarray(joint_state.get("quality", np.ones(len(timestamp))), dtype=float)
    discomfort_values = None if discomfort is None else np.asarray(discomfort, dtype=bool)
    continues_values = None if user_continues is None else np.asarray(user_continues, dtype=bool)
    for index, state in enumerate(states):
        rows.append(
            {
                "timestamp_ms": float(timestamp[index]),
                "theta_FE": float(np.asarray(joint_state["theta_FE"])[index]),
                "theta_RUD": float(np.asarray(joint_state["theta_RUD"])[index]),
                "quality": float(quality[index]),
                "angle_zone": state["angle_zone"],
                "pressure_zone": state["pressure_zone"],
                "discomfort": None if discomfort_values is None else bool(discomfort_values[index]),
                "user_continues": None if continues_values is None else bool(continues_values[index]),
                "activity_shadow": str(activity[index]),
                "alert": bool(state["alert"]),
                "would_alert": bool(state["would_alert"]),
                "alert_reason": state["alert_reason"],
                "safety_stop": bool(state["safety_stop"]),
            }
        )
    return rows


def analyze_session(
    joint_state: Mapping[str, np.ndarray],
    algorithm_config: Mapping[str, object],
    ml_config: Mapping[str, object],
    activity_pipeline,
    explanation_config: Mapping[str, object],
    session_id: str,
    evidence_type: str,
    chunk_size: int = 128,
    pressure_kpa: np.ndarray | None = None,
    discomfort: np.ndarray | None = None,
    cable_tension_n: np.ndarray | None = None,
    lever_arm_m: float | np.ndarray | None = None,
    user_continues: np.ndarray | None = None,
    angle_alerts_enabled: bool = True,
    mechanical_recommendations_enabled: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    analysis, replay_audit = verify_chunked_replay(
        joint_state,
        algorithm_config,
        ml_config,
        activity_pipeline,
        session_id,
        evidence_type,
        chunk_size,
        pressure_kpa=pressure_kpa,
        discomfort=discomfort,
        cable_tension_n=cable_tension_n,
        lever_arm_m=lever_arm_m,
        user_continues=user_continues,
        angle_alerts_enabled=angle_alerts_enabled,
        mechanical_recommendations_enabled=mechanical_recommendations_enabled,
    )
    analysis["replay"] = replay_audit
    analysis["explanation"] = explain_analysis(analysis, explanation_config)
    analysis["evidence_limits"] = [
        "Public/replayed data provide an engineering baseline, not target-hardware validation.",
        "No output estimates disease risk, tissue strain, treatment benefit, or clinical safety.",
        "ML and explanation providers have no alarm or mechanical-control authority.",
    ]
    rows = timeline_rows(
        joint_state,
        analysis,
        algorithm_config,
        chunk_size,
        pressure_kpa,
        discomfort,
        user_continues,
        angle_alerts_enabled,
        mechanical_recommendations_enabled,
    )
    return analysis, rows