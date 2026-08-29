"""Import and audit the wired SheWrist wide-table capture format."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .quality import sensor_fault_quality

G_TO_MPS2 = 9.80665
DPS_TO_RAD_S = math.pi / 180.0
DEFAULT_ADC_MAX = 4095.0
DEFAULT_TARGET_SAMPLE_RATE_HZ = 50.0
MINIMUM_ANALYSIS_SAMPLE_RATE_HZ = 20.0
CANONICAL_FIELDS = [
    "device_ms",
    "sensor_id",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "quality",
    "quality_flags",
    "fsr_raw_adc",
]


class CaptureFormatError(ValueError):
    """Raised when a hardware capture cannot be converted without guessing."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise CaptureFormatError(f"{path.name}: CSV must be UTF-8 encoded") from exc
    if not fields or not rows:
        raise CaptureFormatError(f"{path.name}: CSV must include a header and at least one row")
    if len(fields) != len(set(fields)):
        raise CaptureFormatError(f"{path.name}: duplicate column names are not allowed")
    if any(None in row for row in rows):
        raise CaptureFormatError(f"{path.name}: one or more rows contain extra columns")
    return fields, rows


def _numbers(rows: Iterable[dict[str, str]], field: str, source_name: str) -> np.ndarray:
    values = []
    for row_number, row in enumerate(rows, start=2):
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError) as exc:
            raise CaptureFormatError(f"{source_name}: invalid {field} at row {row_number}") from exc
        if not math.isfinite(value):
            raise CaptureFormatError(f"{source_name}: non-finite {field} at row {row_number}")
        values.append(value)
    return np.asarray(values, dtype=float)


def _append_reason(reasons: list[list[str]], indices: np.ndarray, reason: str) -> None:
    for index in np.asarray(indices, dtype=int):
        if reason not in reasons[index]:
            reasons[index].append(reason)


def _sensor_values(
    rows: list[dict[str, str]],
    prefix: str,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    accel = np.column_stack(
        [_numbers(rows, f"{prefix}_{axis}_g", source_name) for axis in ("ax", "ay", "az")]
    ) * G_TO_MPS2
    gyro = np.column_stack(
        [_numbers(rows, f"{prefix}_{axis}_dps", source_name) for axis in ("gx", "gy", "gz")]
    ) * DPS_TO_RAD_S
    return accel, gyro


def _timestamp_audit(timestamp_ms: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    if len(timestamp_ms) < 2:
        raise CaptureFormatError("capture requires at least two samples")
    delta_ms = np.diff(timestamp_ms)
    if np.any(delta_ms <= 0.0):
        duplicate_count = int(np.count_nonzero(delta_ms == 0.0))
        backward_count = int(np.count_nonzero(delta_ms < 0.0))
        raise CaptureFormatError(
            f"timestamps must be strictly increasing; duplicates={duplicate_count}, backwards={backward_count}"
        )
    nominal_ms = float(np.median(delta_ms))
    gap_indices = np.flatnonzero(delta_ms > 1.5 * nominal_ms) + 1
    estimated_missing = int(
        np.sum(np.maximum(np.rint(delta_ms[gap_indices - 1] / nominal_ms).astype(int) - 1, 0))
    )
    report = {
        "duration_s": float((timestamp_ms[-1] - timestamp_ms[0]) / 1000.0),
        "sample_rate_hz_median": float(1000.0 / nominal_ms),
        "median_interval_ms": nominal_ms,
        "p95_interval_ms": float(np.percentile(delta_ms, 95)),
        "max_interval_ms": float(np.max(delta_ms)),
        "gap_count": int(len(gap_indices)),
        "estimated_missing_samples": estimated_missing,
        "duplicate_timestamp_count": 0,
        "backward_timestamp_count": 0,
    }
    return report, gap_indices


def _pressure_audit(
    rows: list[dict[str, str]],
    fields: list[str],
    source_name: str,
    adc_max: float,
) -> tuple[str | None, np.ndarray | None, dict[str, Any]]:
    pressure_field = next(
        (field for field in ("pressure_adc_raw", "pressure_adc") if field in fields),
        None,
    )
    if pressure_field is None:
        return None, None, {
            "available": False,
            "calibrated_to_pressure": False,
            "unit": "adc_count",
        }
    values = _numbers(rows, pressure_field, source_name)
    if np.any(values < 0.0):
        raise CaptureFormatError(f"{source_name}: {pressure_field} must be non-negative")
    saturated = values >= adc_max
    return pressure_field, values, {
        "available": True,
        "source_field": pressure_field,
        "unit": "adc_count",
        "calibrated_to_pressure": False,
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
        "adc_maximum_assumed": float(adc_max),
        "saturated_sample_count": int(np.count_nonzero(saturated)),
        "saturated_sample_pct": float(100.0 * np.mean(saturated)),
    }


def _quality_audit(
    accel: np.ndarray,
    gyro: np.ndarray,
    gap_indices: np.ndarray,
) -> tuple[np.ndarray, list[list[str]], dict[str, Any]]:
    quality, reasons = sensor_fault_quality(accel, gyro)
    zero_accel = np.flatnonzero(np.all(np.abs(accel) <= 1e-12, axis=1))
    if len(zero_accel):
        quality[zero_accel] = 0.0
        _append_reason(reasons, zero_accel, "zero_accelerometer_vector")
    if len(gap_indices):
        quality[gap_indices] = 0.0
        _append_reason(reasons, gap_indices, "timestamp_gap")
    reason_counts: dict[str, int] = {}
    for sample_reasons in reasons:
        for reason in sample_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    invalid = quality < 0.2
    report = {
        "valid_sample_count": int(np.count_nonzero(~invalid)),
        "invalid_sample_count": int(np.count_nonzero(invalid)),
        "valid_sample_pct": float(100.0 * np.mean(~invalid)),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    return quality, reasons, report


def _format_number(value: float) -> str:
    return f"{float(value):.12g}"


def audit_and_convert_capture(
    source: str | Path,
    destination: str | Path,
    *,
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    adc_max: float = DEFAULT_ADC_MAX,
) -> dict[str, Any]:
    """Convert one immutable wide capture to the canonical interleaved schema.

    The source is only opened for reading. The output carries SI units and per-node
    quality flags, but it is not considered analysis-ready without a separate CAL
    recording and session identity metadata.
    """
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise ValueError("destination must differ from source")
    source_hash_before = _sha256(source_path)
    fields, rows = _read_csv(source_path)

    timestamp_fields = [field for field in ("device_us", "device_ms") if field in fields]
    if len(timestamp_fields) != 1:
        raise CaptureFormatError(
            f"{source_path.name}: exactly one of device_us or device_ms is required"
        )
    required = {
        f"{prefix}_{axis}_{unit}"
        for prefix in ("hand", "arm")
        for axis, unit in (
            ("ax", "g"),
            ("ay", "g"),
            ("az", "g"),
            ("gx", "dps"),
            ("gy", "dps"),
            ("gz", "dps"),
        )
    }
    missing = sorted(required - set(fields))
    if missing:
        raise CaptureFormatError(f"{source_path.name}: missing required columns: {missing}")

    timestamp_field = timestamp_fields[0]
    source_timestamp = _numbers(rows, timestamp_field, source_path.name)
    timestamp_ms = source_timestamp / 1000.0 if timestamp_field == "device_us" else source_timestamp
    timing, gap_indices = _timestamp_audit(timestamp_ms)
    hand_accel, hand_gyro = _sensor_values(rows, "hand", source_path.name)
    forearm_accel, forearm_gyro = _sensor_values(rows, "arm", source_path.name)
    hand_quality, hand_reasons, hand_report = _quality_audit(hand_accel, hand_gyro, gap_indices)
    forearm_quality, forearm_reasons, forearm_report = _quality_audit(
        forearm_accel, forearm_gyro, gap_indices
    )
    pressure_field, pressure_values, pressure = _pressure_audit(
        rows, fields, source_path.name, adc_max
    )

    warnings = []
    rate = float(timing["sample_rate_hz_median"])
    tolerance = 0.1 * target_sample_rate_hz
    if rate < MINIMUM_ANALYSIS_SAMPLE_RATE_HZ:
        warnings.append("sample_rate_below_minimum_analysis_rate")
    if abs(rate - target_sample_rate_hz) > tolerance:
        warnings.append("sample_rate_differs_from_target_50hz")
    if timing["gap_count"]:
        warnings.append("timestamp_gaps_detected")
    if forearm_report["invalid_sample_count"]:
        warnings.append("forearm_invalid_samples_detected")
    if hand_report["invalid_sample_count"]:
        warnings.append("hand_invalid_samples_detected")
    if pressure.get("saturated_sample_count", 0):
        warnings.append("pressure_saturation_observed")
    if pressure.get("saturated_sample_pct", 0.0) > 50.0:
        warnings.append("pressure_majority_saturated")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS, lineterminator="\n")
            writer.writeheader()
            for index, device_ms in enumerate(timestamp_ms):
                for node, accel, gyro, quality, reasons in (
                    (
                        "forearm",
                        forearm_accel[index],
                        forearm_gyro[index],
                        forearm_quality[index],
                        forearm_reasons[index],
                    ),
                    (
                        "hand",
                        hand_accel[index],
                        hand_gyro[index],
                        hand_quality[index],
                        hand_reasons[index],
                    ),
                ):
                    writer.writerow(
                        {
                            "device_ms": _format_number(device_ms),
                            "sensor_id": node,
                            "ax": _format_number(accel[0]),
                            "ay": _format_number(accel[1]),
                            "az": _format_number(accel[2]),
                            "gx": _format_number(gyro[0]),
                            "gy": _format_number(gyro[1]),
                            "gz": _format_number(gyro[2]),
                            "quality": _format_number(quality),
                            "quality_flags": ";".join(reasons),
                            "fsr_raw_adc": (
                                _format_number(pressure_values[index])
                                if node == "forearm" and pressure_values is not None
                                else ""
                            ),
                        }
                    )
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    source_hash_after = _sha256(source_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"source capture changed while converting: {source_path}")
    return {
        "source_file": source_path.name,
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": source_hash_before,
        "source_unchanged": True,
        "source_schema": "wired_wide_dual_imu",
        "source_time_field": timestamp_field,
        "source_pressure_field": pressure_field,
        "source_units": {"acceleration": "g", "angular_velocity": "deg/s"},
        "output_file": destination_path.name,
        "output_path": str(destination_path),
        "output_sha256": _sha256(destination_path),
        "output_schema": "canonical_interleaved_dual_imu",
        "output_units": {"acceleration": "m/s2", "angular_velocity": "rad/s"},
        "row_count": len(rows),
        "output_row_count": 2 * len(rows),
        "timing": timing,
        "quality": {"forearm": forearm_report, "hand": hand_report},
        "pressure": pressure,
        "warnings": warnings,
        "conversion_status": "converted_with_warnings" if warnings else "converted",
        "canonical_parser_compatible": True,
        "analysis_ready": False,
        "analysis_blockers": [
            "missing_participant_condition_and_calibration_identity",
            "missing_separate_calibration_recording",
        ],
        "allowed_use": ["data_link_validation", "parser_compatibility", "fault_testing"],
    }


def audit_processed_angle_capture(source: str | Path) -> dict[str, Any]:
    """Audit legacy angle output without promoting it to canonical joint state."""
    source_path = Path(source).resolve()
    fields, rows = _read_csv(source_path)
    required = {"device_ms", "flex_deg", "deviation_deg"}
    missing = sorted(required - set(fields))
    if missing:
        raise CaptureFormatError(f"{source_path.name}: missing required columns: {missing}")
    timestamp_ms = _numbers(rows, "device_ms", source_path.name)
    timing, _ = _timestamp_audit(timestamp_ms)
    flex = _numbers(rows, "flex_deg", source_path.name)
    deviation = _numbers(rows, "deviation_deg", source_path.name)
    wrap_jumps = np.flatnonzero(np.abs(np.diff(deviation)) > 180.0) + 1
    warnings = ["no_external_angle_reference", "missing_calibration_identity"]
    if len(wrap_jumps):
        warnings.append("deviation_wrap_jumps_over_180deg")
    if np.max(np.abs(deviation)) > 90.0:
        warnings.append("deviation_absolute_value_over_90deg")
    return {
        "source_file": source_path.name,
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": _sha256(source_path),
        "source_schema": "legacy_processed_wrist_angles",
        "row_count": len(rows),
        "timing": timing,
        "flex_deg": {
            "minimum": float(np.min(flex)),
            "maximum": float(np.max(flex)),
            "range": float(np.ptp(flex)),
        },
        "deviation_deg": {
            "minimum": float(np.min(deviation)),
            "maximum": float(np.max(deviation)),
            "range": float(np.ptp(deviation)),
            "wrap_jump_count_over_180deg": int(len(wrap_jumps)),
        },
        "warnings": warnings,
        "analysis_ready": False,
        "promoted_to_joint_state": False,
        "allowed_use": ["diagnostic_review", "regression_fixture_after_manual_approval"],
    }


def import_capture_directory(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    adc_max: float = DEFAULT_ADC_MAX,
) -> dict[str, Any]:
    """Convert all known raw captures and write one evidence-bounded audit report."""
    source_dir = Path(source_directory).resolve()
    output_dir = Path(output_directory).resolve()
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")
    try:
        output_dir.relative_to(source_dir)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the immutable source directory")

    raw_files = sorted(source_dir.glob("imu_pressure_*.csv"))
    angle_files = sorted(source_dir.glob("wrist_*.csv"))
    if not raw_files and not angle_files:
        raise ValueError(f"no supported capture CSV files found in {source_dir}")
    standardized_dir = output_dir / "standardized"
    captures = []
    rejected = []
    for source_path in raw_files:
        destination = standardized_dir / f"{source_path.stem}_canonical.csv"
        try:
            captures.append(
                audit_and_convert_capture(
                    source_path,
                    destination,
                    target_sample_rate_hz=target_sample_rate_hz,
                    adc_max=adc_max,
                )
            )
        except CaptureFormatError as exc:
            rejected.append(
                {
                    "source_file": source_path.name,
                    "source_sha256": _sha256(source_path),
                    "conversion_status": "rejected",
                    "error": str(exc),
                }
            )
    processed_angles = []
    for source_path in angle_files:
        try:
            processed_angles.append(audit_processed_angle_capture(source_path))
        except CaptureFormatError as exc:
            rejected.append(
                {
                    "source_file": source_path.name,
                    "source_sha256": _sha256(source_path),
                    "conversion_status": "rejected",
                    "error": str(exc),
                }
            )

    rates = [float(item["timing"]["sample_rate_hz_median"]) for item in captures]
    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_classification": "unlabeled_wired_hardware_pilot",
        "source_directory": str(source_dir),
        "source_policy": "read_only_immutable_evidence",
        "identity": {
            "participant_id": None,
            "condition": None,
            "calibration_id": None,
            "task_type": None,
            "reminder_events_available": False,
        },
        "analysis_eligibility": {
            "data_link_validation": True,
            "parser_compatibility": bool(captures),
            "fault_testing": True,
            "wrist_angle_accuracy": False,
            "abc_effect_comparison": False,
            "clinical_or_prevention_claims": False,
            "reasons": [
                "no participant or A/B/C labels",
                "no separate CAL recording or calibration identity",
                "no external angle truth",
                "pressure is uncalibrated and often saturated",
            ],
        },
        "summary": {
            "raw_capture_file_count": len(raw_files),
            "converted_raw_capture_count": len(captures),
            "processed_angle_file_count": len(angle_files),
            "rejected_file_count": len(rejected),
            "raw_sample_count": int(sum(item["row_count"] for item in captures)),
            "raw_duration_s_sum": float(sum(item["timing"]["duration_s"] for item in captures)),
            "processed_angle_sample_count": int(
                sum(item["row_count"] for item in processed_angles)
            ),
            "processed_angle_duration_s_sum": float(
                sum(item["timing"]["duration_s"] for item in processed_angles)
            ),
            "sample_rate_hz_min": min(rates) if rates else None,
            "sample_rate_hz_max": max(rates) if rates else None,
            "files_with_timestamp_gaps": int(
                sum(item["timing"]["gap_count"] > 0 for item in captures)
            ),
            "files_with_majority_pressure_saturation": int(
                sum(item["pressure"].get("saturated_sample_pct", 0.0) > 50.0 for item in captures)
            ),
            "forearm_invalid_sample_count": int(
                sum(item["quality"]["forearm"]["invalid_sample_count"] for item in captures)
            ),
            "hand_invalid_sample_count": int(
                sum(item["quality"]["hand"]["invalid_sample_count"] for item in captures)
            ),
        },
        "captures": captures,
        "processed_angle_captures": processed_angles,
        "rejected_files": rejected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "audit_report.json"
    temporary_path = report_path.with_suffix(".json.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, report_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    report["audit_report_path"] = str(report_path)
    return report
