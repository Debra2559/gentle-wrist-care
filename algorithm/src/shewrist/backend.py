"""File-backed backend service for auditable SheWrist offline analysis jobs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .baseline import (
    PersonalBaseline,
    build_personal_report,
    init_personal_baseline,
    load_personal_baseline,
    save_personal_baseline,
    session_exposure_summary,
    update_personal_baseline,
)
from .data import align_mechanical_to_joint, load_config, load_joint_state_csv, load_mechanical_csv, write_csv, write_json
from .ml import ShadowActivityPipeline
from .reporting import plot_session_report
from .session import (
    analyze_session,
    prepare_joint_state_from_calibration_profile,
    prepare_joint_state_from_raw,
    sha256_file,
)


SCHEMA_VERSION = "1.0"
ALGORITHM_RELEASE = "offline-v0.8"
EVIDENCE_TYPES = {"bench", "replay", "simulation", "human"}
INPUT_TYPES = {"joint_state", "raw_dual_imu"}
TARGET_SENSOR_PLACEMENTS = {
    "forearm": {"right_distal_forearm", "left_distal_forearm"},
    "hand": {"right_hand_third_metacarpal_dorsum", "left_hand_third_metacarpal_dorsum"},
}
TRIAL_CONDITIONS = {
    "A": {"support_level": 0, "reminder_enabled": False},
    "B": {"support_level": 1, "reminder_enabled": False},
    "C": {"support_level": 1, "reminder_enabled": True},
}
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{20}$")
ARTIFACT_NAMES = {
    "analysis.json",
    "joint_state.csv",
    "timeline.csv",
    "tokens.json",
    "manifest.json",
    "session_report.png",
    "session_report.svg",
    "personal_baseline.json",
}
METRIC_NAMES = {
    "task_duration_s": "task_duration_s",
    "valid_sample_pct": "valid_sample_pct",
    "high_posture_time_pct": "P_high_pct",
    "fe_excess_dose_deg_s": "D_FE_deg_s",
    "rud_excess_dose_deg_s": "D_RUD_deg_s",
    "total_excess_dose_deg_s": "D_total_deg_s",
    "longest_high_posture_s": "L_max_s",
    "fe_cycles_per_min": "FE_cycles_per_min",
    "rud_cycles_per_min": "RUD_cycles_per_min",
    "max_abs_fe_deg": "max_abs_FE_deg",
    "max_abs_rud_deg": "max_abs_RUD_deg",
    "alert_count": "alert_count",
    "would_alert_count": "would_alert_count",
    "mechanical_recommendation_count": "mechanical_recommendation_count",
    "safety_stop_count": "safety_stop_count",
    "max_pressure_kpa": "max_pressure_kPa",
    "pressure_over_screening_s": "pressure_over_screening_s",
    "mean_external_assist_torque_nm": "mean_external_assist_torque_Nm",
    "max_external_assist_torque_nm": "max_external_assist_torque_Nm",
}


class BackendError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: Optional[str] = None,
        retryable: bool = False,
        details: Optional[Mapping[str, object]] = None,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.retryable = retryable
        self.details = dict(details or {})
        self.http_status = int(http_status)

    def payload(self) -> dict[str, object]:
        error: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.field is not None:
            error["field"] = self.field
        if self.details:
            error["details"] = self.details
        return {"schema_version": SCHEMA_VERSION, "error": error}


@dataclass(frozen=True)
class BackendSettings:
    project_root: Path
    output_root: Path
    algorithm_config: Path
    ml_config: Path
    explanation_config: Path
    model_path: Path

    @classmethod
    def default(cls, project_root: str | Path) -> "BackendSettings":
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            output_root=root / "outputs/api",
            algorithm_config=root / "config/thresholds.yaml",
            ml_config=root / "config/ml_activity.json",
            explanation_config=root / "config/explanation_api.json",
            model_path=root / "outputs/ml/activity_cnn_hmm_shadow.npz",
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_fingerprint(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BackendError("INVALID_SCHEMA", f"{field} must be an object.", field=field)
    return dict(value)


def _finite_number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendError("INVALID_SCHEMA", f"{field} must be a finite number.", field=field) from exc
    if not math.isfinite(number):
        raise BackendError("INVALID_SCHEMA", f"{field} must be a finite number.", field=field)
    return number


def _validate_raw_sensor_block(metadata: Mapping[str, object]) -> None:
    """Validate raw dual-IMU sensor units, node identity, and target placements."""
    units = _mapping(metadata.get("sensor_units", {}), "metadata.sensor_units")
    if units.get("acceleration") != "m/s2" or units.get("angular_velocity") != "rad/s":
        raise BackendError(
            "INVALID_UNIT",
            "Raw IMU requires acceleration=m/s2 and angular_velocity=rad/s.",
            field="metadata.sensor_units",
        )
    sensors = metadata.get("sensors")
    if not isinstance(sensors, list) or len(sensors) != 2:
        raise BackendError(
            "MISSING_SENSOR_NODE",
            "sensors must declare exactly forearm and hand nodes.",
            field="metadata.sensors",
        )
    sensor_entries = [_mapping(item, f"metadata.sensors[{index}]") for index, item in enumerate(sensors)]
    sensor_ids = [str(item.get("sensor_id", "")) for item in sensor_entries]
    if len(set(sensor_ids)) != 2 or set(sensor_ids) != {"forearm", "hand"}:
        raise BackendError(
            "MISSING_SENSOR_NODE",
            "sensors must declare exactly forearm and hand nodes.",
            field="metadata.sensors",
        )
    sensor_by_id = {str(item["sensor_id"]): item for item in sensor_entries}
    for node, allowed in TARGET_SENSOR_PLACEMENTS.items():
        placement = str(sensor_by_id[node].get("placement", ""))
        if placement not in allowed:
            raise BackendError(
                "INVALID_SENSOR_PLACEMENT",
                f"{node} must use a target-hardware placement.",
                field="metadata.sensors",
                details={"sensor_id": node, "placement": placement, "allowed": sorted(allowed)},
            )
        if str(sensor_by_id[node].get("coordinate_frame", "")) != "sensor_local":
            raise BackendError(
                "INVALID_SENSOR_PLACEMENT",
                f"{node} coordinate_frame must be sensor_local.",
                field="metadata.sensors",
                details={"sensor_id": node},
            )
    sides = {str(sensor_by_id[node]["placement"]).split("_", 1)[0] for node in TARGET_SENSOR_PLACEMENTS}
    if len(sides) != 1:
        raise BackendError(
            "INVALID_SENSOR_PLACEMENT",
            "forearm and hand placements must be on the same side.",
            field="metadata.sensors",
        )
    magnetic_unit = units.get("magnetic_field")
    if magnetic_unit not in {None, "uT"}:
        raise BackendError("INVALID_UNIT", "magnetic_field must use uT when supplied.", field="metadata.sensor_units.magnetic_field")


def _validate_calibration_segments(metadata: Mapping[str, object]) -> None:
    """Validate that neutral plus directional calibration intervals are present and bounded."""
    calibration = _mapping(metadata.get("calibration", {}), "metadata.calibration")
    segments = calibration.get("segments")
    if not isinstance(segments, list):
        raise BackendError("CALIBRATION_REQUIRED", "calibration.segments is required.", field="metadata.calibration.segments")
    required = {"neutral", "flexion", "extension", "ulnar_deviation"}
    observed = set()
    for index, segment_value in enumerate(segments):
        segment = _mapping(segment_value, f"metadata.calibration.segments[{index}]")
        kind = str(segment.get("type", ""))
        start = _finite_number(segment.get("start_ms"), f"metadata.calibration.segments[{index}].start_ms")
        end = _finite_number(segment.get("end_ms"), f"metadata.calibration.segments[{index}].end_ms")
        if end <= start:
            raise BackendError("CALIBRATION_REQUIRED", "Calibration interval end must exceed start.", field=f"metadata.calibration.segments[{index}]")
        if kind != "neutral" and end - start > 15000.0:
            raise BackendError("CALIBRATION_REQUIRED", "Functional calibration intervals must not exceed 15 seconds.", field=f"metadata.calibration.segments[{index}]")
        observed.add(kind)
    missing = sorted(required - observed)
    if missing:
        raise BackendError("CALIBRATION_REQUIRED", "Missing required calibration segments.", field="metadata.calibration.segments", details={"missing": missing})


def validate_calibration_metadata(payload: object) -> dict[str, object]:
    """Validate metadata for a standalone calibration-profile record (POST /api/v1/calibrations)."""
    metadata = _mapping(payload, "metadata")
    version = str(metadata.get("schema_version", ""))
    if version not in {"1", "1.0"}:
        raise BackendError("INVALID_SCHEMA", "schema_version must be 1.0.", field="metadata.schema_version")
    calibration_id = str(metadata.get("calibration_id", ""))
    if not SESSION_ID_PATTERN.fullmatch(calibration_id):
        raise BackendError(
            "INVALID_SCHEMA",
            "calibration_id must contain 1-128 ASCII letters, digits, dots, underscores, or hyphens.",
            field="metadata.calibration_id",
        )
    participant_value = metadata.get("participant_id")
    participant_id = None if participant_value in {None, ""} else str(participant_value)
    if participant_id is not None and not SESSION_ID_PATTERN.fullmatch(participant_id):
        raise BackendError(
            "INVALID_SCHEMA",
            "participant_id must contain 1-128 ASCII letters, digits, dots, underscores, or hyphens.",
            field="metadata.participant_id",
        )
    _validate_raw_sensor_block(metadata)
    _validate_calibration_segments(metadata)
    normalized = deepcopy(metadata)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["calibration_id"] = calibration_id
    if participant_id is not None:
        normalized["participant_id"] = participant_id
    return normalized


def validate_metadata(payload: object) -> dict[str, object]:
    metadata = _mapping(payload, "metadata")
    version = str(metadata.get("schema_version", ""))
    if version not in {"1", "1.0"}:
        raise BackendError("INVALID_SCHEMA", "schema_version must be 1.0.", field="metadata.schema_version")
    session_id = str(metadata.get("session_id", ""))
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise BackendError(
            "INVALID_SCHEMA",
            "session_id must contain 1-128 ASCII letters, digits, dots, underscores, or hyphens.",
            field="metadata.session_id",
        )
    participant_value = metadata.get("participant_id")
    participant_id = None if participant_value in {None, ""} else str(participant_value)
    if participant_id is not None and not SESSION_ID_PATTERN.fullmatch(participant_id):
        raise BackendError(
            "INVALID_SCHEMA",
            "participant_id must contain 1-128 ASCII letters, digits, dots, underscores, or hyphens.",
            field="metadata.participant_id",
        )
    input_type = str(metadata.get("input_type", ""))
    if input_type not in INPUT_TYPES:
        raise BackendError("INVALID_SCHEMA", "input_type must be joint_state or raw_dual_imu.", field="metadata.input_type")
    evidence_type = str(metadata.get("evidence_type", ""))
    if evidence_type not in EVIDENCE_TYPES:
        raise BackendError("INVALID_SCHEMA", "Unsupported evidence_type.", field="metadata.evidence_type")
    timestamp_basis = str(metadata.get("timestamp_basis", "session_relative_ms"))
    if timestamp_basis not in {"session_relative_ms", "device_ms"}:
        raise BackendError("INVALID_SCHEMA", "timestamp_basis must be session_relative_ms or device_ms.", field="metadata.timestamp_basis")
    if input_type == "joint_state" and timestamp_basis != "session_relative_ms":
        raise BackendError("INVALID_SCHEMA", "joint_state input requires session_relative_ms.", field="metadata.timestamp_basis")
    if evidence_type == "human":
        compliance = _mapping(metadata.get("compliance", {}), "metadata.compliance")
        if compliance.get("deidentified") is not True or compliance.get("consent_confirmed") is not True:
            raise BackendError(
                "HUMAN_DATA_CONFIRMATION_REQUIRED",
                "Human data require deidentified=true and consent_confirmed=true.",
                field="metadata.compliance",
            )
    condition_value = metadata.get("condition")
    condition = None if condition_value in {None, ""} else str(condition_value).upper()
    if condition is not None:
        if condition not in TRIAL_CONDITIONS:
            raise BackendError("INVALID_SCHEMA", "condition must be A, B, or C.", field="metadata.condition")
        expected = TRIAL_CONDITIONS[condition]
        try:
            support_level = int(metadata.get("support_level"))
        except (TypeError, ValueError) as exc:
            raise BackendError("INVALID_SCHEMA", "support_level is required for A/B/C.", field="metadata.support_level") from exc
        reminder_enabled = metadata.get("reminder_enabled")
        if not isinstance(reminder_enabled, bool):
            raise BackendError("INVALID_SCHEMA", "reminder_enabled must be boolean for A/B/C.", field="metadata.reminder_enabled")
        if support_level != expected["support_level"] or reminder_enabled is not expected["reminder_enabled"]:
            raise BackendError(
                "INVALID_TRIAL_CONDITION",
                "A/B/C settings do not match the frozen field protocol.",
                field="metadata.condition",
                details={"condition": condition, "expected": expected},
            )
        if evidence_type == "human":
            for version_field in ("firmware_version", "task_version"):
                if not str(metadata.get(version_field, "")).strip():
                    raise BackendError("INVALID_SCHEMA", f"{version_field} is required for human A/B/C data.", field=f"metadata.{version_field}")
            calibration_value = metadata.get("calibration")
            calibration_id = calibration_value.get("calibration_id") if isinstance(calibration_value, Mapping) else None
            if not str(calibration_id or "").strip():
                raise BackendError("INVALID_SCHEMA", "calibration.calibration_id is required for human A/B/C data.", field="metadata.calibration.calibration_id")
    options = _mapping(metadata.get("options", {}), "metadata.options")
    if options.get("enable_ml_shadow", True) is not True:
        raise BackendError(
            "UNSUPPORTED_OPTION",
            "API v1 requires enable_ml_shadow=true; the model remains non-controlling.",
            field="metadata.options.enable_ml_shadow",
        )
    try:
        chunk_size = int(options.get("chunk_size", 128))
    except (TypeError, ValueError) as exc:
        raise BackendError("INVALID_SCHEMA", "chunk_size must be an integer.", field="metadata.options.chunk_size") from exc
    if chunk_size < 1 or chunk_size > 100000:
        raise BackendError("INVALID_SCHEMA", "chunk_size must be between 1 and 100000.", field="metadata.options.chunk_size")
    provider = str(options.get("explanation_provider", "local_template"))
    if provider not in {"local_template", "template", "openai_compatible"}:
        raise BackendError("INVALID_SCHEMA", "Unsupported explanation_provider.", field="metadata.options.explanation_provider")
    external_enabled = bool(options.get("enable_external_api", False))
    if provider == "openai_compatible" and not external_enabled:
        raise BackendError(
            "EXPLANATION_CONFIG_ERROR",
            "openai_compatible requires enable_external_api=true.",
            field="metadata.options.enable_external_api",
        )
    if provider != "openai_compatible" and external_enabled:
        raise BackendError(
            "EXPLANATION_CONFIG_ERROR",
            "enable_external_api is only valid with openai_compatible.",
            field="metadata.options.enable_external_api",
        )
    threshold_version = str(options.get("threshold_version", "engineering_v1"))
    if threshold_version != "engineering_v1":
        raise BackendError("UNSUPPORTED_OPTION", "Only engineering_v1 is available.", field="metadata.options.threshold_version")
    baseline_role = str(options.get("personal_baseline_role", "auto"))
    if baseline_role not in {"auto", "enroll", "update"}:
        raise BackendError("INVALID_SCHEMA", "personal_baseline_role must be auto, enroll, or update.", field="metadata.options.personal_baseline_role")
    if baseline_role != "auto" and participant_id is None:
        raise BackendError("INVALID_SCHEMA", "personal_baseline_role requires metadata.participant_id.", field="metadata.participant_id")
    if "baseline_target_reduction_pct" in options:
        target_reduction = _finite_number(options["baseline_target_reduction_pct"], "metadata.options.baseline_target_reduction_pct")
        if target_reduction < 0.0 or target_reduction >= 100.0:
            raise BackendError("INVALID_SCHEMA", "baseline_target_reduction_pct must be within [0, 100).", field="metadata.options.baseline_target_reduction_pct")
    if "lever_arm_m" in options:
        lever_arm = _finite_number(options["lever_arm_m"], "metadata.options.lever_arm_m")
        if lever_arm <= 0.0:
            raise BackendError("INVALID_SCHEMA", "lever_arm_m must be positive.", field="metadata.options.lever_arm_m")
    if input_type == "raw_dual_imu":
        _validate_raw_sensor_block(metadata)
        calibration = _mapping(metadata.get("calibration", {}), "metadata.calibration")
        use_stored_profile = bool(calibration.get("use_stored_profile", False))
        if use_stored_profile:
            if not str(calibration.get("calibration_id", "") or "").strip():
                raise BackendError(
                    "CALIBRATION_REQUIRED",
                    "calibration.calibration_id is required when calibration.use_stored_profile is true.",
                    field="metadata.calibration.calibration_id",
                )
        else:
            _validate_calibration_segments(metadata)
    normalized = deepcopy(metadata)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["source_timestamp_basis"] = timestamp_basis
    normalized["timestamp_basis"] = "session_relative_ms"
    if participant_id is not None:
        normalized["participant_id"] = participant_id
    if condition is not None:
        normalized["condition"] = condition
        normalized["support_level"] = TRIAL_CONDITIONS[condition]["support_level"]
        normalized["reminder_enabled"] = TRIAL_CONDITIONS[condition]["reminder_enabled"]
    normalized["options"] = {
        **options,
        "enable_ml_shadow": True,
        "chunk_size": chunk_size,
        "threshold_version": threshold_version,
        "explanation_provider": provider,
        "enable_external_api": external_enabled,
        "generate_charts": bool(options.get("generate_charts", True)),
        "language": str(options.get("language", "zh-CN")),
        "personal_baseline_role": baseline_role,
    }
    return normalized


def _decode_csv(data: bytes, field: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BackendError("INVALID_SCHEMA", "CSV must be UTF-8 encoded.", field=field) from exc
    reader = csv.DictReader(io.StringIO(text))
    fields = list(reader.fieldnames or [])
    rows = list(reader)
    if not fields or not rows:
        raise BackendError("INVALID_SCHEMA", "CSV must include a header and at least one row.", field=field)
    return fields, rows


def _parse_number(row: Mapping[str, str], column: str, field: str, row_number: int) -> float:
    try:
        value = float(row.get(column, ""))
    except (TypeError, ValueError) as exc:
        raise BackendError("INVALID_SCHEMA", f"Invalid {column} at CSV row {row_number}.", field=field) from exc
    if not math.isfinite(value):
        raise BackendError("INVALID_SCHEMA", f"Non-finite {column} at CSV row {row_number}.", field=field)
    return value


def _validate_binary_column(rows: list[dict[str, str]], column: str, field: str) -> None:
    for row_number, row in enumerate(rows, start=2):
        value = _parse_number(row, column, field, row_number)
        if value not in {0.0, 1.0}:
            raise BackendError(
                "INVALID_SCHEMA",
                f"{column} must contain only 0 or 1.",
                field=field,
                details={"row": row_number, "column": column},
            )


def parse_raw_dual_imu(
    data: bytes,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]], float]:
    fields, rows = _decode_csv(data, "data_file")
    timestamp_field = "timestamp_ms" if "timestamp_ms" in fields else "device_ms" if "device_ms" in fields else None
    required = {"sensor_id", "ax", "ay", "az", "gx", "gy", "gz"}
    missing = sorted(required - set(fields))
    if timestamp_field is None:
        missing.insert(0, "timestamp_ms or device_ms")
    if missing:
        raise BackendError("INVALID_SCHEMA", "Raw IMU CSV is missing required columns.", field="data_file", details={"missing": missing})
    fsr_fields = [field for field in ("fsr_raw_adc", "fsr_raw") if field in fields]
    if len(fsr_fields) > 1:
        raise BackendError("INVALID_SCHEMA", "Use only one of fsr_raw_adc or fsr_raw.", field="data_file")
    fsr_field = fsr_fields[0] if fsr_fields else None
    mag_fields = {"mx", "my", "mz"}
    if mag_fields & set(fields) and not mag_fields <= set(fields):
        raise BackendError("INVALID_SCHEMA", "mx, my, and mz must be supplied together.", field="data_file")
    grouped: dict[str, dict[str, list[object]]] = {
        node: {"time": [], "accel": [], "gyro": [], "mag": [], "quality": []} for node in ("forearm", "hand")
    }
    has_mag = mag_fields <= set(fields)
    has_quality = "quality" in fields
    fsr_samples: dict[float, float] = {}
    for row_number, row in enumerate(rows, start=2):
        node = str(row.get("sensor_id", "")).strip()
        if node not in grouped:
            raise BackendError("MISSING_SENSOR_NODE", f"Unsupported sensor_id at CSV row {row_number}.", field="data_file", details={"sensor_id": node})
        target = grouped[node]
        timestamp_ms = _parse_number(row, str(timestamp_field), "data_file", row_number)
        target["time"].append(timestamp_ms / 1000.0)
        target["accel"].append([_parse_number(row, key, "data_file", row_number) for key in ("ax", "ay", "az")])
        target["gyro"].append([_parse_number(row, key, "data_file", row_number) for key in ("gx", "gy", "gz")])
        if has_mag:
            target["mag"].append([_parse_number(row, key, "data_file", row_number) for key in ("mx", "my", "mz")])
        if has_quality:
            quality = _parse_number(row, "quality", "data_file", row_number)
            if quality < 0.0 or quality > 1.0:
                raise BackendError("INVALID_SCHEMA", "quality must be within 0..1.", field="data_file", details={"row": row_number})
            target["quality"].append(quality)
        if fsr_field and row.get(fsr_field, "") not in {"", None}:
            fsr_value = _parse_number(row, fsr_field, "data_file", row_number)
            if fsr_value < 0.0:
                raise BackendError("INVALID_SCHEMA", f"{fsr_field} must be non-negative.", field="data_file", details={"row": row_number})
            previous = fsr_samples.get(timestamp_ms)
            if previous is not None and not math.isclose(previous, fsr_value, rel_tol=0.0, abs_tol=1e-9):
                raise BackendError("INVALID_SCHEMA", "Conflicting FSR values share one timestamp.", field="data_file", details={"row": row_number})
            fsr_samples[timestamp_ms] = fsr_value
    missing_nodes = [node for node, values in grouped.items() if not values["time"]]
    if missing_nodes:
        raise BackendError(
            "MISSING_SENSOR_NODE",
            "Raw IMU CSV must contain both forearm and hand samples.",
            field="data_file",
            details={"missing": missing_nodes},
        )
    time_origin_s = min(float(values["time"][0]) for values in grouped.values()) if timestamp_field == "device_ms" else 0.0
    raw: dict[str, np.ndarray] = {}
    supplied_quality: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rates = []
    for node, values in grouped.items():
        timestamp = np.asarray(values["time"], dtype=float) - time_origin_s
        if len(timestamp) < 10:
            raise BackendError("INSUFFICIENT_VALID_DATA", f"{node} requires at least 10 samples.", field="data_file")
        if np.any(np.diff(timestamp) <= 0.0):
            raise BackendError("NON_MONOTONIC_TIMESTAMP", f"{node} timestamps must be strictly increasing.", field="data_file")
        raw[f"{node}_timestamp_s"] = timestamp
        raw[f"{node}_accel"] = np.asarray(values["accel"], dtype=float)
        raw[f"{node}_gyro"] = np.asarray(values["gyro"], dtype=float)
        if has_mag:
            raw[f"{node}_mag"] = np.asarray(values["mag"], dtype=float)
        if has_quality:
            supplied_quality[node] = (timestamp, np.asarray(values["quality"], dtype=float))
        rates.append(1.0 / float(np.median(np.diff(timestamp))))
    if fsr_samples:
        ordered = sorted(fsr_samples.items())
        raw["fsr_timestamp_s"] = np.asarray([item[0] / 1000.0 - time_origin_s for item in ordered], dtype=float)
        raw["fsr_raw_adc"] = np.asarray([item[1] for item in ordered], dtype=float)
    return raw, supplied_quality, float(np.median(rates))


def normalize_joint_state(path: Path, data: bytes) -> dict[str, np.ndarray]:
    fields, rows = _decode_csv(data, "data_file")
    required = {"timestamp_ms", "theta_FE", "theta_RUD"}
    missing = sorted(required - set(fields))
    if missing:
        raise BackendError("INVALID_SCHEMA", "Joint-state CSV is missing required columns.", field="data_file", details={"missing": missing})
    timestamp = np.asarray([_parse_number(row, "timestamp_ms", "data_file", index) for index, row in enumerate(rows, start=2)])
    if len(timestamp) < 2 or np.any(np.diff(timestamp) <= 0.0):
        raise BackendError("NON_MONOTONIC_TIMESTAMP", "timestamp_ms must be strictly increasing.", field="data_file")
    joint = load_joint_state_csv(path)
    for key in ("theta_FE", "theta_RUD"):
        if np.any(~np.isfinite(np.asarray(joint[key], dtype=float))):
            raise BackendError("INVALID_SCHEMA", f"{key} must contain finite values.", field="data_file")
    count = len(timestamp)
    if "quality" not in joint:
        joint["quality"] = np.ones(count, dtype=float)
    else:
        quality = np.asarray(joint["quality"], dtype=float)
        finite = np.isfinite(quality)
        if np.any((quality[finite] < 0.0) | (quality[finite] > 1.0)):
            raise BackendError("INVALID_SCHEMA", "quality must be within 0..1.", field="data_file")
        joint["quality"] = np.where(finite, quality, 0.0)
    if "theta_thumb" not in joint:
        joint["theta_thumb"] = np.full(count, np.nan)
    if "angular_velocity" not in joint:
        time_s = timestamp / 1000.0
        joint["angular_velocity"] = np.sqrt(np.gradient(joint["theta_FE"], time_s) ** 2 + np.gradient(joint["theta_RUD"], time_s) ** 2)
    if "calibration_id" not in joint:
        joint["calibration_id"] = np.full(count, "provided-by-input", dtype=object)
    return joint


def calibration_from_metadata(metadata: Mapping[str, object]) -> tuple[tuple[float, float], list[dict[str, object]]]:
    segments = metadata["calibration"]["segments"]
    neutral_segments = [segment for segment in segments if segment["type"] == "neutral"]
    neutral = max(neutral_segments, key=lambda item: float(item["end_ms"]) - float(item["start_ms"]))
    labels = {
        "flexion": "Flexion",
        "extension": "Extension",
        "radial_deviation": "Radial Deviation",
        "ulnar_deviation": "Ulnar Deviation",
    }
    annotations = [
        {
            "Type": labels[segment["type"]],
            "Segment": "wrist",
            "Init": float(segment["start_ms"]) / 1000.0,
            "End": float(segment["end_ms"]) / 1000.0,
        }
        for segment in segments
        if segment["type"] in labels
    ]
    return (float(neutral["start_ms"]) / 1000.0, float(neutral["end_ms"]) / 1000.0), annotations


def joint_rows(joint: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    fields = ("timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality")
    rows = []
    for index in range(len(joint["timestamp_ms"])):
        row: dict[str, object] = {}
        for field in fields:
            value = joint[field][index]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                row[field] = ""
            else:
                row[field] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return rows


def _align_optional_series(
    target_timestamp_ms: np.ndarray,
    source_timestamp_s: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    return np.interp(
        np.asarray(target_timestamp_ms, dtype=float) / 1000.0,
        np.asarray(source_timestamp_s, dtype=float),
        np.asarray(source_values, dtype=float),
        left=np.nan,
        right=np.nan,
    )


def _fsr_proxy_summary(values: Optional[np.ndarray], source: Optional[str]) -> dict[str, object]:
    if values is None:
        return {
            "available": False,
            "source": None,
            "unit": None,
            "calibrated_to_pressure": False,
            "mean": None,
            "p95": None,
            "max": None,
        }
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "available": False,
            "source": source,
            "unit": "normalized_pct" if source == "fsr_normalized_pct" else "adc_count",
            "calibrated_to_pressure": False,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "available": True,
        "source": source,
        "unit": "normalized_pct" if source == "fsr_normalized_pct" else "adc_count",
        "calibrated_to_pressure": False,
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _mechanical_channels(
    joint: Mapping[str, np.ndarray],
    path: Optional[Path],
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    dict[str, object],
]:
    channels = {
        "pressure": False,
        "fsr_raw": False,
        "tension": False,
        "discomfort": False,
        "discomfort_nrs": False,
        "safety_symptom": False,
        "user_continues": False,
    }
    if path is None:
        return None, None, None, None, None, None, None, None, channels
    mechanical = load_mechanical_csv(path)
    source_time = np.asarray(mechanical["timestamp_ms"], dtype=float)
    if len(source_time) < 2 or np.any(~np.isfinite(source_time)) or np.any(np.diff(source_time) <= 0.0):
        raise BackendError("NON_MONOTONIC_TIMESTAMP", "Mechanical timestamps must be finite and strictly increasing.", field="mechanical_file")
    aligned = align_mechanical_to_joint(np.asarray(joint["timestamp_ms"], dtype=float), mechanical)
    pressure_columns = [key for key in ("p_radial_kPa", "p_dorsal_kPa", "p_ulnar_kPa") if key in aligned]
    pressure = None
    if pressure_columns:
        values = np.column_stack([aligned[key] for key in pressure_columns])
        if np.any(values[np.isfinite(values)] < 0.0):
            raise BackendError("INVALID_SCHEMA", "Pressure values must be non-negative.", field="mechanical_file")
        finite = np.isfinite(values)
        pressure = np.max(np.where(finite, values, -np.inf), axis=1)
        pressure[~np.any(finite, axis=1)] = np.nan
        channels["pressure"] = bool(np.any(np.isfinite(pressure)))
    fsr_raw = aligned.get("fsr_raw_adc")
    if fsr_raw is None:
        fsr_raw = aligned.get("fsr_normalized_pct")
    if fsr_raw is not None:
        fsr_raw = np.asarray(fsr_raw, dtype=float)
        channels["fsr_raw"] = bool(np.any(np.isfinite(fsr_raw)))
        channels["fsr_source"] = "fsr_normalized_pct" if "fsr_normalized_pct" in aligned else "fsr_raw_adc"
    tension = aligned.get("cable_tension_N")
    if tension is not None:
        finite_tension = np.asarray(tension)[np.isfinite(tension)]
        if np.any(finite_tension < 0.0):
            raise BackendError("INVALID_SCHEMA", "Cable tension must be non-negative.", field="mechanical_file")
        channels["tension"] = bool(len(finite_tension))
    discomfort = aligned.get("discomfort")
    if discomfort is not None:
        discomfort = np.asarray(discomfort, dtype=float)
        if np.any(~np.isfinite(discomfort)):
            raise BackendError("INVALID_SCHEMA", "discomfort must cover the complete joint-state time range.", field="mechanical_file")
        channels["discomfort"] = True
        discomfort = discomfort == 1.0
    discomfort_nrs = aligned.get("discomfort_nrs")
    if discomfort_nrs is not None:
        discomfort_nrs = np.asarray(discomfort_nrs, dtype=float)
        channels["discomfort_nrs"] = bool(np.any(np.isfinite(discomfort_nrs)))
    safety_symptom = aligned.get("safety_symptom_flag")
    if safety_symptom is not None:
        safety_symptom = np.asarray(safety_symptom, dtype=float)
        if np.any(~np.isfinite(safety_symptom)):
            raise BackendError("INVALID_SCHEMA", "safety_symptom_flag must cover the complete joint-state time range.", field="mechanical_file")
        channels["safety_symptom"] = True
        safety_symptom = safety_symptom == 1.0
    if discomfort is None:
        safety_trigger = safety_symptom
    elif safety_symptom is None:
        safety_trigger = discomfort
    else:
        safety_trigger = discomfort | safety_symptom
    user_continues = aligned.get("user_continues")
    if user_continues is not None:
        user_continues = np.asarray(user_continues, dtype=float)
        if np.any(~np.isfinite(user_continues)):
            raise BackendError("INVALID_SCHEMA", "user_continues must cover the complete joint-state time range.", field="mechanical_file")
        channels["user_continues"] = True
        user_continues = user_continues == 1.0
    return pressure, fsr_raw, tension, safety_trigger, discomfort, discomfort_nrs, safety_symptom, user_continues, channels


class AnalysisService:
    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings
        self.settings.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = self.settings.output_root / "_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.baselines_root = self.settings.output_root / "_baselines"
        self.baselines_root.mkdir(parents=True, exist_ok=True)
        self.calibrations_root = self.settings.output_root / "_calibrations"
        self.calibrations_root.mkdir(parents=True, exist_ok=True)
        self.algorithm_config = load_config(settings.algorithm_config)
        self.ml_config = load_config(settings.ml_config)
        self.base_explanation_config = load_config(settings.explanation_config)
        self.pipeline = ShadowActivityPipeline.load(settings.model_path)
        self._lock = threading.RLock()

    def _job_path(self, job_id: str) -> Path:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise BackendError("JOB_NOT_FOUND", "Analysis job was not found.", http_status=404)
        return self.jobs_root / f"{job_id}.json"

    def _session_dir(self, session_id: str) -> Path:
        return self.settings.output_root / session_id

    def _participant_dir(self, participant_id: str) -> Path:
        return self.baselines_root / participant_id

    def _load_history(self, participant_id: str) -> list[dict[str, object]]:
        path = self._participant_dir(participant_id) / "history.json"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        sessions = payload.get("sessions", []) if isinstance(payload, Mapping) else []
        return [dict(item) for item in sessions if isinstance(item, Mapping)]

    def _save_history(self, participant_id: str, sessions: list[dict[str, object]]) -> None:
        _atomic_json(
            self._participant_dir(participant_id) / "history.json",
            {"schema_version": SCHEMA_VERSION, "participant_id": participant_id, "sessions": sessions},
        )

    def _calibration_path(self, calibration_id: str) -> Path:
        if not SESSION_ID_PATTERN.fullmatch(str(calibration_id)):
            raise BackendError("CALIBRATION_NOT_FOUND", "Calibration profile was not found.", http_status=404)
        return self.calibrations_root / f"{calibration_id}.json"

    def _load_calibration_record(self, calibration_id: str) -> dict[str, object]:
        path = self._calibration_path(calibration_id)
        if not path.exists():
            raise BackendError(
                "CALIBRATION_NOT_FOUND",
                "Calibration profile was not found.",
                field="metadata.calibration.calibration_id",
                details={"calibration_id": str(calibration_id)},
                http_status=404,
            )
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _public_calibration(self, record: Mapping[str, object]) -> dict[str, object]:
        profile = record.get("calibration", {})
        calibration_id = str(record["calibration_id"])
        return {
            "schema_version": SCHEMA_VERSION,
            "calibration_id": calibration_id,
            "participant_id": record.get("participant_id"),
            "status": record.get("status"),
            "created_at": record.get("created_at"),
            "sample_rate_hz": record.get("sample_rate_hz"),
            "quality_gate_passed": profile.get("quality_gate_passed"),
            "quality_reasons": list(profile.get("quality_reasons", [])),
            "algorithm": profile.get("algorithm"),
            "neutral_stationary_sample_pct": profile.get("neutral_stationary_sample_pct"),
            "neutral_stationary_sample_pct_min": profile.get("neutral_stationary_sample_pct_min"),
            "flexion_extension_axis": profile.get("flexion_extension_axis"),
            "radial_ulnar_axis": profile.get("radial_ulnar_axis"),
            "pronation_supination_axis": profile.get("pronation_supination_axis"),
            "neutral_quaternion": profile.get("neutral_quaternion"),
            "self_url": f"/api/v1/calibrations/{calibration_id}",
        }

    def create_calibration(
        self,
        metadata_payload: object,
        calibration: bytes,
        calibration_filename: str,
    ) -> dict[str, object]:
        metadata = validate_calibration_metadata(metadata_payload)
        if not calibration:
            raise BackendError("INVALID_SCHEMA", "calibration_file is empty.", field="calibration_file")
        if not str(calibration_filename).lower().endswith(".csv"):
            raise BackendError("INVALID_SCHEMA", "calibration_file must be a CSV file.", field="calibration_file")
        calibration_id = str(metadata["calibration_id"])
        raw, _, sample_rate = parse_raw_dual_imu(calibration)
        neutral, annotations = calibration_from_metadata(metadata)
        with self._lock:
            path = self._calibration_path(calibration_id)
            if path.exists():
                raise BackendError(
                    "CALIBRATION_EXISTS",
                    "calibration_id already exists.",
                    field="metadata.calibration_id",
                    http_status=409,
                )
            _, audit = prepare_joint_state_from_raw(
                raw,
                neutral,
                annotations,
                self.algorithm_config,
                calibration_id,
                sample_rate_hz=sample_rate,
                initialize_from_accel=True,
            )
            profile = audit["calibration"]
            if not profile.get("quality_gate_passed", False):
                raise BackendError(
                    "CALIBRATION_QUALITY_FAILED",
                    "Calibration recording did not pass the quality gate; re-record the neutral and functional segments.",
                    field="calibration_file",
                    details={"quality_reasons": list(profile.get("quality_reasons", []))},
                    http_status=422,
                )
            record = {
                "schema_version": SCHEMA_VERSION,
                "calibration_id": calibration_id,
                "participant_id": metadata.get("participant_id"),
                "status": profile.get("status", "passed"),
                "created_at": utc_now(),
                "input_sha256": hashlib.sha256(calibration).hexdigest(),
                "sample_rate_hz": float(sample_rate),
                "calibration": profile,
                "synchronization": audit.get("synchronization"),
            }
            _atomic_json(path, record)
        return self._public_calibration(record)

    def get_calibration(self, calibration_id: str) -> dict[str, object]:
        return self._public_calibration(self._load_calibration_record(calibration_id))

    def _personal_baseline_report(
        self,
        metadata: Mapping[str, object],
        joint: Mapping[str, np.ndarray],
        discomfort_nrs: Optional[np.ndarray],
    ) -> dict[str, object]:
        participant_id = str(metadata["participant_id"])
        options = metadata["options"]
        config = self.algorithm_config
        pae = config["personal_baseline"]
        tracked = list(pae["tracked_metrics"])
        primary_metric = str(pae["symptom"]["primary_metric"])
        min_valid_pct = float(pae["enrollment"]["min_valid_sample_pct"])
        role = str(options.get("personal_baseline_role", "auto"))
        target_value = options.get("baseline_target_reduction_pct")
        target_reduction = float(target_value) if target_value is not None else None

        timestamp_s = np.asarray(joint["timestamp_ms"], dtype=float) / 1000.0
        summary = session_exposure_summary(
            timestamp_s,
            np.asarray(joint["theta_FE"], dtype=float),
            np.asarray(joint["theta_RUD"], dtype=float),
            config,
            quality=np.asarray(joint["quality"], dtype=float) if "quality" in joint else None,
        )

        session_pain: Optional[float] = None
        if discomfort_nrs is not None:
            finite = np.asarray(discomfort_nrs, dtype=float)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                session_pain = float(np.mean(finite))

        with self._lock:
            baseline_path = self._participant_dir(participant_id) / "baseline.json"
            existing = load_personal_baseline(baseline_path) if baseline_path.exists() else None
            prior = existing if existing is not None and existing.status != "rejected" else None

            low_quality = summary["valid_minutes"] <= 0.0 or float(summary["valid_sample_pct"]) < min_valid_pct
            if low_quality:
                reference = prior if prior is not None else PersonalBaseline(
                    participant_id, {name: None for name in tracked}, 0.0, 0, "not_established", []
                )
                report = build_personal_report(reference, summary, config, None, None, target_reduction)
                report["personal_baseline_role"] = role
                report["update_applied"] = False
                report["update_skip_reason"] = "low_quality_session"
                return report

            now = utc_now()
            persist = True
            if role == "enroll":
                candidate = init_personal_baseline(participant_id, summary, config, updated_at=now)
                if candidate.status == "rejected" and prior is not None:
                    baseline = prior
                    persist = False
                else:
                    baseline = candidate
            elif prior is not None:
                baseline = update_personal_baseline(prior, summary, config, updated_at=now)
            else:
                bootstrap = PersonalBaseline(
                    participant_id, {name: None for name in tracked}, 0.0, 0, "provisional", []
                )
                baseline = update_personal_baseline(bootstrap, summary, config, updated_at=now)

            if persist:
                save_personal_baseline(baseline_path, baseline)

            sessions = [item for item in self._load_history(participant_id) if item.get("session_id") != str(metadata["session_id"])]
            sessions.append(
                {
                    "session_id": str(metadata["session_id"]),
                    "updated_at": now,
                    "exposure": summary["metrics"].get(primary_metric),
                    "pain": session_pain,
                    "valid_minutes": summary["valid_minutes"],
                }
            )
            self._save_history(participant_id, sessions)

            paired = [
                (item["exposure"], item["pain"])
                for item in sessions
                if item.get("exposure") is not None and item.get("pain") is not None
            ]
            exposure_series = [float(item[0]) for item in paired] if paired else None
            pain_series = [float(item[1]) for item in paired] if paired else None

            report_reference = prior if prior is not None else baseline
            report = build_personal_report(
                report_reference, summary, config, exposure_series, pain_series, target_reduction
            )
            report["baseline"] = baseline.to_dict()
            report["baseline_status"] = baseline.status
            report["personal_baseline_role"] = role
            report["update_applied"] = persist
            return report

    def _load_job(self, job_id: str) -> dict[str, object]:
        path = self._job_path(job_id)
        if not path.exists():
            raise BackendError("JOB_NOT_FOUND", "Analysis job was not found.", http_status=404)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_job(self, record: Mapping[str, object]) -> None:
        _atomic_json(self._job_path(str(record["job_id"])), dict(record))

    def _update_job(self, job_id: str, **updates: object) -> dict[str, object]:
        with self._lock:
            record = self._load_job(job_id)
            record.update(updates)
            record["updated_at"] = utc_now()
            self._save_job(record)
            return record

    def _find_idempotency(self, key_hash: str) -> Optional[dict[str, object]]:
        for path in self.jobs_root.glob("job_*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("idempotency_key_sha256") == key_hash:
                return record
        return None

    def create_job(
        self,
        metadata_payload: object,
        data: bytes,
        data_filename: str,
        mechanical: Optional[bytes] = None,
        mechanical_filename: Optional[str] = None,
        calibration: Optional[bytes] = None,
        calibration_filename: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, object]:
        metadata = validate_metadata(metadata_payload)
        if not data:
            raise BackendError("INVALID_SCHEMA", "data_file is empty.", field="data_file")
        if not str(data_filename).lower().endswith(".csv"):
            raise BackendError("INVALID_SCHEMA", "data_file must be a CSV file.", field="data_file")
        if mechanical is not None and not str(mechanical_filename or "").lower().endswith(".csv"):
            raise BackendError("INVALID_SCHEMA", "mechanical_file must be a CSV file.", field="mechanical_file")
        if calibration is not None and not str(calibration_filename or "").lower().endswith(".csv"):
            raise BackendError("INVALID_SCHEMA", "calibration_file must be a CSV file.", field="calibration_file")
        if calibration is not None and metadata["input_type"] != "raw_dual_imu":
            raise BackendError("INVALID_SCHEMA", "calibration_file is only valid with raw_dual_imu.", field="calibration_file")
        calibration_meta = metadata.get("calibration") if isinstance(metadata.get("calibration"), Mapping) else {}
        use_stored_profile = bool(calibration_meta.get("use_stored_profile", False)) and metadata["input_type"] == "raw_dual_imu"
        if use_stored_profile:
            if calibration is not None:
                raise BackendError(
                    "INVALID_SCHEMA",
                    "calibration_file must not be supplied when calibration.use_stored_profile is true.",
                    field="calibration_file",
                )
            self._load_calibration_record(str(calibration_meta.get("calibration_id")))
        elif metadata.get("condition") is not None and metadata["input_type"] == "raw_dual_imu" and calibration is None:
            raise BackendError("CALIBRATION_REQUIRED", "A/B/C raw_dual_imu sessions require the separate CAL/validation file or a stored calibration profile.", field="calibration_file")
        if metadata["input_type"] == "raw_dual_imu":
            parse_raw_dual_imu(data)
            if calibration is not None:
                parse_raw_dual_imu(calibration)
        else:
            _decode_csv(data, "data_file")
        if mechanical is not None:
            mechanical_fields, mechanical_rows = _decode_csv(mechanical, "mechanical_file")
            timestamp_field = "timestamp_ms" if "timestamp_ms" in mechanical_fields else "device_ms" if "device_ms" in mechanical_fields else None
            if timestamp_field is None:
                raise BackendError("INVALID_SCHEMA", "Mechanical CSV must include timestamp_ms or device_ms.", field="mechanical_file")
            mechanical_time = np.asarray([
                _parse_number(row, timestamp_field, "mechanical_file", index)
                for index, row in enumerate(mechanical_rows, start=2)
            ])
            if len(mechanical_time) < 2 or np.any(np.diff(mechanical_time) <= 0.0):
                raise BackendError("NON_MONOTONIC_TIMESTAMP", "Mechanical timestamps must be strictly increasing.", field="mechanical_file")
            for column in ("discomfort", "safety_symptom_flag", "user_continues"):
                if column in mechanical_fields:
                    _validate_binary_column(mechanical_rows, column, "mechanical_file")
            if metadata.get("condition") is not None and "condition" in mechanical_fields:
                observed_conditions = {str(row.get("condition", "")).strip().upper() for row in mechanical_rows}
                if observed_conditions != {str(metadata["condition"])}:
                    raise BackendError("INVALID_TRIAL_CONDITION", "mechanical_file condition does not match metadata.", field="mechanical_file", details={"observed": sorted(observed_conditions), "expected": metadata["condition"]})
            if metadata.get("condition") is not None and "support_level" in mechanical_fields:
                observed_support = {
                    int(_parse_number(row, "support_level", "mechanical_file", row_number))
                    for row_number, row in enumerate(mechanical_rows, start=2)
                }
                if observed_support != {int(metadata["support_level"])}:
                    raise BackendError("INVALID_TRIAL_CONDITION", "mechanical_file support_level does not match metadata.", field="mechanical_file", details={"observed": sorted(observed_support), "expected": metadata["support_level"]})
            if "discomfort_nrs" in mechanical_fields:
                for row_number, row in enumerate(mechanical_rows, start=2):
                    if row.get("discomfort_nrs", "") not in {"", None}:
                        value = _parse_number(row, "discomfort_nrs", "mechanical_file", row_number)
                        if value < 0.0 or value > 10.0:
                            raise BackendError("INVALID_SCHEMA", "discomfort_nrs must be within 0..10.", field="mechanical_file", details={"row": row_number})
            if "fsr_raw" in mechanical_fields and "fsr_raw_adc" in mechanical_fields:
                raise BackendError("INVALID_SCHEMA", "Use only one of fsr_raw or fsr_raw_adc.", field="mechanical_file")
            for column, maximum in (("fsr_raw", None), ("fsr_raw_adc", None), ("fsr_normalized_pct", 100.0)):
                if column in mechanical_fields:
                    for row_number, row in enumerate(mechanical_rows, start=2):
                        if row.get(column, "") not in {"", None}:
                            value = _parse_number(row, column, "mechanical_file", row_number)
                            if value < 0.0 or (maximum is not None and value > maximum):
                                message = f"{column} must be non-negative." if maximum is None else f"{column} must be within 0..{maximum:g}."
                                raise BackendError("INVALID_SCHEMA", message, field="mechanical_file", details={"row": row_number})
        request_fingerprint = hashlib.sha256(
            data + (mechanical or b"") + (calibration or b"") + _json_fingerprint(metadata).encode("ascii")
        ).hexdigest()
        key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest() if idempotency_key else None
        with self._lock:
            if key_hash:
                existing = self._find_idempotency(key_hash)
                if existing:
                    if existing.get("request_fingerprint") != request_fingerprint:
                        raise BackendError("IDEMPOTENCY_CONFLICT", "Idempotency-Key was already used for a different request.", http_status=409)
                    payload = self.job_payload(existing)
                    payload["idempotent_replay"] = True
                    return payload
            session_dir = self._session_dir(str(metadata["session_id"]))
            if session_dir.exists():
                raise BackendError("SESSION_EXISTS", "session_id already exists.", field="metadata.session_id", http_status=409)
            session_dir.mkdir(parents=True)
            write_json(session_dir / "metadata.json", metadata)
            (session_dir / "input.csv").write_bytes(data)
            if mechanical is not None:
                (session_dir / "mechanical.csv").write_bytes(mechanical)
            if calibration is not None:
                (session_dir / "calibration.csv").write_bytes(calibration)
            job_id = f"job_{uuid.uuid4().hex[:20]}"
            now = utc_now()
            record: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "job_id": job_id,
                "session_id": metadata["session_id"],
                "status": "queued",
                "stage": "validation",
                "progress_pct": 0,
                "created_at": now,
                "updated_at": now,
                "idempotency_key_sha256": key_hash,
                "request_fingerprint": request_fingerprint,
            }
            self._save_job(record)
        return self.job_payload(record)

    def job_payload(self, record: Mapping[str, object]) -> dict[str, object]:
        payload = {
            key: record[key]
            for key in ("schema_version", "job_id", "session_id", "status", "stage", "progress_pct", "created_at", "updated_at")
            if key in record
        }
        session_id = str(record["session_id"])
        job_id = str(record["job_id"])
        payload["status_url"] = f"/api/v1/analysis-jobs/{job_id}"
        payload["result_url"] = f"/api/v1/sessions/{session_id}"
        if "error" in record:
            payload["error"] = record["error"]
        if "analysis_status" in record:
            payload["analysis_status"] = record["analysis_status"]
        return payload

    def get_job(self, job_id: str) -> dict[str, object]:
        return self.job_payload(self._load_job(job_id))

    def run_job(self, job_id: str) -> None:
        try:
            self._update_job(job_id, status="running", stage="validation", progress_pct=5)
            record = self._load_job(job_id)
            result = self._process_session(job_id, str(record["session_id"]))
            self._update_job(job_id, status="succeeded", stage="reporting", progress_pct=100, analysis_status=result["analysis_status"])
        except BackendError as exc:
            self._update_job(job_id, status="failed", progress_pct=100, error=exc.payload()["error"])
        except Exception as exc:
            error = BackendError(
                "ANALYSIS_FAILED",
                str(exc),
                retryable=False,
                details={"exception_type": type(exc).__name__},
                http_status=422,
            )
            self._update_job(job_id, status="failed", progress_pct=100, error=error.payload()["error"])

    def _process_session(self, job_id: str, session_id: str) -> dict[str, object]:
        session_dir = self._session_dir(session_id)
        with (session_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        input_path = session_dir / "input.csv"
        input_bytes = input_path.read_bytes()
        input_type = str(metadata["input_type"])
        options = metadata["options"]
        preprocessing: dict[str, object]
        if input_type == "raw_dual_imu":
            raw, supplied_quality, sample_rate = parse_raw_dual_imu(input_bytes)
            calibration_meta = metadata.get("calibration") if isinstance(metadata.get("calibration"), Mapping) else {}
            use_stored_profile = bool(calibration_meta.get("use_stored_profile", False))
            self._update_job(job_id, stage="synchronization", progress_pct=20)
            calibration_path = session_dir / "calibration.csv"
            if use_stored_profile:
                calibration_id = str(calibration_meta["calibration_id"])
                stored = self._load_calibration_record(calibration_id)
                joint, audit = prepare_joint_state_from_calibration_profile(
                    raw,
                    stored["calibration"],
                    self.algorithm_config,
                    calibration_id,
                    sample_rate_hz=sample_rate,
                )
                audit["calibration"]["application_mode"] = "stored_calibration_profile"
                audit["calibration"]["source_calibration_id"] = calibration_id
                audit["calibration"]["source_calibration_created_at"] = stored.get("created_at")
            elif calibration_path.exists():
                neutral, annotations = calibration_from_metadata(metadata)
                calibration_id = str(metadata.get("calibration", {}).get("calibration_id") or f"{session_id}-{_json_fingerprint(metadata['calibration'])[:12]}")
                calibration_raw, _, calibration_rate = parse_raw_dual_imu(calibration_path.read_bytes())
                _, calibration_audit = prepare_joint_state_from_raw(
                    calibration_raw,
                    neutral,
                    annotations,
                    self.algorithm_config,
                    calibration_id,
                    sample_rate_hz=calibration_rate,
                    initialize_from_accel=True,
                )
                joint, audit = prepare_joint_state_from_calibration_profile(
                    raw,
                    calibration_audit["calibration"],
                    self.algorithm_config,
                    calibration_id,
                    sample_rate_hz=sample_rate,
                )
                audit["calibration"]["source_file"] = "calibration.csv"
                audit["calibration"]["source_profile_quality_gate_passed"] = calibration_audit["calibration"]["quality_gate_passed"]
                audit["calibration_source_synchronization"] = calibration_audit["synchronization"]
            else:
                neutral, annotations = calibration_from_metadata(metadata)
                calibration_id = str(metadata.get("calibration", {}).get("calibration_id") or f"{session_id}-{_json_fingerprint(metadata['calibration'])[:12]}")
                joint, audit = prepare_joint_state_from_raw(
                    raw,
                    neutral,
                    annotations,
                    self.algorithm_config,
                    calibration_id,
                    sample_rate_hz=sample_rate,
                )
                audit["calibration"]["application_mode"] = "embedded_in_data_file"
            if supplied_quality:
                target = np.asarray(joint["timestamp_ms"], dtype=float) / 1000.0
                source_quality = [np.interp(target, *supplied_quality[node]) for node in ("forearm", "hand")]
                joint["quality"] = np.minimum.reduce((np.asarray(joint["quality"]), *source_quality))
            embedded_fsr = (
                _align_optional_series(joint["timestamp_ms"], raw["fsr_timestamp_s"], raw["fsr_raw_adc"])
                if "fsr_raw_adc" in raw
                else None
            )
            preprocessing = {
                **audit,
                "source": {
                    "type": "raw_dual_imu_csv",
                    "quality_source": "provided_and_algorithm_minimum" if supplied_quality else "algorithm_derived",
                    "ground_truth_limit": "Uploaded IMU streams do not provide independent angle, calibrated pressure, pain, or clinical truth.",
                },
            }
        else:
            embedded_fsr = None
            joint = normalize_joint_state(input_path, input_bytes)
            calibration_ids = [str(value) for value in joint.get("calibration_id", []) if str(value)]
            preprocessing = {
                "source": {"type": "joint_state_csv"},
                "calibration": {
                    "status": "provided_by_input",
                    "calibration_id": calibration_ids[0] if calibration_ids else "provided-by-input",
                },
            }
        duration_s = (float(joint["timestamp_ms"][-1]) - float(joint["timestamp_ms"][0])) / 1000.0
        required_duration = float(self.ml_config["window"]["window_seconds"])
        if duration_s < required_duration:
            raise BackendError(
                "INSUFFICIENT_VALID_DATA",
                "Session is shorter than one ML analysis window.",
                field="data_file",
                details={"duration_s": duration_s, "required_duration_s": required_duration},
                http_status=422,
            )
        mechanical_path = session_dir / "mechanical.csv"
        pressure, fsr_raw, tension, safety_trigger, reported_discomfort, discomfort_nrs, safety_symptom, user_continues, mechanical_channels = _mechanical_channels(
            joint,
            mechanical_path if mechanical_path.exists() else None,
        )
        if embedded_fsr is not None:
            if fsr_raw is not None:
                raise BackendError("INVALID_SCHEMA", "FSR raw data must be supplied in either data_file or mechanical_file, not both.", field="mechanical_file")
            fsr_raw = embedded_fsr
            mechanical_channels["fsr_raw"] = bool(np.any(np.isfinite(fsr_raw)))
            mechanical_channels["fsr_source"] = "fsr_raw_adc"
        fsr_summary = _fsr_proxy_summary(fsr_raw, str(mechanical_channels.get("fsr_source") or "") or None)
        lever_arm = options.get("lever_arm_m")
        explanation_config = deepcopy(self.base_explanation_config)
        provider = str(options["explanation_provider"])
        explanation_config["provider"] = "template" if provider == "local_template" else provider
        explanation_config["enabled"] = bool(options["enable_external_api"])
        explanation_config["language"] = str(options["language"])
        if options.get("explanation_model"):
            explanation_config["model"] = str(options["explanation_model"])
        self._update_job(job_id, stage="deterministic_analysis", progress_pct=50)
        analysis, timeline = analyze_session(
            joint,
            self.algorithm_config,
            self.ml_config,
            self.pipeline,
            explanation_config,
            session_id,
            str(metadata["evidence_type"]),
            int(options["chunk_size"]),
            pressure_kpa=pressure,
            discomfort=safety_trigger,
            cable_tension_n=tension,
            lever_arm_m=float(lever_arm) if lever_arm is not None else None,
            user_continues=user_continues,
            angle_alerts_enabled=bool(metadata.get("reminder_enabled", True)),
            mechanical_recommendations_enabled="condition" not in metadata,
        )
        analysis["preprocessing"] = preprocessing
        safety_channel_available = bool(
            mechanical_channels["pressure"]
            or mechanical_channels["discomfort"]
            or mechanical_channels["safety_symptom"]
        )
        for index, row in enumerate(timeline):
            row["discomfort"] = None if reported_discomfort is None else bool(reported_discomfort[index])
            row["fsr_raw"] = None if fsr_raw is None or not np.isfinite(fsr_raw[index]) else float(fsr_raw[index])
            row["discomfort_nrs"] = (
                None if discomfort_nrs is None or not np.isfinite(discomfort_nrs[index]) else float(discomfort_nrs[index])
            )
            row["safety_symptom"] = None if safety_symptom is None else bool(safety_symptom[index])
        if not mechanical_channels["pressure"]:
            for row in timeline:
                row["pressure_zone"] = None
        if not safety_channel_available:
            for row in timeline:
                row["safety_stop"] = None
        self._update_job(job_id, stage="ml_finalize", progress_pct=75)
        joint_path = session_dir / "joint_state.csv"
        timeline_path = session_dir / "timeline.csv"
        analysis_path = session_dir / "analysis.json"
        tokens_path = session_dir / "tokens.json"
        write_csv(joint_path, ["timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality"], joint_rows(joint))
        write_csv(
            timeline_path,
            [
                "timestamp_ms",
                "theta_FE",
                "theta_RUD",
                "quality",
                "angle_zone",
                "pressure_zone",
                "discomfort",
                "user_continues",
                "activity_shadow",
                "alert",
                "would_alert",
                "alert_reason",
                "safety_stop",
                "fsr_raw",
                "discomfort_nrs",
                "safety_symptom",
            ],
            timeline,
        )
        write_json(analysis_path, analysis)
        write_json(tokens_path, {"schema_version": SCHEMA_VERSION, "operating_mode": "shadow", "tokens": analysis["ml_shadow"]["tokens"]})
        personal_report: Optional[dict[str, object]] = None
        personal_path: Optional[Path] = None
        if metadata.get("participant_id"):
            personal_report = self._personal_baseline_report(metadata, joint, discomfort_nrs)
            personal_path = session_dir / "personal_baseline.json"
            write_json(personal_path, personal_report)
        chart_paths: list[Path] = []
        warnings = []
        if fsr_summary["available"]:
            warnings.append("RFP-602 values are an uncalibrated contact-load proxy only; they are not N, kPa, tendon force, or tendon-sheath pressure.")
        if (session_dir / "calibration.csv").exists():
            warnings.append("Separate-file 6-axis reference assumes unchanged mounting and repeatable startup orientation; heading drift is not independently observable without calibrated magnetometer data or continuous fusion state.")
        if bool(options["generate_charts"]):
            try:
                chart_paths = plot_session_report(timeline, analysis, session_dir / "session_report")
            except RuntimeError as exc:
                warnings.append(str(exc))
        output_paths = [joint_path, timeline_path, analysis_path, tokens_path, *chart_paths]
        if personal_path is not None:
            output_paths.append(personal_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "job_id": job_id,
            "evidence_type": metadata["evidence_type"],
            "algorithm_release": ALGORITHM_RELEASE,
            "input_schema": input_type,
            "versions": {
                "threshold_schema": self.algorithm_config.get("schema_version"),
                "threshold_version": options["threshold_version"],
                "ml_schema": self.ml_config.get("schema_version"),
                "explanation_schema": explanation_config.get("schema_version"),
                "model_sha256": sha256_file(self.settings.model_path),
                "calibration_id": preprocessing.get("calibration", {}).get("calibration_id"),
            },
            "inputs": {
                "metadata.json": sha256_file(session_dir / "metadata.json"),
                "input.csv": sha256_file(input_path),
                **({"mechanical.csv": sha256_file(mechanical_path)} if mechanical_path.exists() else {}),
                **({"calibration.csv": sha256_file(session_dir / "calibration.csv")} if (session_dir / "calibration.csv").exists() else {}),
            },
            "configuration": {
                path.name: sha256_file(path)
                for path in (self.settings.algorithm_config, self.settings.ml_config, self.settings.explanation_config, self.settings.model_path)
            },
            "outputs": {path.name: sha256_file(path) for path in output_paths},
            "replay_acceptance": {
                "input_reconstruction_equal": analysis["replay"]["input_reconstruction_equal"],
                "deterministic_state_equal": analysis["replay"]["deterministic_state_equal"],
                "final_analysis_equal": analysis["replay"]["final_analysis_equal"],
            },
            "control_policy": analysis["control_policy"],
            "limitations": analysis["evidence_limits"],
            "warnings": warnings,
        }
        manifest["session_fingerprint"] = _json_fingerprint({"inputs": manifest["inputs"], "configuration": manifest["configuration"], "session_id": session_id})
        manifest_path = session_dir / "manifest.json"
        write_json(manifest_path, manifest)
        result = self._public_result(
            job_id,
            metadata,
            joint,
            analysis,
            preprocessing,
            mechanical_channels,
            fsr_summary,
            [*output_paths, manifest_path],
            warnings,
            personal_report,
        )
        _atomic_json(session_dir / "result.json", result)
        return result

    def _public_result(
        self,
        job_id: str,
        metadata: Mapping[str, object],
        joint: Mapping[str, np.ndarray],
        analysis: Mapping[str, object],
        preprocessing: Mapping[str, object],
        mechanical_channels: Mapping[str, object],
        fsr_summary: Mapping[str, object],
        artifacts: list[Path],
        warnings: list[str],
        personal_report: Optional[Mapping[str, object]] = None,
    ) -> dict[str, object]:
        session_id = str(metadata["session_id"])
        internal_metrics = analysis["deterministic_control"]["metrics"]
        metrics = {public: internal_metrics.get(internal) for public, internal in METRIC_NAMES.items()}
        safety_channel_available = bool(
            mechanical_channels["pressure"]
            or mechanical_channels["discomfort"]
            or mechanical_channels["safety_symptom"]
        )
        if not mechanical_channels["pressure"]:
            metrics["max_pressure_kpa"] = None
            metrics["pressure_over_screening_s"] = None
        if not safety_channel_available:
            metrics["safety_stop_count"] = None
        if not mechanical_channels["tension"] or metadata["options"].get("lever_arm_m") is None:
            metrics["mean_external_assist_torque_nm"] = None
            metrics["max_external_assist_torque_nm"] = None
        timestamp = np.asarray(joint["timestamp_ms"], dtype=float)
        sample_rate = 1000.0 / float(np.median(np.diff(timestamp)))
        synchronization = preprocessing.get("synchronization", {}) if isinstance(preprocessing.get("synchronization"), Mapping) else {}
        nearest = synchronization.get("nearest_sync", {}) if isinstance(synchronization.get("nearest_sync"), Mapping) else {}
        sync_gate = synchronization.get("sync_gate_passed")
        valid_pct = float(internal_metrics.get("valid_sample_pct", 0.0))
        valid_pct_min = float(self.algorithm_config.get("acceptance", {}).get("session_valid_sample_pct_min", 80.0))
        valid_gate = valid_pct >= valid_pct_min
        calibration = preprocessing.get("calibration", {}) if isinstance(preprocessing.get("calibration"), Mapping) else {}
        calibration_gate = calibration.get("quality_gate_passed")
        analysis_status = "accepted" if valid_gate and sync_gate is not False and calibration_gate is not False else "rejected"
        rejection_reasons = []
        if valid_pct <= 0.0:
            rejection_reasons.append("no_valid_angle_samples")
        elif not valid_gate:
            rejection_reasons.append("insufficient_valid_angle_samples")
        if sync_gate is False:
            rejection_reasons.append("dual_imu_sync_gate_failed")
        if calibration_gate is False:
            rejection_reasons.append("calibration_quality_gate_failed")
        thumb = np.asarray(joint.get("theta_thumb", np.full(len(timestamp), np.nan)), dtype=float)
        explanation = analysis["explanation"]
        explanation_response = explanation["response"]
        sensors = metadata.get("sensors", []) if metadata["input_type"] == "raw_dual_imu" else []
        sensor_installation = None
        if isinstance(sensors, list) and sensors:
            nodes = [
                {
                    "sensor_id": str(item["sensor_id"]),
                    "placement": str(item["placement"]),
                    "coordinate_frame": str(item["coordinate_frame"]),
                }
                for item in sensors
                if isinstance(item, Mapping)
            ]
            sensor_installation = {
                "contract_validated": True,
                "joint_crossing_pair": True,
                "side": str(nodes[0]["placement"]).split("_", 1)[0],
                "nodes": sorted(nodes, key=lambda item: item["sensor_id"]),
                "physical_verification": "metadata_only_not_physically_verified",
            }
        personal_baseline_block: Optional[dict[str, object]] = None
        if personal_report is not None:
            baseline_view = personal_report.get("baseline", {}) if isinstance(personal_report.get("baseline"), Mapping) else {}
            personal_baseline_block = {
                "participant_id": personal_report.get("participant_id"),
                "status": personal_report.get("baseline_status"),
                "role": personal_report.get("personal_baseline_role"),
                "update_applied": personal_report.get("update_applied"),
                "session_count": baseline_view.get("session_count"),
                "observed_minutes": baseline_view.get("observed_minutes"),
                "relative_exposure": personal_report.get("relative_exposure"),
                "goal_line": personal_report.get("goal_line"),
                "symptom_association": personal_report.get("symptom_association"),
                "exposure_tolerance": personal_report.get("exposure_tolerance"),
                "suggestions": personal_report.get("suggestions"),
                "control_effect": "none",
                "artifact_url": f"/api/v1/sessions/{session_id}/artifacts/personal_baseline.json",
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "session_id": session_id,
            "status": "succeeded",
            "analysis_status": analysis_status,
            "rejection_reasons": rejection_reasons,
            "evidence_type": metadata["evidence_type"],
            "algorithm_release": ALGORITHM_RELEASE,
            "sensor_installation": sensor_installation,
            "channels": {
                "wrist_angles": {"available": True, "source": "derived" if metadata["input_type"] == "raw_dual_imu" else "provided"},
                "thumb_angle": {"available": bool(np.any(np.isfinite(thumb)))},
                "pressure": {"available": bool(mechanical_channels["pressure"]), "source": "calibrated_kpa" if mechanical_channels["pressure"] else None},
                "fsr_raw": {"available": bool(mechanical_channels["fsr_raw"]), "source": mechanical_channels.get("fsr_source")},
                "tension": {"available": bool(mechanical_channels["tension"]), "source": "cable_tension_N" if mechanical_channels["tension"] else None},
                "discomfort": {"available": bool(mechanical_channels["discomfort"]), "source": "legacy_binary" if mechanical_channels["discomfort"] else None},
                "discomfort_nrs": {"available": bool(mechanical_channels["discomfort_nrs"]), "source": "participant_report_0_10" if mechanical_channels["discomfort_nrs"] else None},
                "safety_symptom": {"available": bool(mechanical_channels["safety_symptom"]), "source": "operator_safety_stop_flag" if mechanical_channels["safety_symptom"] else None},
                "user_continues": {"available": bool(mechanical_channels["user_continues"])},
            },
            "data_quality": {
                "sample_count": int(len(timestamp)),
                "valid_sample_pct": valid_pct,
                "valid_sample_pct_min": valid_pct_min,
                "valid_sample_gate_passed": valid_gate,
                "sample_rate_hz": sample_rate,
                "median_sync_error_ms": nearest.get("median_sync_error_ms"),
                "p95_sync_error_ms": nearest.get("p95_sync_error_ms"),
                "max_sync_error_ms": nearest.get("max_sync_error_ms"),
                "sync_limit_ms": synchronization.get("sync_limit_ms"),
                "sync_gate_passed": sync_gate,
            },
            "calibration": preprocessing.get("calibration"),
            "trial_condition": (
                {
                    "condition": str(metadata["condition"]),
                    "support_level": int(metadata["support_level"]),
                    "reminder_enabled": bool(metadata["reminder_enabled"]),
                    "protocol_order": "A_then_B_then_C",
                }
                if "condition" in metadata
                else None
            ),
            "fsr_proxy": dict(fsr_summary),
            "metrics": metrics,
            "alerts": [
                {
                    "timestamp_ms": 1000.0 * float(item["timestamp_s"]),
                    "zone": item["zone"],
                    "reason": item["reason"],
                    "recommend_mechanical": item["recommend_mechanical"],
                    "safety_stop": item["safety_stop"] if safety_channel_available else None,
                }
                for item in analysis["deterministic_control"]["alerts"]
            ],
            "ml_shadow": {
                "operating_mode": "shadow",
                "timing_semantics": "full_session_finalize",
                "window_count": analysis["ml_shadow"]["window_count"],
                "accepted_window_count": analysis["ml_shadow"]["accepted_window_count"],
                "rejected_window_count": analysis["ml_shadow"]["rejected_window_count"],
                "rejection_reasons": analysis["ml_shadow"]["rejection_reasons"],
                "token_count": len(analysis["ml_shadow"]["tokens"]),
                "tokens_url": f"/api/v1/sessions/{session_id}/tokens",
                "safety_effect": "none",
            },
            "control_policy": analysis["control_policy"],
            "explanation": {
                "provider": explanation["provider"],
                "model": explanation["model"],
                "api_called": explanation["api_called"],
                "summary": explanation_response["summary"],
                "observations": explanation_response["observations"],
                "limitations": explanation_response["limitations"],
                "next_steps": explanation_response["next_steps"],
                "safety_effect": "none",
            },
            "artifacts": [
                {
                    "name": path.name,
                    "media_type": {
                        ".json": "application/json",
                        ".csv": "text/csv",
                        ".png": "image/png",
                        ".svg": "image/svg+xml",
                    }.get(path.suffix.lower(), "application/octet-stream"),
                    "sha256": sha256_file(path),
                    "url": f"/api/v1/sessions/{session_id}/artifacts/{path.name}",
                }
                for path in artifacts
                if path.name in ARTIFACT_NAMES
            ],
            "warnings": warnings,
            "evidence_limits": analysis["evidence_limits"],
            "personal_baseline": personal_baseline_block,
        }

    def get_result(self, session_id: str) -> dict[str, object]:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise BackendError("SESSION_NOT_FOUND", "Session was not found.", http_status=404)
        path = self._session_dir(session_id) / "result.json"
        if not path.exists():
            matching = [self._load_job(item.stem) for item in self.jobs_root.glob("job_*.json")]
            record = next((item for item in matching if item.get("session_id") == session_id), None)
            if record and record.get("status") == "failed":
                raise BackendError("JOB_FAILED", "Analysis job failed.", details={"job_id": record["job_id"], "error": record.get("error")}, http_status=422)
            if record:
                raise BackendError("RESULT_NOT_READY", "Analysis result is not ready.", details={"job_id": record["job_id"], "status": record["status"]}, http_status=409)
            raise BackendError("SESSION_NOT_FOUND", "Session was not found.", http_status=404)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_tokens(self, session_id: str) -> dict[str, object]:
        self.get_result(session_id)
        with (self._session_dir(session_id) / "tokens.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_timeline(self, session_id: str, offset: int, limit: int) -> dict[str, object]:
        self.get_result(session_id)
        path = self._session_dir(session_id) / "timeline.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = rows[offset : offset + limit]
        typed = []
        for row in selected:
            typed.append(
                {
                    "timestamp_ms": float(row["timestamp_ms"]),
                    "theta_FE": float(row["theta_FE"]),
                    "theta_RUD": float(row["theta_RUD"]),
                    "quality": float(row["quality"]),
                    "angle_zone": row["angle_zone"],
                    "pressure_zone": row["pressure_zone"] or None,
                    "discomfort": None if row.get("discomfort", "") == "" else row["discomfort"].lower() == "true",
                    "user_continues": None if row.get("user_continues", "") == "" else row["user_continues"].lower() == "true",
                    "activity_shadow": row["activity_shadow"],
                    "alert": row["alert"].lower() == "true",
                    "would_alert": row.get("would_alert", "false").lower() == "true",
                    "alert_reason": row["alert_reason"],
                    "safety_stop": None if row["safety_stop"] == "" else row["safety_stop"].lower() == "true",
                    "fsr_raw": None if row.get("fsr_raw", "") == "" else float(row["fsr_raw"]),
                    "discomfort_nrs": None if row.get("discomfort_nrs", "") == "" else float(row["discomfort_nrs"]),
                    "safety_symptom": None if row.get("safety_symptom", "") == "" else row["safety_symptom"].lower() == "true",
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "offset": offset,
            "limit": limit,
            "total": len(rows),
            "items": typed,
        }

    def get_artifact(self, session_id: str, name: str) -> Path:
        self.get_result(session_id)
        if name not in ARTIFACT_NAMES:
            raise BackendError("ARTIFACT_NOT_FOUND", "Artifact was not found.", http_status=404)
        path = self._session_dir(session_id) / name
        if not path.is_file():
            raise BackendError("ARTIFACT_NOT_FOUND", "Artifact was not found.", http_status=404)
        return path