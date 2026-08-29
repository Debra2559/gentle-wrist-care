"""Typed response contracts exposed by the SheWrist Backend API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorEnvelope(BaseModel):
    schema_version: str = "1.0"
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    schema_version: str
    algorithm_release: str
    mode: str


class JobResponse(BaseModel):
    schema_version: str
    job_id: str
    session_id: str
    status: str
    stage: str
    progress_pct: int = Field(ge=0, le=100)
    created_at: str
    updated_at: str
    status_url: str
    result_url: str
    idempotent_replay: bool = False
    analysis_status: Optional[str] = None
    error: Optional[ErrorDetail] = None


class ChannelDescriptor(BaseModel):
    available: bool
    source: Optional[str] = None


class SensorNode(BaseModel):
    sensor_id: str
    placement: str
    coordinate_frame: str


class SensorInstallation(BaseModel):
    contract_validated: bool
    joint_crossing_pair: bool
    side: str
    nodes: List[SensorNode]
    physical_verification: str


class Channels(BaseModel):
    wrist_angles: ChannelDescriptor
    thumb_angle: ChannelDescriptor
    pressure: ChannelDescriptor
    fsr_raw: ChannelDescriptor
    tension: ChannelDescriptor
    discomfort: ChannelDescriptor
    discomfort_nrs: ChannelDescriptor
    safety_symptom: ChannelDescriptor
    user_continues: ChannelDescriptor


class DataQuality(BaseModel):
    sample_count: int
    valid_sample_pct: float
    valid_sample_pct_min: float
    valid_sample_gate_passed: bool
    sample_rate_hz: float
    median_sync_error_ms: Optional[float] = None
    p95_sync_error_ms: Optional[float] = None
    max_sync_error_ms: Optional[float] = None
    sync_limit_ms: Optional[float] = None
    sync_gate_passed: Optional[bool] = None


class Metrics(BaseModel):
    task_duration_s: Optional[float] = None
    valid_sample_pct: Optional[float] = None
    high_posture_time_pct: Optional[float] = None
    fe_excess_dose_deg_s: Optional[float] = None
    rud_excess_dose_deg_s: Optional[float] = None
    total_excess_dose_deg_s: Optional[float] = None
    longest_high_posture_s: Optional[float] = None
    fe_cycles_per_min: Optional[float] = None
    rud_cycles_per_min: Optional[float] = None
    max_abs_fe_deg: Optional[float] = None
    max_abs_rud_deg: Optional[float] = None
    alert_count: Optional[int] = None
    would_alert_count: Optional[int] = None
    mechanical_recommendation_count: Optional[int] = None
    safety_stop_count: Optional[int] = None
    max_pressure_kpa: Optional[float] = None
    pressure_over_screening_s: Optional[float] = None
    mean_external_assist_torque_nm: Optional[float] = None
    max_external_assist_torque_nm: Optional[float] = None


class Alert(BaseModel):
    timestamp_ms: float
    zone: str
    reason: str
    recommend_mechanical: bool
    safety_stop: Optional[bool] = None


class MlShadowSummary(BaseModel):
    operating_mode: str
    timing_semantics: str
    window_count: int
    accepted_window_count: int
    rejected_window_count: int
    rejection_reasons: Dict[str, int]
    token_count: int
    tokens_url: str
    safety_effect: str


class ControlPolicy(BaseModel):
    angle_alert_authority: str
    pressure_stop_authority: str
    mechanical_action: str
    ml_control_authority: str
    llm_control_authority: str


class ExplanationSummary(BaseModel):
    provider: str
    model: str
    api_called: bool
    summary: str
    observations: List[str]
    limitations: List[str]
    next_steps: List[str]
    safety_effect: str


class Artifact(BaseModel):
    name: str
    media_type: str
    sha256: str
    url: str


class FsrProxySummary(BaseModel):
    available: bool
    source: Optional[str] = None
    unit: Optional[str] = None
    calibrated_to_pressure: bool = False
    mean: Optional[float] = None
    p95: Optional[float] = None
    max: Optional[float] = None


class TrialCondition(BaseModel):
    condition: str
    support_level: int
    reminder_enabled: bool
    protocol_order: str = "A_then_B_then_C"


class SessionResult(BaseModel):
    schema_version: str
    job_id: str
    session_id: str
    status: str
    analysis_status: str
    rejection_reasons: List[str]
    evidence_type: str
    algorithm_release: str
    sensor_installation: Optional[SensorInstallation] = None
    channels: Channels
    data_quality: DataQuality
    calibration: Optional[Dict[str, Any]] = None
    trial_condition: Optional[TrialCondition] = None
    fsr_proxy: FsrProxySummary
    metrics: Metrics
    alerts: List[Alert]
    ml_shadow: MlShadowSummary
    control_policy: ControlPolicy
    explanation: ExplanationSummary
    artifacts: List[Artifact]
    warnings: List[str]
    evidence_limits: List[str]
    personal_baseline: Optional[Dict[str, Any]] = None


class CalibrationResult(BaseModel):
    schema_version: str
    calibration_id: str
    participant_id: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    sample_rate_hz: Optional[float] = None
    quality_gate_passed: Optional[bool] = None
    quality_reasons: List[str] = Field(default_factory=list)
    algorithm: Optional[str] = None
    neutral_stationary_sample_pct: Optional[float] = None
    neutral_stationary_sample_pct_min: Optional[float] = None
    flexion_extension_axis: Optional[List[float]] = None
    radial_ulnar_axis: Optional[List[float]] = None
    pronation_supination_axis: Optional[List[float]] = None
    neutral_quaternion: Optional[List[float]] = None
    self_url: str


class TimelineRow(BaseModel):
    timestamp_ms: float
    theta_FE: float
    theta_RUD: float
    quality: float
    angle_zone: str
    pressure_zone: Optional[str] = None
    discomfort: Optional[bool] = None
    user_continues: Optional[bool] = None
    activity_shadow: str
    alert: bool
    would_alert: bool
    alert_reason: str
    safety_stop: Optional[bool] = None
    fsr_raw: Optional[float] = None
    discomfort_nrs: Optional[float] = None
    safety_symptom: Optional[bool] = None


class TimelineResponse(BaseModel):
    schema_version: str
    session_id: str
    offset: int
    limit: int
    total: int
    items: List[TimelineRow]


class InertialTokenResponse(BaseModel):
    schema_version: str
    session_id: str
    event_type: str
    source: str
    evidence_type: str
    operating_mode: str
    start_ms: int
    end_ms: int
    duration_ms: int
    confidence: float
    mean_quality: float
    peak_abs_fe_deg: float
    peak_abs_rud_deg: float
    model_name: str
    model_version: str
    threshold_version: str
    safety_effect: str


class TokensResponse(BaseModel):
    schema_version: str
    operating_mode: str
    tokens: List[InertialTokenResponse]