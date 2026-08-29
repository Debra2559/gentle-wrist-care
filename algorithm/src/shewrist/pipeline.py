"""Combined deterministic safety analysis and non-controlling ML shadow path."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .analysis import analyze_condition
from .ml import ShadowActivityPipeline
from .ml_data import build_joint_state_windows
from .tokens import build_inertial_tokens, feedback_from_token


def analyze_with_shadow(
    joint_state: Mapping[str, np.ndarray],
    algorithm_config: Mapping[str, object],
    ml_config: Mapping[str, object],
    activity_pipeline: ShadowActivityPipeline,
    session_id: str,
    evidence_type: str,
    pressure_kpa: np.ndarray | None = None,
    discomfort: np.ndarray | None = None,
    cable_tension_n: np.ndarray | None = None,
    lever_arm_m: float | np.ndarray | None = None,
    user_continues: np.ndarray | None = None,
    angle_alerts_enabled: bool = True,
    mechanical_recommendations_enabled: bool = True,
) -> dict[str, object]:
    """Run both branches while keeping authority in deterministic logic.

    The shadow model can add descriptive activity tokens.  It cannot suppress,
    create, or modify angle alerts, pressure stops, or mechanical suggestions.
    """
    if evidence_type not in {"bench", "replay", "simulation", "human"}:
        raise ValueError("unsupported evidence_type")
    timestamp_s = np.asarray(joint_state["timestamp_ms"], dtype=float) / 1000.0
    metrics, alerts = analyze_condition(
        timestamp_s,
        np.asarray(joint_state["theta_FE"], dtype=float),
        np.asarray(joint_state["theta_RUD"], dtype=float),
        algorithm_config,
        quality=joint_state.get("quality"),
        pressure_kpa=pressure_kpa,
        discomfort=discomfort,
        cable_tension_n=cable_tension_n,
        lever_arm_m=lever_arm_m,
        user_continues=user_continues,
        angle_alerts_enabled=angle_alerts_enabled,
        mechanical_recommendations_enabled=mechanical_recommendations_enabled,
    )
    windows = build_joint_state_windows(joint_state, ml_config, session_id)
    prediction = activity_pipeline.predict(windows)
    tokens = build_inertial_tokens(
        prediction.accepted_labels,
        prediction.confidence,
        windows.start_s,
        windows.end_s,
        windows.mean_quality,
        windows.windows,
        activity_pipeline.class_names,
        activity_pipeline.feature_names,
        session_id,
        sequence_ids=windows.sequence_ids,
        evidence_type=evidence_type,
    )
    return {
        "schema_version": 1,
        "session_id": session_id,
        "evidence_type": evidence_type,
        "control_policy": {
            "angle_alert_authority": "deterministic_exposure_engine" if angle_alerts_enabled else "disabled_by_trial_condition",
            "pressure_stop_authority": "deterministic_calibrated_pressure_or_safety_symptom",
            "mechanical_action": "manual_only" if mechanical_recommendations_enabled else "disabled_by_trial_protocol",
            "ml_control_authority": "none",
            "llm_control_authority": "none",
        },
        "deterministic_control": {
            "metrics": metrics,
            "alerts": alerts,
        },
        "ml_shadow": {
            "operating_mode": "shadow",
            "window_count": len(windows),
            "accepted_window_count": int(np.count_nonzero(prediction.accepted_labels >= 0)),
            "rejected_window_count": int(np.count_nonzero(prediction.accepted_labels < 0)),
            "rejection_reasons": {
                str(reason): int(np.count_nonzero(prediction.rejection_reason == reason))
                for reason in sorted(set(prediction.rejection_reason.tolist()))
            },
            "tokens": [token.to_dict() for token in tokens],
            "feedback": [feedback_from_token(token) for token in tokens],
            "alarm_control_effect": "none",
            "mechanical_control_effect": "none",
        },
        "interpretation": (
            "Deterministic angle, pressure, and reported-discomfort rules are the only control path. "
            "ML labels are experimental context from a public healthy-participant dataset."
        ),
    }