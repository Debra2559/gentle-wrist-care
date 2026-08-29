"""Offline trial analysis and A/B/C aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np

from .exposure import ExposureEngine
from .metrics import exposure_metrics, intervention_efficiency
from .validation import evaluate_go_no_go, paired_condition_comparison


def analyze_condition(
    timestamp_s: np.ndarray,
    theta_fe_deg: np.ndarray,
    theta_rud_deg: np.ndarray,
    config: Mapping[str, object],
    quality: np.ndarray | None = None,
    pressure_kpa: np.ndarray | None = None,
    discomfort: np.ndarray | None = None,
    cable_tension_n: np.ndarray | None = None,
    lever_arm_m: float | np.ndarray | None = None,
    user_continues: np.ndarray | None = None,
    angle_alerts_enabled: bool = True,
    mechanical_recommendations_enabled: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    engine = ExposureEngine(config)
    states = engine.process(
        timestamp_s,
        theta_fe_deg,
        theta_rud_deg,
        pressure_kpa,
        discomfort,
        quality,
        user_continues=user_continues,
        angle_alerts_enabled=angle_alerts_enabled,
        mechanical_recommendations_enabled=mechanical_recommendations_enabled,
    )
    metrics = exposure_metrics(
        timestamp_s,
        theta_fe_deg,
        theta_rud_deg,
        config,
        zones=[state.angle_zone for state in states],
        quality=quality,
        pressure_kpa=pressure_kpa,
        cable_tension_n=cable_tension_n,
        lever_arm_m=lever_arm_m,
    )
    metrics.update(
        {
            "alert_count": int(sum(state.alert for state in states)),
            "would_alert_count": int(sum(state.would_alert for state in states)),
            "mechanical_recommendation_count": int(sum(state.recommend_mechanical for state in states)),
            "safety_stop_count": int(
                sum(
                    state.safety_stop and (index == 0 or not states[index - 1].safety_stop)
                    for index, state in enumerate(states)
                )
            ),
        }
    )
    alerts = [
        {
            "timestamp_s": state.timestamp_s,
            "zone": state.zone,
            "reason": state.alert_reason,
            "recommend_mechanical": state.recommend_mechanical,
            "safety_stop": state.safety_stop,
        }
        for state in states
        if state.alert or state.recommend_mechanical
    ]
    return metrics, alerts


def summarize_conditions(records: list[dict[str, object]], config: Mapping[str, object]) -> dict[str, object]:
    metrics_to_compare = ["P_high_pct", "D_FE_deg_s", "D_RUD_deg_s", "D_total_deg_s", "L_max_s"]
    comparisons: dict[str, object] = {}
    for metric in metrics_to_compare:
        for pair in (("A", "B"), ("A", "C"), ("B", "C")):
            key = f"{metric}_{pair[0]}_vs_{pair[1]}"
            try:
                comparisons[key] = paired_condition_comparison(records, metric, pair[0], pair[1])
            except ValueError:
                comparisons[key] = None
    participants: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in records:
        participants[str(row["participant_id"])][str(row["condition_id"])] = row
    participant_efficiency = {}
    participant_support_efficiency = {}
    for participant, conditions in participants.items():
        if "A" in conditions and "B" in conditions:
            participant_support_efficiency[participant] = intervention_efficiency(
                float(conditions["A"].get("D_total_deg_s", np.nan)),
                float(conditions["B"].get("D_total_deg_s", np.nan)),
            )
        if "A" in conditions and "C" in conditions:
            participant_efficiency[participant] = intervention_efficiency(
                float(conditions["A"].get("D_total_deg_s", np.nan)),
                float(conditions["C"].get("D_total_deg_s", np.nan)),
            )
    ac = comparisons.get("D_total_deg_s_A_vs_C")
    bc = comparisons.get("D_total_deg_s_B_vs_C")
    condition_c = [row for row in records if str(row.get("condition_id")) == "C"]
    max_pressure = max(
        (float(row["max_pressure_kPa"]) for row in condition_c if row.get("max_pressure_kPa") is not None),
        default=None,
    )
    performance_drop = np.mean(
        [float(row["task_performance_drop_pct"]) for row in condition_c if row.get("task_performance_drop_pct") is not None]
    ) if any(row.get("task_performance_drop_pct") is not None for row in condition_c) else None
    comfort = np.mean(
        [float(row["comfort"]) for row in condition_c if row.get("comfort") is not None]
    ) if any(row.get("comfort") is not None for row in condition_c) else None
    discomfort_values = [bool(row["pressure_discomfort"]) for row in condition_c if row.get("pressure_discomfort") is not None]
    pressure_discomfort = any(discomfort_values) if discomfort_values else None
    alert_acceptance = np.mean(
        [
            float(row["effective_alert_acceptance_pct"])
            for row in condition_c
            if row.get("effective_alert_acceptance_pct") is not None
        ]
    ) if any(row.get("effective_alert_acceptance_pct") is not None for row in condition_c) else None
    return {
        "records": records,
        "comparisons": comparisons,
        "comparison_meanings": {
            "A_vs_B": "support_increment",
            "B_vs_C": "reminder_increment_at_same_support",
            "A_vs_C": "support_plus_reminder_combined",
        },
        "participant_support_efficiency_pct": participant_support_efficiency,
        "participant_combined_efficiency_pct": participant_efficiency,
        "participant_mechanical_efficiency_pct": participant_efficiency,
        "go_no_go": evaluate_go_no_go(
            ac,
            bc,
            max_pressure,
            performance_drop,
            comfort,
            config,
            pressure_discomfort=pressure_discomfort,
            effective_alert_acceptance_pct=alert_acceptance,
        ),
    }