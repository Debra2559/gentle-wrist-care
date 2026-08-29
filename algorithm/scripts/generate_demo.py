#!/usr/bin/env python3
"""Generate deterministic A/B/C software-integration data under the frozen field protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.analysis import analyze_condition, summarize_conditions
from shewrist.data import load_config, write_csv, write_json


DATA_STATUS = "SYNTHETIC DEMONSTRATION ONLY - NOT HUMAN EXPERIMENT RESULTS"
TRIAL_SETTINGS = {
    "A": {"support_level": 0, "reminder_enabled": False},
    "B": {"support_level": 1, "reminder_enabled": False},
    "C": {"support_level": 1, "reminder_enabled": True},
}


def condition_signal(t: np.ndarray, participant: int, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = participant * 0.17
    task_phase = np.where(t < 90.0, 0.0, 1.0)
    frequency = np.where(task_phase == 0.0, 0.11, 0.075)
    support_scale = 1.0 if condition == "A" else 0.86
    reminder_scale = np.where(t < 20.0, 1.0, 0.72) if condition == "C" else np.ones(len(t))
    fe = support_scale * reminder_scale * (24.0 + 12.0 * np.sin(2.0 * np.pi * frequency * t + phase))
    rud = support_scale * reminder_scale * (13.0 + 8.0 * np.sin(2.0 * np.pi * (frequency * 0.8) * t + 0.7 + phase))
    fe[t < 15.0] = 35.0
    fsr_base = 900.0 if condition == "A" else 1300.0
    fsr_raw = fsr_base + 35.0 * np.sin(2.0 * np.pi * 0.03 * t + phase)
    return fe, rud, fsr_raw


def main() -> None:
    config = load_config(PROJECT_ROOT / "config/thresholds.yaml")
    output_dir = PROJECT_ROOT / "examples/synthetic_abc"
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, 180.0, 0.02)
    records: list[dict[str, object]] = []
    for participant in range(1, 9):
        for condition in ("A", "B", "C"):
            settings = TRIAL_SETTINGS[condition]
            fe, rud, fsr_raw = condition_signal(t, participant, condition)
            metrics, alerts = analyze_condition(
                t,
                fe,
                rud,
                config,
                angle_alerts_enabled=bool(settings["reminder_enabled"]),
                mechanical_recommendations_enabled=False,
            )
            performance_drop = {"A": 0.0, "B": 1.0, "C": 1.5}[condition] + 0.03 * participant
            comfort = {"A": 5.2, "B": 5.5, "C": 5.8}[condition] - 0.02 * participant
            row = {
                "data_status": DATA_STATUS,
                "participant_id": f"SYN{participant:02d}",
                "condition_id": condition,
                "support_level": settings["support_level"],
                "reminder_enabled": settings["reminder_enabled"],
                **metrics,
                "fsr_raw_adc_mean": float(np.mean(fsr_raw)),
                "fsr_raw_adc_p95": float(np.percentile(fsr_raw, 95)),
                "task_performance_drop_pct": performance_drop,
                "comfort": comfort,
                "pressure_discomfort": False,
                "effective_alert_acceptance_pct": 82.0 + 0.2 * participant if condition == "C" else None,
            }
            records.append(row)
            if participant == 1:
                joint_rows = [
                    {
                        "data_status": DATA_STATUS,
                        "timestamp_ms": int(round(t[index] * 1000.0)),
                        "theta_FE": fe[index],
                        "theta_RUD": rud[index],
                        "theta_thumb": "",
                        "angular_velocity": "",
                        "calibration_id": "SYNTHETIC-NOT-CLINICAL",
                        "quality": 1.0,
                        "task_phase": "typing" if t[index] < 90.0 else "mouse",
                    }
                    for index in range(len(t))
                ]
                mechanical_rows = [
                    {
                        "data_status": DATA_STATUS,
                        "device_ms": 1000000 + int(round(t[index] * 1000.0)),
                        "condition": condition,
                        "support_level": settings["support_level"],
                        "fsr_raw_adc": fsr_raw[index],
                        "discomfort_nrs": 1.0,
                        "safety_symptom_flag": 0,
                        "user_continues": 1,
                        "task_phase": "typing" if t[index] < 90.0 else "mouse",
                    }
                    for index in range(len(t))
                ]
                write_csv(output_dir / f"SYN01_{condition}_joint_state.csv", list(joint_rows[0]), joint_rows)
                write_csv(output_dir / f"SYN01_{condition}_mechanical.csv", list(mechanical_rows[0]), mechanical_rows)
                write_json(
                    output_dir / f"SYN01_{condition}_alerts.json",
                    {
                        "data_status": DATA_STATUS,
                        "condition": condition,
                        "support_level": settings["support_level"],
                        "reminder_enabled": settings["reminder_enabled"],
                        "alert_count": metrics["alert_count"],
                        "would_alert_count": metrics["would_alert_count"],
                        "alerts": alerts,
                    },
                )
    summary = summarize_conditions(records, config)
    summary["data_status"] = DATA_STATUS
    summary["protocol"] = {
        "order": "A_then_B_then_C",
        "A": TRIAL_SETTINGS["A"],
        "B": TRIAL_SETTINGS["B"],
        "C": TRIAL_SETTINGS["C"],
        "task_duration_s": 180,
        "task_split": "90_s_typing_then_90_s_mouse",
        "pressure_evidence": "uncalibrated_fsr_proxy_only",
    }
    write_csv(PROJECT_ROOT / "outputs/demo_condition_metrics.csv", list(records[0]), records)
    write_json(PROJECT_ROOT / "outputs/demo_summary.json", summary)
    print(
        json.dumps(
            {
                "records": len(records),
                "decision": summary["go_no_go"]["decision"],
                "status": summary["data_status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
