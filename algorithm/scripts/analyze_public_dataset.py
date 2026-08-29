#!/usr/bin/env python3
"""Analyze the downloaded 11-subject wrist-motion IMU dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.analysis import analyze_condition
from shewrist.calibration import interval_mask
from shewrist.data import (
    find_neutral_interval,
    load_annotations,
    load_config,
    load_public_trial,
    write_csv,
    write_json,
)
from shewrist.kinematics import compute_wrist_kinematics, kinematics_rows
from shewrist.quality import sample_quality, timestamp_quality


def split_functional_repeats(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = {"Flexion", "Extension", "Radial Deviation", "Ulnar Deviation"}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("Segment") == "wrist" and row.get("Type") in labels:
            duration = float(row["End"]) - float(row["Init"])
            if 0.0 < duration <= 15.0:
                grouped.setdefault(str(row["Type"]), []).append(row)
    calibration_rows: list[dict] = []
    validation_rows: list[dict] = []
    for label in sorted(labels):
        repeats = sorted(grouped.get(label, []), key=lambda row: float(row["Init"]))
        if repeats:
            calibration_rows.append(repeats[0])
            validation_rows.extend(repeats[1:])
    if not all(any(row.get("Type") == label for row in calibration_rows) for label in labels):
        raise ValueError("functional calibration requires one usable interval for every wrist direction")
    return calibration_rows, validation_rows


def interval_means(timestamp: np.ndarray, values: np.ndarray, rows: list[dict]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for row in rows:
        if row["Segment"] != "wrist" or row["Type"] == "AnatomicalPos":
            continue
        duration = float(row["End"]) - float(row["Init"])
        if duration <= 0.0 or duration > 15.0:
            continue
        mask = interval_mask(timestamp, float(row["Init"]), float(row["End"]))
        if np.count_nonzero(mask) >= 3:
            result.setdefault(row["Type"], []).append(float(np.nanmedian(values[mask])))
    return result


def direction_score(fe_means: dict[str, list[float]], rud_means: dict[str, list[float]]) -> tuple[int, int]:
    checks = []
    checks.extend(value > 0.0 for value in fe_means.get("Extension", []))
    checks.extend(value < 0.0 for value in fe_means.get("Flexion", []))
    checks.extend(value > 0.0 for value in rud_means.get("Ulnar Deviation", []))
    checks.extend(value < 0.0 for value in rud_means.get("Radial Deviation", []))
    return sum(checks), len(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/raw/UpperBodyMovements")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/public_dataset_summary.csv")
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "outputs/public_dataset_validation.json")
    parser.add_argument("--algorithm", choices=("madgwick", "mahony"), default="madgwick")
    parser.add_argument(
        "--use-magnetometer",
        action="store_true",
        help="Use 9-axis fusion; requires a calibrated magnetometer for quantitative interpretation.",
    )
    parser.add_argument(
        "--no-joint-output",
        action="store_true",
        help="Do not export the first participant's joint_state CSV.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    use_magnetometer = bool(args.use_magnetometer or config["fusion"].get("use_magnetometer", False))
    annotations_path = args.dataset / "annotations.csv"
    subjects = sorted(path.name for path in args.dataset.glob("subject*") if path.is_dir())
    if not subjects:
        raise SystemExit(f"no subject directories under {args.dataset}")
    summary_rows: list[dict[str, object]] = []
    calibration_details: dict[str, object] = {}
    total_correct = total_checks = 0
    descriptive_correct = descriptive_checks = 0
    for subject in subjects:
        annotations = load_annotations(annotations_path, subject, "set2")
        calibration_annotations, validation_annotations = split_functional_repeats(annotations)
        neutral_interval = find_neutral_interval(annotations)
        trial = load_public_trial(args.dataset, subject, "set2", 100.0)
        t = np.asarray(trial["timestamp_s"])
        forearm_quality, _ = sample_quality(
            t,
            np.asarray(trial["forearm_accel"]),
            np.asarray(trial["forearm_gyro"]),
            np.asarray(trial["forearm_mag"]) if use_magnetometer else None,
            config["quality"],
        )
        hand_quality, _ = sample_quality(
            t,
            np.asarray(trial["hand_accel"]),
            np.asarray(trial["hand_gyro"]),
            np.asarray(trial["hand_mag"]) if use_magnetometer else None,
            config["quality"],
        )
        result = compute_wrist_kinematics(
            t,
            np.asarray(trial["forearm_accel"]),
            np.asarray(trial["forearm_gyro"]),
            np.asarray(trial["hand_accel"]),
            np.asarray(trial["hand_gyro"]),
            neutral_interval,
            calibration_annotations,
            np.asarray(trial["forearm_mag"]) if use_magnetometer else None,
            np.asarray(trial["hand_mag"]) if use_magnetometer else None,
            args.algorithm,
            config["fusion"],
            config["quality"],
        )
        quality = np.minimum.reduce((forearm_quality, hand_quality, result.quality))
        metrics, alerts = analyze_condition(t, result.theta_fe_deg, result.theta_rud_deg, config, quality=quality)
        fe_means = interval_means(t, result.theta_fe_deg, annotations)
        rud_means = interval_means(t, result.theta_rud_deg, annotations)
        validation_fe_means = interval_means(t, result.theta_fe_deg, validation_annotations)
        validation_rud_means = interval_means(t, result.theta_rud_deg, validation_annotations)
        correct, checks = direction_score(validation_fe_means, validation_rud_means)
        all_correct, all_checks = direction_score(fe_means, rud_means)
        total_correct += correct
        total_checks += checks
        descriptive_correct += all_correct
        descriptive_checks += all_checks
        row = {
            "subject_id": subject,
            "algorithm": args.algorithm,
            "sensor_mode": "9-axis" if use_magnetometer else "6-axis",
            "sample_rate_hz": timestamp_quality(t)["nominal_rate_hz"],
            "duration_s": metrics["task_duration_s"],
            "valid_sample_pct": metrics["valid_sample_pct"],
            "P_high_pct": metrics["P_high_pct"],
            "D_FE_deg_s": metrics["D_FE_deg_s"],
            "D_RUD_deg_s": metrics["D_RUD_deg_s"],
            "L_max_s": metrics["L_max_s"],
            "max_abs_FE_deg": metrics["max_abs_FE_deg"],
            "max_abs_RUD_deg": metrics["max_abs_RUD_deg"],
            "direction_correct": correct,
            "direction_checks": checks,
            "direction_accuracy_pct": 100.0 * correct / checks if checks else None,
            "all_interval_direction_accuracy_pct": 100.0 * all_correct / all_checks if all_checks else None,
            "alerts": len(alerts),
        }
        summary_rows.append(row)
        calibration_details[subject] = {
            "neutral_interval_s": list(neutral_interval),
            "flexion_extension_axis": result.axes.flexion_extension.tolist(),
            "radial_ulnar_axis": result.axes.radial_ulnar.tolist(),
            "pronation_supination_axis": result.axes.pronation_supination.tolist(),
            "forearm_gyro_bias_rad_s": result.forearm_gyro_bias_rad_s.tolist(),
            "hand_gyro_bias_rad_s": result.hand_gyro_bias_rad_s.tolist(),
            "calibration_intervals": calibration_annotations,
            "validation_intervals": validation_annotations,
            "interval_medians_deg": {"FE": fe_means, "RUD": rud_means},
        }
        if subject == subjects[0] and not args.no_joint_output:
            write_csv(
                PROJECT_ROOT / "data/processed/public_subject01_set2_joint_state.csv",
                ["timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality"],
                kinematics_rows(result, f"{subject}-set2-{args.algorithm}"),
            )
    fields = list(summary_rows[0])
    write_csv(args.output, fields, summary_rows)
    payload = {
        "dataset": {
            "name": "Upper-body movements: precise tracking of human motion using inertial sensors",
            "doi": "10.5281/zenodo.4029127",
            "license": "CC BY 4.0",
            "subjects": len(subjects),
            "sampling_rate_hz": 100.0,
            "scope": "set2 wrist flexion/extension and radial/ulnar deviation",
            "ground_truth_limit": "interval labels only; no independent angle reference",
        },
        "algorithm": args.algorithm,
        "sensor_mode": "9-axis" if use_magnetometer else "6-axis",
        "direction_validation_design": "The first usable repeat of each direction calibrates the functional axes; later repeats are held out for direction checks.",
        "held_out_direction_accuracy_pct": 100.0 * total_correct / total_checks if total_checks else None,
        "held_out_direction_correct": total_correct,
        "held_out_direction_checks": total_checks,
        "all_interval_direction_accuracy_pct": 100.0 * descriptive_correct / descriptive_checks if descriptive_checks else None,
        "subject_summaries": summary_rows,
        "calibration": calibration_details,
        "interpretation": [
            "This run validates executable dual-node processing and held-out movement direction on public raw data.",
            "It does not establish absolute angle MAE because this dataset has no goniometer or optical angle ground truth.",
            "Exposure values describe the dataset protocol and are not clinical risk estimates.",
            "The default is 6-axis fusion because the public magnetometer streams are not accompanied by a reproducible calibration; 9-axis results are exploratory only.",
        ],
    }
    write_json(args.json, payload)
    print(json.dumps({"subjects": len(subjects), "held_out_direction_accuracy_pct": payload["held_out_direction_accuracy_pct"], "csv": str(args.output), "json": str(args.json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
