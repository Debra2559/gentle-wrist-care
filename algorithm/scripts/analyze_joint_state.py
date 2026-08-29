#!/usr/bin/env python3
"""Analyze one exported SheWrist joint_state CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.analysis import analyze_condition
from shewrist.data import align_mechanical_to_joint, load_config, load_joint_state_csv, load_mechanical_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("joint_state", type=Path)
    parser.add_argument("--mechanical", type=Path)
    parser.add_argument("--lever-arm-m", type=float)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/trial_summary.json")
    args = parser.parse_args()
    config = load_config(args.config)
    joint = load_joint_state_csv(args.joint_state)
    mechanical = None
    if args.mechanical:
        mechanical = align_mechanical_to_joint(joint["timestamp_ms"], load_mechanical_csv(args.mechanical))
    pressure = None
    tension = None
    discomfort = None
    user_continues = None
    if mechanical:
        pressure_columns = [key for key in ("p_radial_kPa", "p_dorsal_kPa", "p_ulnar_kPa") if key in mechanical]
        pressure = np.nanmax(np.column_stack([mechanical[key] for key in pressure_columns]), axis=1) if pressure_columns else None
        tension = mechanical.get("cable_tension_N")
        if "discomfort" in mechanical:
            discomfort = np.asarray(mechanical["discomfort"], dtype=float) == 1.0
        if "user_continues" in mechanical:
            user_continues = np.asarray(mechanical["user_continues"], dtype=float) == 1.0
    metrics, alerts = analyze_condition(
        joint["timestamp_ms"] / 1000.0,
        joint["theta_FE"],
        joint["theta_RUD"],
        config,
        quality=joint.get("quality"),
        pressure_kpa=pressure,
        discomfort=discomfort,
        cable_tension_n=tension,
        lever_arm_m=args.lever_arm_m,
        user_continues=user_continues,
    )
    payload = {"input": str(args.joint_state), "metrics": metrics, "alerts": alerts}
    calibration_ids = joint.get("calibration_id")
    data_status = joint.get("data_status")
    synthetic = bool(
        (calibration_ids is not None and any("SYNTHETIC" in str(value).upper() for value in calibration_ids))
        or (data_status is not None and any("SYNTHETIC" in str(value).upper() for value in data_status))
    )
    if synthetic:
        payload["data_status"] = "SYNTHETIC DEMONSTRATION ONLY - NOT HUMAN EXPERIMENT RESULTS"
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()