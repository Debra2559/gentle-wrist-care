#!/usr/bin/env python3
"""Run deterministic alerts and the ML shadow model in one auditable analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import (
    align_mechanical_to_joint,
    load_config,
    load_joint_state_csv,
    load_mechanical_csv,
    write_json,
)
from shewrist.ml import ShadowActivityPipeline
from shewrist.pipeline import analyze_with_shadow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joint_state", type=Path)
    parser.add_argument("--mechanical", type=Path)
    parser.add_argument("--lever-arm-m", type=float)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "outputs/ml/activity_cnn_hmm_shadow.npz")
    parser.add_argument("--algorithm-config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--session-id", default="offline-analysis")
    parser.add_argument("--evidence-type", choices=("bench", "replay", "simulation", "human"), default="replay")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/ml/combined_analysis.json")
    args = parser.parse_args()

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
    payload = analyze_with_shadow(
        joint,
        load_config(args.algorithm_config),
        load_config(args.ml_config),
        ShadowActivityPipeline.load(args.model),
        args.session_id,
        args.evidence_type,
        pressure_kpa=pressure,
        discomfort=discomfort,
        cable_tension_n=tension,
        lever_arm_m=args.lever_arm_m,
        user_continues=user_continues,
    )
    payload["inputs"] = {
        "joint_state": str(args.joint_state),
        "mechanical": None if args.mechanical is None else str(args.mechanical),
        "model": str(args.model),
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "deterministic_alerts": len(payload["deterministic_control"]["alerts"]),
                "shadow_tokens": len(payload["ml_shadow"]["tokens"]),
                "ml_control_authority": "none",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()