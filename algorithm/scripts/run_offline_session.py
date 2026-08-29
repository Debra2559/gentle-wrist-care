#!/usr/bin/env python3
"""Run one complete, auditable SheWrist offline session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import load_annotations, load_config, load_joint_state_csv, write_csv, write_json
from shewrist.explanation import provider_from_config
from shewrist.faults import FaultSpec
from shewrist.ml import ShadowActivityPipeline
from shewrist.reporting import plot_session_report
from shewrist.session import analyze_session, prepare_public_joint_state, sha256_file


def parse_fault(value: str) -> FaultSpec:
    parts = value.split(":")
    kind = parts[0]
    target = parts[1] if len(parts) > 1 and parts[1] else "hand"
    magnitude = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
    return FaultSpec(kind=kind, target=target, magnitude=magnitude)


def joint_rows(joint: dict[str, np.ndarray]) -> list[dict[str, object]]:
    fields = ("timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality")
    rows = []
    for index in range(len(joint["timestamp_ms"])):
        row = {}
        for field in fields:
            value = joint[field][index]
            row[field] = "" if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--joint-state", type=Path)
    source.add_argument("--public-subject", default="subject01")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/raw/UpperBodyMovements")
    parser.add_argument("--set", dest="set_name", default="set2")
    parser.add_argument("--fault", action="append", default=[], type=parse_fault, metavar="KIND[:TARGET[:MAGNITUDE]]")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "outputs/ml/activity_cnn_hmm_shadow.npz")
    parser.add_argument("--algorithm-config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--explanation-config", type=Path, default=PROJECT_ROOT / "config/explanation_api.json")
    parser.add_argument("--explanation-provider", choices=("template", "openai_compatible"))
    parser.add_argument("--enable-external-api", action="store_true")
    parser.add_argument("--session-id", default="public-subject01-offline-v08")
    parser.add_argument("--evidence-type", choices=("bench", "replay", "simulation", "human"), default="replay")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/offline_session")
    parser.add_argument("--skip-charts", action="store_true")
    args = parser.parse_args()

    algorithm_config = load_config(args.algorithm_config)
    ml_config = load_config(args.ml_config)
    explanation_config = load_config(args.explanation_config)
    if args.explanation_provider:
        explanation_config["provider"] = args.explanation_provider
    if args.enable_external_api:
        explanation_config["enabled"] = True
    provider_from_config(explanation_config)
    pipeline = ShadowActivityPipeline.load(args.model)
    input_paths: list[Path]
    preprocessing: dict[str, object]
    if args.joint_state:
        joint = load_joint_state_csv(args.joint_state)
        input_paths = [args.joint_state]
        preprocessing = {
            "source": "joint_state_csv",
            "calibration": {"status": "provided_by_input", "calibration_id": str(joint.get("calibration_id", [""])[0])},
            "fault_injection": {"enabled": False, "note": "Raw-IMU fault injection is unavailable for precomputed joint_state."},
        }
        if args.fault:
            raise SystemExit("--fault requires a public raw-IMU source, not --joint-state")
    else:
        annotations_path = args.dataset / "annotations.csv"
        annotations = load_annotations(annotations_path, args.public_subject, args.set_name)
        joint, preprocessing, raw_paths = prepare_public_joint_state(
            args.dataset,
            annotations,
            args.public_subject,
            args.set_name,
            algorithm_config,
            args.fault,
        )
        input_paths = [annotations_path, *raw_paths]
        preprocessing["source"] = {
            "type": "public_raw_dual_imu",
            "subject": args.public_subject,
            "set": args.set_name,
            "ground_truth_limit": "Movement intervals only; no independent angle, pressure, pain, or clinical truth.",
        }
    analysis, timeline = analyze_session(
        joint,
        algorithm_config,
        ml_config,
        pipeline,
        explanation_config,
        args.session_id,
        args.evidence_type,
        args.chunk_size,
    )
    analysis["preprocessing"] = preprocessing
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint_path = args.output_dir / "joint_state.csv"
    timeline_path = args.output_dir / "timeline.csv"
    analysis_path = args.output_dir / "analysis.json"
    tokens_path = args.output_dir / "tokens.json"
    joint_fields = ["timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality"]
    timeline_fields = ["timestamp_ms", "theta_FE", "theta_RUD", "quality", "angle_zone", "pressure_zone", "activity_shadow", "alert", "alert_reason", "safety_stop"]
    write_csv(joint_path, joint_fields, joint_rows(joint))
    write_csv(timeline_path, timeline_fields, timeline)
    write_json(analysis_path, analysis)
    write_json(tokens_path, {"schema_version": 1, "operating_mode": "shadow", "tokens": analysis["ml_shadow"]["tokens"]})
    chart_paths = [] if args.skip_charts else plot_session_report(timeline, analysis, args.output_dir / "session_report")
    manifest = {
        "schema_version": 1,
        "session_id": args.session_id,
        "evidence_type": args.evidence_type,
        "algorithm_release": "offline-v0.8",
        "input_schema": "public_raw_dual_imu" if not args.joint_state else "joint_state_csv",
        "versions": {
            "threshold_schema": algorithm_config.get("schema_version"),
            "ml_schema": ml_config.get("schema_version"),
            "explanation_schema": explanation_config.get("schema_version"),
            "model_sha256": sha256_file(args.model),
            "calibration_id": preprocessing.get("calibration", {}).get("calibration_id") if isinstance(preprocessing.get("calibration"), dict) else None,
        },
        "explanation_api": {
            "provider": analysis["explanation"]["provider"],
            "model": analysis["explanation"]["model"],
            "api_called": analysis["explanation"]["api_called"],
            "replaceable_protocol": explanation_config.get("compatible_protocol"),
            "control_authority": "none",
        },
        "inputs": {str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): sha256_file(path) for path in input_paths},
        "configuration": {
            str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): sha256_file(path)
            for path in (args.algorithm_config, args.ml_config, args.explanation_config, args.model)
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (joint_path, timeline_path, analysis_path, tokens_path, *chart_paths)
        },
        "replay_acceptance": {
            "input_reconstruction_equal": analysis["replay"]["input_reconstruction_equal"],
            "deterministic_state_equal": analysis["replay"]["deterministic_state_equal"],
            "final_analysis_equal": analysis["replay"]["final_analysis_equal"],
        },
        "control_policy": analysis["control_policy"],
        "limitations": analysis["evidence_limits"],
    }
    manifest["session_fingerprint"] = __import__("hashlib").sha256(
        json.dumps({"inputs": manifest["inputs"], "configuration": manifest["configuration"], "session_id": args.session_id}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "passed",
        "output_dir": str(args.output_dir),
        "manifest": str(manifest_path),
        "report": None if not chart_paths else str(chart_paths[0]),
        "valid_sample_pct": analysis["deterministic_control"]["metrics"]["valid_sample_pct"],
        "shadow_tokens": len(analysis["ml_shadow"]["tokens"]),
        "external_api_called": analysis["explanation"]["api_called"],
        "replay_equal": analysis["replay"]["final_analysis_equal"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
