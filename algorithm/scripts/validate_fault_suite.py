#!/usr/bin/env python3
"""Run the system-level raw dual-IMU fault suite on one public replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import load_annotations, load_config, write_csv, write_json
from shewrist.explanation import provider_from_config
from shewrist.faults import default_fault_suite
from shewrist.ml import ShadowActivityPipeline
from shewrist.session import analyze_session, prepare_public_joint_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/raw/UpperBodyMovements")
    parser.add_argument("--subject", default="subject01")
    parser.add_argument("--set", dest="set_name", default="set2")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "outputs/ml/activity_cnn_hmm_shadow.npz")
    parser.add_argument("--algorithm-config", type=Path, default=PROJECT_ROOT / "config/thresholds.yaml")
    parser.add_argument("--ml-config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--explanation-config", type=Path, default=PROJECT_ROOT / "config/explanation_api.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/fault_suite")
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()

    algorithm_config = load_config(args.algorithm_config)
    ml_config = load_config(args.ml_config)
    explanation_config = load_config(args.explanation_config)
    provider_from_config(explanation_config)
    model = ShadowActivityPipeline.load(args.model)
    annotations = load_annotations(args.dataset / "annotations.csv", args.subject, args.set_name)
    results = []
    baseline_joint = None
    for name, faults in default_fault_suite().items():
        row = {"scenario": name, "fault_count": len(faults)}
        try:
            joint, preprocessing, _ = prepare_public_joint_state(
                args.dataset,
                annotations,
                args.subject,
                args.set_name,
                algorithm_config,
                faults,
            )
            analysis, _ = analyze_session(
                joint,
                algorithm_config,
                ml_config,
                model,
                explanation_config,
                f"fault-{name}",
                "simulation" if faults else "replay",
                args.chunk_size,
            )
            if baseline_joint is None:
                baseline_joint = joint
            comparable = baseline_joint is not None and len(joint["theta_FE"]) == len(baseline_joint["theta_FE"])
            angle_delta = None
            if comparable:
                angle_delta = float(np.sqrt(np.mean((joint["theta_FE"] - baseline_joint["theta_FE"]) ** 2 + (joint["theta_RUD"] - baseline_joint["theta_RUD"]) ** 2)))
            reason_counts = preprocessing["synchronization"]["quality_reason_counts"]
            recalibration_required = any(
                "recalibration_required_after_fault" in reasons
                for reasons in reason_counts.values()
            )
            row.update({
                "outcome": "completed",
                "valid_sample_pct": analysis["deterministic_control"]["metrics"]["valid_sample_pct"],
                "rejected_windows": analysis["ml_shadow"]["rejected_window_count"],
                "deterministic_alerts": len(analysis["deterministic_control"]["alerts"]),
                "sync_gate_passed": preprocessing["synchronization"]["sync_gate_passed"],
                "recalibration_required": recalibration_required,
                "angle_rmse_vs_baseline_deg": angle_delta,
                "batch_stream_equal": analysis["replay"]["final_analysis_equal"],
                "ml_control_authority": analysis["control_policy"]["ml_control_authority"],
                "detected_reasons": json.dumps(reason_counts, ensure_ascii=False, sort_keys=True),
                "interpretation": "Observable quality fault is degraded/rejected." if name in {"dropout", "timestamp_offset_50ms", "silence", "saturation"} else "Sensitivity result only; independent reference is required to prove detection/compensation." if faults else "Reference replay.",
            })
        except ValueError as exc:
            row.update({
                "outcome": "rejected",
                "valid_sample_pct": "",
                "rejected_windows": "",
                "deterministic_alerts": "",
                "sync_gate_passed": False,
                "recalibration_required": False,
                "angle_rmse_vs_baseline_deg": "",
                "batch_stream_equal": "",
                "ml_control_authority": "none",
                "detected_reasons": str(exc),
                "interpretation": "Invalid input was rejected before inference.",
            })
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    by_name = {row["scenario"]: row for row in results}
    acceptance = {
        "all_scenarios_executed": len(results) == len(default_fault_suite()),
        "out_of_order_rejected": by_name["out_of_order"]["outcome"] == "rejected",
        "timestamp_offset_sync_rejected": by_name["timestamp_offset_50ms"]["sync_gate_passed"] is False,
        "silence_quality_degraded": float(by_name["silence"]["valid_sample_pct"]) < float(by_name["baseline"]["valid_sample_pct"]),
        "saturation_quality_degraded": float(by_name["saturation"]["valid_sample_pct"]) < float(by_name["baseline"]["valid_sample_pct"]),
        "observable_faults_require_recalibration": all(
            bool(by_name[name]["recalibration_required"])
            for name in ("dropout", "silence", "saturation")
        ),
        "observable_faults_do_not_add_alerts": all(
            int(by_name[name]["deterministic_alerts"]) <= int(by_name["baseline"]["deterministic_alerts"])
            for name in ("dropout", "timestamp_offset_50ms", "silence", "saturation")
        ),
        "ml_never_controls": all(row["ml_control_authority"] == "none" for row in results),
    }
    acceptance["passed"] = all(acceptance.values())
    payload = {
        "schema_version": 1,
        "subject": args.subject,
        "set": args.set_name,
        "evidence_type": "simulation",
        "results": results,
        "acceptance": acceptance,
        "limits": [
            "Injected faults validate software behavior, not target-hardware fault rates.",
            "Gyro bias, rigid mounting rotation, and gradual slip may be unobservable without an independent reference or hardware metadata.",
            "No fault scenario grants ML or an explanation provider control authority.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    write_csv(args.output_dir / "fault_matrix.csv", fields, results)
    write_json(args.output_dir / "fault_report.json", payload)
    if not acceptance["passed"]:
        raise SystemExit("fault-suite acceptance failed")
    print(json.dumps({"status": "passed", "report": str(args.output_dir / "fault_report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
