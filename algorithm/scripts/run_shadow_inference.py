#!/usr/bin/env python3
"""Run a saved activity model on joint_state CSV without affecting controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shewrist.data import load_config, load_joint_state_csv, write_json
from shewrist.ml import ShadowActivityPipeline
from shewrist.ml_data import build_joint_state_windows
from shewrist.tokens import build_inertial_tokens, feedback_from_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joint_state", type=Path)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "outputs/ml/activity_cnn_hmm_shadow.npz")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/ml_activity.json")
    parser.add_argument("--session-id", default="offline-replay")
    parser.add_argument("--evidence-type", choices=("replay", "bench", "simulation", "human"), default="replay")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/ml/shadow_inference.json")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = ShadowActivityPipeline.load(args.model)
    joint_state = load_joint_state_csv(args.joint_state)
    dataset = build_joint_state_windows(joint_state, config, args.session_id)
    prediction = pipeline.predict(dataset)
    tokens = build_inertial_tokens(
        prediction.accepted_labels,
        prediction.confidence,
        dataset.start_s,
        dataset.end_s,
        dataset.mean_quality,
        dataset.windows,
        pipeline.class_names,
        pipeline.feature_names,
        args.session_id,
        sequence_ids=dataset.sequence_ids,
        evidence_type=args.evidence_type,
    )
    payload = {
        "schema_version": 1,
        "operating_mode": "shadow",
        "alarm_control_effect": "none",
        "mechanical_control_effect": "none",
        "source_file": str(args.joint_state),
        "model_file": str(args.model),
        "window_count": len(dataset),
        "accepted_window_count": int((prediction.accepted_labels >= 0).sum()),
        "rejected_window_count": int((prediction.accepted_labels < 0).sum()),
        "tokens": [token.to_dict() for token in tokens],
        "feedback": [feedback_from_token(token) for token in tokens],
        "warning": "Activity labels are an experimental public-dataset mapping. They do not estimate strain, pain, disease risk, pressure safety, or treatment need.",
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "tokens": len(tokens)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
