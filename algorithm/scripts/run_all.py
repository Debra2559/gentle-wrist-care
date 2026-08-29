#!/usr/bin/env python3
"""Run every reproducible analysis and offline-v0.8 acceptance check."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def environment() -> dict[str, str]:
    values = os.environ.copy()
    values["PYTHONPATH"] = str(ROOT / "src")
    values["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    values["PYTHONDONTWRITEBYTECODE"] = "1"
    return values


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        env=environment(),
        check=True,
    )


def run_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        env=environment(),
        check=True,
    )


def main() -> None:
    run("analyze_public_dataset.py")
    run(
        "analyze_public_dataset.py",
        "--algorithm",
        "mahony",
        "--no-joint-output",
        "--output",
        "outputs/public_dataset/mahony_summary.csv",
        "--json",
        "outputs/public_dataset/mahony_validation.json",
    )
    run(
        "analyze_public_dataset.py",
        "--use-magnetometer",
        "--no-joint-output",
        "--output",
        "outputs/public_dataset/madgwick_9axis_summary.csv",
        "--json",
        "outputs/public_dataset/madgwick_9axis_validation.json",
    )
    run("validate_opto_reference.py")
    run("generate_demo.py")
    run(
        "analyze_joint_state.py",
        "examples/synthetic_abc/SYN01_C_joint_state.csv",
        "--mechanical",
        "examples/synthetic_abc/SYN01_C_mechanical.csv",
        "--lever-arm-m",
        "0.02",
        "--output",
        "outputs/trial_summary.json",
    )
    run("train_activity_model.py", "--output-dir", "outputs/ml")
    run(
        "analyze_with_shadow.py",
        "data/processed/public_subject01_set2_joint_state.csv",
        "--model",
        "outputs/ml/activity_cnn_hmm_shadow.npz",
        "--session-id",
        "public-subject01",
        "--evidence-type",
        "replay",
        "--output",
        "outputs/ml/combined_public_subject01.json",
    )
    run("select_activity_model.py", "--epochs", "20", "--output-dir", "outputs/model_selection")
    run(
        "run_offline_session.py",
        "--public-subject",
        "subject01",
        "--session-id",
        "public-subject01-offline-v08",
        "--evidence-type",
        "replay",
        "--output-dir",
        "outputs/offline_session",
    )
    run("validate_fault_suite.py", "--subject", "subject01", "--output-dir", "outputs/fault_suite")
    run("visualize_project_status.py")
    run("visualize_offline_v08.py")
    run_tests()


if __name__ == "__main__":
    main()
