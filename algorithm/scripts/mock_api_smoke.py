#!/usr/bin/env python3
"""Exercise the frozen SheWrist HTTP contract with generated local data."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_JOB_STATES = {"succeeded", "failed"}
TRIAL_SETTINGS = {
    "A": {"support_level": 0, "reminder_enabled": False},
    "B": {"support_level": 1, "reminder_enabled": False},
    "C": {"support_level": 1, "reminder_enabled": True},
}


def request_json(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return response.status, json.loads(payload)
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            details: dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError:
            details = {"raw_response": payload}
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {json.dumps(details, ensure_ascii=False)}") from exc


def encode_multipart(metadata: dict[str, Any], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = f"----SheWristSmoke{uuid.uuid4().hex}"
    chunks = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="metadata"\r\n',
        b"Content-Type: application/json\r\n\r\n",
        json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        b"\r\n",
    ]
    for field, (filename, payload) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
                b"Content-Type: text/csv\r\n\r\n",
                payload,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def build_joint_state_csv(duration_s: float = 12.0, sample_rate_hz: float = 50.0) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["timestamp_ms", "theta_FE", "theta_RUD", "quality"])
    for index in range(int(duration_s * sample_rate_hz) + 1):
        writer.writerow([1000.0 * index / sample_rate_hz, 35.0, 0.0, 1.0])
    return output.getvalue().encode("utf-8")


def build_mechanical_csv(
    condition: str,
    duration_s: float = 12.0,
    safety_symptom_start_s: float | None = None,
) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "device_ms",
            "condition",
            "support_level",
            "fsr_raw_adc",
            "discomfort_nrs",
            "safety_symptom_flag",
            "user_continues",
        ]
    )
    settings = TRIAL_SETTINGS[condition]
    for index in range(int(duration_s * 2.0) + 1):
        timestamp_s = index / 2.0
        writer.writerow(
            [
                500000.0 + timestamp_s * 1000.0,
                condition,
                settings["support_level"],
                900.0 + 400.0 * settings["support_level"] + 10.0 * math.sin(timestamp_s),
                2.0,
                int(safety_symptom_start_s is not None and timestamp_s >= safety_symptom_start_s),
                1,
            ]
        )
    return output.getvalue().encode("utf-8")


def build_raw_imu_csv(*, calibration: bool, duration_s: float, sample_rate_hz: float = 100.0) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["device_ms", "sensor_id", "ax", "ay", "az", "gx", "gy", "gz", "quality"])
    count = int(duration_s * sample_rate_hz) + 1
    for index in range(count):
        timestamp_s = index / sample_rate_hz
        phase = float(index)
        forearm_accel = [1e-4 * math.sin(phase), 1e-4 * math.cos(phase), 9.80665 + 1e-4 * math.sin(0.3 * phase)]
        hand_accel = list(forearm_accel)
        forearm_gyro = [1e-4 * math.sin(0.7 * phase), 1e-4 * math.cos(0.9 * phase), 1e-4 * math.sin(0.3 * phase)]
        hand_gyro = list(forearm_gyro)
        if calibration:
            if 1.0 <= timestamp_s <= 1.9:
                hand_gyro[0] -= 0.5
            if 2.0 <= timestamp_s <= 2.9:
                hand_gyro[0] += 0.5
            if 4.0 <= timestamp_s <= 4.9:
                hand_gyro[1] += 0.5
        elif 1.0 <= timestamp_s <= 3.5:
            hand_gyro[0] += 0.25
        for sensor_id, accel, gyro in (
            ("forearm", forearm_accel, forearm_gyro),
            ("hand", hand_accel, hand_gyro),
        ):
            writer.writerow([700000.0 + timestamp_s * 1000.0, sensor_id, *accel, *gyro, 1.0])
    return output.getvalue().encode("utf-8")


def base_metadata(session_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "input_type": "joint_state",
        "evidence_type": "simulation",
        "timestamp_basis": "session_relative_ms",
        "options": {
            "enable_ml_shadow": True,
            "threshold_version": "engineering_v1",
            "explanation_provider": "local_template",
            "enable_external_api": False,
            "generate_charts": False,
        },
    }


def trial_metadata(condition: str, session_id: str) -> dict[str, Any]:
    metadata = base_metadata(session_id)
    metadata.update({"condition": condition, **TRIAL_SETTINGS[condition]})
    return metadata


def raw_metadata(session_id: str) -> dict[str, Any]:
    metadata = trial_metadata("A", session_id)
    metadata.update(
        {
            "input_type": "raw_dual_imu",
            "timestamp_basis": "device_ms",
            "sensor_units": {"acceleration": "m/s2", "angular_velocity": "rad/s"},
            "sensors": [
                {
                    "sensor_id": "forearm",
                    "placement": "right_distal_forearm",
                    "coordinate_frame": "sensor_local",
                },
                {
                    "sensor_id": "hand",
                    "placement": "right_hand_third_metacarpal_dorsum",
                    "coordinate_frame": "sensor_local",
                },
            ],
            "calibration": {
                "calibration_id": "SMOKE-CAL-001",
                "mode": "neutral_plus_static_validation",
                "segments": [
                    {"type": "neutral", "start_ms": 0, "end_ms": 900},
                    {"type": "flexion", "start_ms": 1000, "end_ms": 1900},
                    {"type": "extension", "start_ms": 2000, "end_ms": 2900},
                    {"type": "ulnar_deviation", "start_ms": 4000, "end_ms": 4900},
                ],
            },
        }
    )
    return metadata


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def wait_for_health(base_url: str, process: subprocess.Popen[bytes] | None, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "API did not respond"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"API process exited during startup with code {process.returncode}")
        try:
            status, payload = request_json("GET", f"{base_url}/healthz", timeout=1.0)
            if status == 200 and payload.get("status") == "ok":
                return payload
            last_error = f"unexpected health response: HTTP {status} {payload}"
        except (OSError, RuntimeError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"API health check timed out after {timeout_s:.1f}s: {last_error}")


def poll_job(base_url: str, status_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, last_payload = request_json("GET", f"{base_url}{status_url}")
        require(status == 200, f"job polling returned HTTP {status}")
        if last_payload.get("status") in TERMINAL_JOB_STATES:
            return last_payload
        time.sleep(0.2)
    raise RuntimeError(f"analysis job timed out after {timeout_s:.1f}s; last response: {last_payload}")


def submit_job(
    base_url: str,
    metadata: dict[str, Any],
    files: dict[str, tuple[str, bytes]],
    timeout_s: float,
) -> dict[str, Any]:
    body, boundary = encode_multipart(metadata, files)
    status, created = request_json(
        "POST",
        f"{base_url}/api/v1/analysis-jobs",
        body=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Idempotency-Key": f"smoke-{uuid.uuid4().hex}",
        },
        timeout=timeout_s,
    )
    require(status == 202, f"job creation returned HTTP {status}, expected 202")
    require(created.get("session_id") == metadata["session_id"], "create response changed session_id")
    job = poll_job(base_url, str(created["status_url"]), timeout_s)
    require(job.get("status") == "succeeded", f"analysis failed: {job.get('error')}")
    result_status, result = request_json("GET", f"{base_url}{created['result_url']}", timeout=timeout_s)
    require(result_status == 200, f"result endpoint returned HTTP {result_status}")
    return result


def run_smoke(base_url: str, timeout_s: float) -> dict[str, Any]:
    prefix = f"smoke-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    trial_results: dict[str, dict[str, Any]] = {}
    for condition in ("A", "B", "C"):
        session_id = f"{prefix}-{condition.lower()}"
        trial_results[condition] = submit_job(
            base_url,
            trial_metadata(condition, session_id),
            {
                "data_file": ("joint_state.csv", build_joint_state_csv()),
                "mechanical_file": ("mechanical.csv", build_mechanical_csv(condition)),
            },
            timeout_s,
        )
    for condition in ("A", "B"):
        metrics = trial_results[condition]["metrics"]
        require(metrics["alert_count"] == 0, f"condition {condition} emitted an actual angle alert")
        require(metrics["would_alert_count"] == 1, f"condition {condition} did not record its silent alert")
    require(trial_results["C"]["metrics"]["alert_count"] == 1, "condition C did not emit its angle alert")
    require(trial_results["C"]["metrics"]["would_alert_count"] == 1, "condition C lost would_alert audit")
    for condition, result in trial_results.items():
        require(result["trial_condition"]["support_level"] == TRIAL_SETTINGS[condition]["support_level"], f"condition {condition} support mismatch")
        require(result["fsr_proxy"]["available"] is True, f"condition {condition} FSR proxy missing")
        require(result["fsr_proxy"]["calibrated_to_pressure"] is False, f"condition {condition} FSR incorrectly marked calibrated")
        require(result["channels"]["pressure"]["available"] is False, f"condition {condition} raw FSR became pressure")
        require(result["metrics"]["mechanical_recommendation_count"] == 0, f"condition {condition} enabled tightening advice")

    raw_session_id = f"{prefix}-raw-cal"
    raw_result = submit_job(
        base_url,
        raw_metadata(raw_session_id),
        {
            "data_file": ("task.csv", build_raw_imu_csv(calibration=False, duration_s=4.0)),
            "mechanical_file": ("mechanical.csv", build_mechanical_csv("A", duration_s=4.0, safety_symptom_start_s=2.0)),
            "calibration_file": ("calibration.csv", build_raw_imu_csv(calibration=True, duration_s=6.0)),
        },
        timeout_s,
    )
    require(raw_result["analysis_status"] == "accepted", f"separate CAL result was rejected: {raw_result['rejection_reasons']}")
    require(raw_result["sensor_installation"]["nodes"][1]["placement"] == "right_hand_third_metacarpal_dorsum", "hand placement contract changed")
    require(raw_result["calibration"]["application_mode"] == "separate_calibration_file", "calibration_file was not applied")
    require(raw_result["calibration"]["task_neutral_reestimated"] is False, "an undocumented task-neutral step was introduced")
    require(raw_result["metrics"]["safety_stop_count"] == 1, "safety symptom did not trigger one stop episode")
    require(raw_result["metrics"]["max_pressure_kpa"] is None, "uncalibrated FSR produced a kPa metric")

    timeline_status, timeline = request_json(
        "GET",
        f"{base_url}/api/v1/sessions/{raw_session_id}/timeline?offset=195&limit=15",
        timeout=timeout_s,
    )
    require(timeline_status == 200 and len(timeline.get("items", [])) == 15, "timeline pagination failed")
    items = timeline["items"]
    require(any(item["safety_symptom"] is True for item in items), "timeline omitted safety_symptom")
    require(all(item["fsr_raw"] is not None for item in items), "timeline omitted fsr_raw")
    require(all(item["pressure_zone"] is None for item in items), "raw FSR was exposed as calibrated pressure")

    tokens_status, tokens = request_json(
        "GET",
        f"{base_url}/api/v1/sessions/{raw_session_id}/tokens",
        timeout=timeout_s,
    )
    require(tokens_status == 200 and tokens.get("operating_mode") == "shadow", "token endpoint failed")
    require(raw_result["control_policy"]["ml_control_authority"] == "none", "ML gained control authority")
    require(raw_result["control_policy"]["llm_control_authority"] == "none", "LLM gained control authority")
    require(raw_result["explanation"]["api_called"] is False, "external explanation API was unexpectedly called")

    return {
        "status": "PASS",
        "base_url": base_url,
        "sessions": {condition: result["session_id"] for condition, result in trial_results.items()},
        "raw_calibration_session": raw_session_id,
        "trial_alert_counts": {condition: result["metrics"]["alert_count"] for condition, result in trial_results.items()},
        "trial_would_alert_counts": {condition: result["metrics"]["would_alert_count"] for condition, result in trial_results.items()},
        "separate_calibration": raw_result["calibration"]["application_mode"],
        "fsr_unit": raw_result["fsr_proxy"]["unit"],
        "safety_stop_count": raw_result["metrics"]["safety_stop_count"],
        "shadow_token_count": len(tokens["tokens"]),
        "external_api_called": raw_result["explanation"]["api_called"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="host used when starting a temporary local API")
    parser.add_argument("--port", type=int, default=8000, help="port used when starting a temporary local API")
    parser.add_argument("--base-url", help="test an already-running API instead of starting one")
    parser.add_argument("--timeout", type=float, default=30.0, help="startup and analysis timeout in seconds")
    args = parser.parse_args()

    process: subprocess.Popen[bytes] | None = None
    log_handle: Any = None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    base_url = args.base_url.rstrip("/") if args.base_url else f"http://{args.host}:{args.port}"
    try:
        if args.base_url is None:
            output_parent = PROJECT_ROOT / "outputs"
            output_parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(prefix="api-smoke-", dir=output_parent)
            temporary_path = Path(temporary.name)
            log_handle = (temporary_path / "server.log").open("wb")
            environment = os.environ.copy()
            environment["SHEWRIST_API_OUTPUT_ROOT"] = str(temporary_path / "api")
            process = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "scripts/run_api.py"), "--host", args.host, "--port", str(args.port)],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        health = wait_for_health(base_url, process, args.timeout)
        summary = run_smoke(base_url, args.timeout)
        summary["health"] = health
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "base_url": base_url, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        if log_handle is not None:
            log_handle.flush()
            log_path = Path(log_handle.name)
            if log_path.exists():
                server_log = log_path.read_text(encoding="utf-8", errors="replace").strip()
                if server_log:
                    print("\nAPI server log:\n" + server_log, file=sys.stderr)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if log_handle is not None:
            log_handle.close()
        if temporary is not None:
            temporary.cleanup()
        cache = PROJECT_ROOT / "scripts/__pycache__"
        if cache.exists():
            shutil.rmtree(cache)


if __name__ == "__main__":
    raise SystemExit(main())
