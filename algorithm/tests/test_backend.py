import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shewrist.backend import AnalysisService, BackendError, BackendSettings, parse_raw_dual_imu, validate_metadata
from shewrist.api_models import SessionResult, TimelineResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def joint_state_csv(duration_s=4.0, sample_rate_hz=50.0, valid_sample_count=None):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality"])
    timestamps = np.arange(int(duration_s * sample_rate_hz) + 1) / sample_rate_hz
    fe = 20.0 * np.sin(2.0 * np.pi * 0.5 * timestamps)
    rud = 12.0 * np.sin(2.0 * np.pi * 0.35 * timestamps)
    speed = np.sqrt(np.gradient(fe, timestamps) ** 2 + np.gradient(rud, timestamps) ** 2)
    for index, timestamp in enumerate(timestamps):
        quality = 1.0 if valid_sample_count is None or index < valid_sample_count else 0.0
        writer.writerow([timestamp * 1000.0, fe[index], rud[index], "", speed[index], "TEST-CAL", quality])
    return output.getvalue().encode("utf-8")


def mechanical_csv(
    duration_s=4.0,
    discomfort_start_s=None,
    user_continues=True,
    *,
    device_time=False,
    fsr_raw_adc=None,
    safety_symptom_start_s=None,
    discomfort_nrs=0.0,
):
    output = io.StringIO()
    writer = csv.writer(output)
    timestamp_field = "device_ms" if device_time else "timestamp_ms"
    writer.writerow([timestamp_field, "fsr_raw_adc", "discomfort", "discomfort_nrs", "safety_symptom_flag", "user_continues"])
    origin = 500000.0 if device_time else 0.0
    for second in np.arange(0.0, duration_s + 0.5, 0.5):
        discomfort = int(discomfort_start_s is not None and second >= discomfort_start_s)
        symptom = int(safety_symptom_start_s is not None and second >= safety_symptom_start_s)
        writer.writerow([
            origin + second * 1000.0,
            "" if fsr_raw_adc is None else fsr_raw_adc + second,
            discomfort,
            discomfort_nrs,
            symptom,
            int(user_continues),
        ])
    return output.getvalue().encode("utf-8")


def raw_dual_imu_csv(
    bad_neutral=False,
    duration_s=6.0,
    sample_rate_hz=100.0,
    *,
    device_time=False,
    include_fsr=False,
    calibration_motion=True,
):
    output = io.StringIO()
    writer = csv.writer(output)
    timestamp_field = "device_ms" if device_time else "timestamp_ms"
    writer.writerow([timestamp_field, "sensor_id", "ax", "ay", "az", "gx", "gy", "gz", "quality", "fsr_raw_adc"])
    timestamps = np.arange(int(duration_s * sample_rate_hz) + 1) / sample_rate_hz
    origin = 700000.0 if device_time else 0.0
    phase = np.arange(len(timestamps), dtype=float)
    forearm_accel = np.column_stack(
        (
            1e-4 * np.sin(phase),
            1e-4 * np.cos(phase),
            9.80665 + 1e-4 * np.sin(0.3 * phase),
        )
    )
    hand_accel = forearm_accel.copy()
    forearm_gyro = np.column_stack(
        (
            1e-4 * np.sin(0.7 * phase),
            1e-4 * np.cos(0.9 * phase),
            1e-4 * np.sin(0.3 * phase),
        )
    )
    hand_gyro = forearm_gyro.copy()
    if calibration_motion:
        hand_gyro[(timestamps >= 1.0) & (timestamps <= 1.9), 0] -= 0.5
        hand_gyro[(timestamps >= 2.0) & (timestamps <= 2.9), 0] += 0.5
        hand_gyro[(timestamps >= 3.0) & (timestamps <= 3.9), 1] -= 0.5
        hand_gyro[(timestamps >= 4.0) & (timestamps <= 4.9), 1] += 0.5
    else:
        hand_gyro[(timestamps >= 1.0) & (timestamps <= 3.5), 0] = 0.25
    if bad_neutral:
        hand_accel[timestamps <= 0.9, 2] = 12.0
    for index, timestamp in enumerate(timestamps):
        for node, accel, gyro in (
            ("forearm", forearm_accel[index], forearm_gyro[index]),
            ("hand", hand_accel[index], hand_gyro[index]),
        ):
            writer.writerow([
                origin + timestamp * 1000.0,
                node,
                *accel,
                *gyro,
                1.0,
                1000.0 + 10.0 * np.sin(timestamp) if include_fsr else "",
            ])
    return output.getvalue().encode("utf-8")


def joint_metadata(session_id="test-session"):
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


def trial_metadata(condition, session_id):
    metadata = joint_metadata(session_id)
    settings = {
        "A": {"support_level": 0, "reminder_enabled": False},
        "B": {"support_level": 1, "reminder_enabled": False},
        "C": {"support_level": 1, "reminder_enabled": True},
    }[condition]
    metadata.update({"condition": condition, **settings})
    return metadata


def high_angle_joint_state_csv(duration_s=12.0, sample_rate_hz=50.0):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp_ms", "theta_FE", "theta_RUD", "quality"])
    for timestamp in np.arange(int(duration_s * sample_rate_hz) + 1) / sample_rate_hz:
        writer.writerow([timestamp * 1000.0, 35.0, 0.0, 1.0])
    return output.getvalue().encode("utf-8")


def raw_metadata(session_id="raw-session"):
    metadata = joint_metadata(session_id)
    metadata.update(
        {
            "input_type": "raw_dual_imu",
            "evidence_type": "bench",
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
                "segments": [
                    {"type": "neutral", "start_ms": 0, "end_ms": 1000},
                    {"type": "flexion", "start_ms": 1000, "end_ms": 2000},
                    {"type": "extension", "start_ms": 2000, "end_ms": 3000},
                    {"type": "radial_deviation", "start_ms": 3000, "end_ms": 4000},
                    {"type": "ulnar_deviation", "start_ms": 4000, "end_ms": 5000},
                ]
            },
        }
    )
    metadata["calibration"]["segments"] = [
        {"type": "neutral", "start_ms": 0, "end_ms": 900},
        {"type": "flexion", "start_ms": 1000, "end_ms": 1900},
        {"type": "extension", "start_ms": 2000, "end_ms": 2900},
        {"type": "ulnar_deviation", "start_ms": 4000, "end_ms": 4900},
    ]
    return metadata


def calibration_metadata(calibration_id="CAL-STORE-001", participant_id=None):
    metadata = {
        "schema_version": "1.0",
        "calibration_id": calibration_id,
        "sensor_units": {"acceleration": "m/s2", "angular_velocity": "rad/s"},
        "sensors": [
            {"sensor_id": "forearm", "placement": "right_distal_forearm", "coordinate_frame": "sensor_local"},
            {"sensor_id": "hand", "placement": "right_hand_third_metacarpal_dorsum", "coordinate_frame": "sensor_local"},
        ],
        "calibration": {
            "segments": [
                {"type": "neutral", "start_ms": 0, "end_ms": 900},
                {"type": "flexion", "start_ms": 1000, "end_ms": 1900},
                {"type": "extension", "start_ms": 2000, "end_ms": 2900},
                {"type": "ulnar_deviation", "start_ms": 4000, "end_ms": 4900},
            ]
        },
    }
    if participant_id is not None:
        metadata["participant_id"] = participant_id
    return metadata


def stored_profile_metadata(session_id, calibration_id):
    metadata = raw_metadata(session_id)
    metadata["calibration"] = {"use_stored_profile": True, "calibration_id": calibration_id}
    return metadata


class MetadataTests(unittest.TestCase):
    def test_human_data_requires_compliance_confirmation(self):
        metadata = joint_metadata()
        metadata["evidence_type"] = "human"
        with self.assertRaises(BackendError) as context:
            validate_metadata(metadata)
        self.assertEqual(context.exception.code, "HUMAN_DATA_CONFIRMATION_REQUIRED")

    def test_raw_input_requires_all_directional_calibration_segments(self):
        metadata = raw_metadata()
        metadata["calibration"]["segments"] = [
            {"type": "neutral", "start_ms": 0, "end_ms": 1000},
            {"type": "flexion", "start_ms": 1000, "end_ms": 2000},
        ]
        with self.assertRaises(BackendError) as context:
            validate_metadata(metadata)
        self.assertEqual(context.exception.code, "CALIBRATION_REQUIRED")
        self.assertIn("extension", context.exception.details["missing"])

    def test_raw_input_enforces_target_sensor_placements(self):
        metadata = raw_metadata()
        metadata["sensors"][1]["placement"] = "right_hand_dorsum"
        with self.assertRaises(BackendError) as context:
            validate_metadata(metadata)
        self.assertEqual(context.exception.code, "INVALID_SENSOR_PLACEMENT")
        self.assertEqual(context.exception.details["sensor_id"], "hand")

    def test_raw_input_accepts_left_target_pair(self):
        metadata = raw_metadata()
        metadata["sensors"][0]["placement"] = "left_distal_forearm"
        metadata["sensors"][1]["placement"] = "left_hand_third_metacarpal_dorsum"
        normalized = validate_metadata(metadata)
        self.assertEqual(normalized["sensors"][1]["sensor_id"], "hand")

    def test_raw_dual_imu_parser_supports_interleaved_nodes(self):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp_ms", "sensor_id", "ax", "ay", "az", "gx", "gy", "gz", "quality"])
        for index in range(10):
            for node in ("forearm", "hand"):
                writer.writerow([index * 10, node, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0, 0.9])
        raw, quality, rate = parse_raw_dual_imu(output.getvalue().encode("utf-8"))
        self.assertEqual(raw["forearm_accel"].shape, (10, 3))
        self.assertEqual(raw["hand_gyro"].shape, (10, 3))
        self.assertEqual(set(quality), {"forearm", "hand"})
        self.assertAlmostEqual(rate, 100.0)

    def test_raw_parser_normalizes_device_ms_and_keeps_fsr_proxy(self):
        raw, _, rate = parse_raw_dual_imu(raw_dual_imu_csv(device_time=True, include_fsr=True))
        self.assertAlmostEqual(raw["forearm_timestamp_s"][0], 0.0)
        self.assertAlmostEqual(raw["hand_timestamp_s"][0], 0.0)
        self.assertEqual(len(raw["fsr_raw_adc"]), 601)
        self.assertAlmostEqual(rate, 100.0)

    def test_raw_parser_reports_an_entirely_missing_node(self):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["device_ms", "sensor_id", "ax", "ay", "az", "gx", "gy", "gz"])
        for index in range(10):
            writer.writerow([1000 + index * 10, "forearm", 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0])
        with self.assertRaises(BackendError) as context:
            parse_raw_dual_imu(output.getvalue().encode("utf-8"))
        self.assertEqual(context.exception.code, "MISSING_SENSOR_NODE")
        self.assertEqual(context.exception.details["missing"], ["hand"])

    def test_trial_condition_settings_are_frozen(self):
        normalized = validate_metadata(trial_metadata("B", "condition-b"))
        self.assertEqual(normalized["support_level"], 1)
        self.assertFalse(normalized["reminder_enabled"])
        invalid = trial_metadata("C", "bad-condition")
        invalid["support_level"] = 0
        with self.assertRaises(BackendError) as context:
            validate_metadata(invalid)
        self.assertEqual(context.exception.code, "INVALID_TRIAL_CONDITION")


class AnalysisServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        defaults = BackendSettings.default(PROJECT_ROOT)
        settings = BackendSettings(
            project_root=PROJECT_ROOT,
            output_root=Path(self.temporary.name) / "api",
            algorithm_config=defaults.algorithm_config,
            ml_config=defaults.ml_config,
            explanation_config=defaults.explanation_config,
            model_path=defaults.model_path,
        )
        self.service = AnalysisService(settings)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_joint_state_job_has_stable_public_contract(self):
        created = self.service.create_job(joint_metadata(), joint_state_csv(), "joint_state.csv")
        self.assertEqual(created["status"], "queued")
        self.service.run_job(created["job_id"])
        job = self.service.get_job(created["job_id"])
        self.assertEqual(job["status"], "succeeded")
        result = self.service.get_result("test-session")
        SessionResult(**result)
        self.assertEqual(result["analysis_status"], "accepted")
        self.assertEqual(result["evidence_type"], "simulation")
        self.assertEqual(result["channels"]["wrist_angles"]["source"], "provided")
        self.assertFalse(result["channels"]["pressure"]["available"])
        self.assertFalse(result["channels"]["discomfort"]["available"])
        self.assertFalse(result["channels"]["user_continues"]["available"])
        self.assertIsNone(result["metrics"]["max_pressure_kpa"])
        self.assertIsNone(result["metrics"]["safety_stop_count"])
        self.assertEqual(result["ml_shadow"]["safety_effect"], "none")
        self.assertEqual(result["control_policy"]["ml_control_authority"], "none")
        self.assertEqual(result["control_policy"]["llm_control_authority"], "none")
        self.assertFalse(result["explanation"]["api_called"])
        self.assertTrue(any(item["name"] == "manifest.json" for item in result["artifacts"]))
        tokens = self.service.get_tokens("test-session")
        self.assertEqual(tokens["operating_mode"], "shadow")
        timeline = self.service.get_timeline("test-session", 0, 3)
        TimelineResponse(**timeline)
        self.assertEqual(timeline["total"], 201)
        self.assertEqual(len(timeline["items"]), 3)
        self.assertIsNone(timeline["items"][0]["pressure_zone"])
        self.assertIsNone(timeline["items"][0]["discomfort"])
        self.assertIsNone(timeline["items"][0]["user_continues"])
        self.assertIsNone(timeline["items"][0]["safety_stop"])

    def test_session_with_some_but_too_few_valid_samples_is_rejected(self):
        created = self.service.create_job(
            joint_metadata("low-validity"),
            joint_state_csv(valid_sample_count=1),
            "input.csv",
        )
        self.service.run_job(created["job_id"])
        result = self.service.get_result("low-validity")
        self.assertEqual(result["analysis_status"], "rejected")
        self.assertIn("insufficient_valid_angle_samples", result["rejection_reasons"])
        self.assertLess(result["data_quality"]["valid_sample_pct"], result["data_quality"]["valid_sample_pct_min"])
        self.assertFalse(result["data_quality"]["valid_sample_gate_passed"])

    def test_discomfort_channel_triggers_stop_without_pressure(self):
        created = self.service.create_job(
            joint_metadata("discomfort-only"),
            joint_state_csv(),
            "input.csv",
            mechanical_csv(discomfort_start_s=2.0),
            "mechanical.csv",
        )
        self.service.run_job(created["job_id"])
        result = self.service.get_result("discomfort-only")
        self.assertFalse(result["channels"]["pressure"]["available"])
        self.assertTrue(result["channels"]["discomfort"]["available"])
        self.assertTrue(result["channels"]["user_continues"]["available"])
        self.assertEqual(result["metrics"]["safety_stop_count"], 1)
        self.assertTrue(any(item["safety_stop"] for item in result["alerts"]))
        timeline = self.service.get_timeline("discomfort-only", 100, 1)
        self.assertTrue(timeline["items"][0]["discomfort"])
        self.assertTrue(timeline["items"][0]["safety_stop"])

    def test_device_time_fsr_and_safety_symptom_are_exposed_without_kpa(self):
        created = self.service.create_job(
            trial_metadata("C", "fsr-safety"),
            high_angle_joint_state_csv(),
            "input.csv",
            mechanical=mechanical_csv(
                duration_s=12.0,
                device_time=True,
                fsr_raw_adc=1200.0,
                safety_symptom_start_s=2.0,
                discomfort_nrs=3.0,
            ),
            mechanical_filename="mechanical.csv",
        )
        self.service.run_job(created["job_id"])
        result = self.service.get_result("fsr-safety")
        SessionResult(**result)
        self.assertFalse(result["channels"]["pressure"]["available"])
        self.assertTrue(result["channels"]["fsr_raw"]["available"])
        self.assertTrue(result["channels"]["safety_symptom"]["available"])
        self.assertEqual(result["fsr_proxy"]["unit"], "adc_count")
        self.assertFalse(result["fsr_proxy"]["calibrated_to_pressure"])
        self.assertIsNone(result["metrics"]["max_pressure_kpa"])
        self.assertEqual(result["metrics"]["safety_stop_count"], 1)
        timeline = self.service.get_timeline("fsr-safety", 100, 1)
        TimelineResponse(**timeline)
        self.assertTrue(timeline["items"][0]["safety_symptom"])
        self.assertEqual(timeline["items"][0]["discomfort_nrs"], 3.0)
        self.assertIsNotNone(timeline["items"][0]["fsr_raw"])
        self.assertIsNone(timeline["items"][0]["pressure_zone"])

    def test_a_and_b_are_silent_while_c_emits_angle_alerts(self):
        results = {}
        for condition in ("A", "B", "C"):
            created = self.service.create_job(
                trial_metadata(condition, f"condition-{condition.lower()}"),
                high_angle_joint_state_csv(),
                "input.csv",
            )
            self.service.run_job(created["job_id"])
            results[condition] = self.service.get_result(f"condition-{condition.lower()}")
        for condition in ("A", "B"):
            self.assertEqual(results[condition]["metrics"]["alert_count"], 0)
            self.assertEqual(results[condition]["metrics"]["would_alert_count"], 1)
            self.assertEqual(results[condition]["control_policy"]["angle_alert_authority"], "disabled_by_trial_condition")
        self.assertEqual(results["C"]["metrics"]["alert_count"], 1)
        self.assertEqual(results["C"]["metrics"]["would_alert_count"], 1)
        self.assertEqual(results["C"]["control_policy"]["angle_alert_authority"], "deterministic_exposure_engine")
        self.assertEqual(results["C"]["metrics"]["mechanical_recommendation_count"], 0)

    def test_separate_calibration_file_applies_profile_without_task_neutral(self):
        metadata = raw_metadata("separate-cal")
        metadata.update({"condition": "A", "support_level": 0, "reminder_enabled": False, "timestamp_basis": "device_ms"})
        metadata["calibration"]["calibration_id"] = "CAL-S01-001"
        created = self.service.create_job(
            metadata,
            raw_dual_imu_csv(duration_s=4.0, device_time=True, include_fsr=True, calibration_motion=False),
            "task.csv",
            calibration=raw_dual_imu_csv(device_time=True),
            calibration_filename="calibration.csv",
        )
        self.service.run_job(created["job_id"])
        result = self.service.get_result("separate-cal")
        SessionResult(**result)
        self.assertEqual(result["analysis_status"], "accepted")
        self.assertEqual(result["calibration"]["application_mode"], "separate_calibration_file")
        self.assertFalse(result["calibration"]["task_neutral_reestimated"])
        self.assertEqual(result["calibration"]["calibration_id"], "CAL-S01-001")
        self.assertTrue(result["fsr_proxy"]["available"])
        self.assertFalse(result["channels"]["pressure"]["available"])
        manifest = self.service.get_artifact("separate-cal", "manifest.json").read_text(encoding="utf-8")
        self.assertIn('"calibration.csv"', manifest)

    def test_create_calibration_persists_profile_and_is_retrievable(self):
        created = self.service.create_calibration(
            calibration_metadata("CAL-STORE-001", participant_id="S01"),
            raw_dual_imu_csv(device_time=True),
            "calibration.csv",
        )
        self.assertEqual(created["calibration_id"], "CAL-STORE-001")
        self.assertEqual(created["status"], "passed")
        self.assertTrue(created["quality_gate_passed"])
        self.assertIsNotNone(created["neutral_quaternion"])
        self.assertEqual(created["self_url"], "/api/v1/calibrations/CAL-STORE-001")
        fetched = self.service.get_calibration("CAL-STORE-001")
        self.assertEqual(fetched["calibration_id"], "CAL-STORE-001")
        self.assertEqual(fetched["participant_id"], "S01")

    def test_stored_calibration_profile_is_reused_by_analysis_job(self):
        self.service.create_calibration(
            calibration_metadata("CAL-STORE-002"),
            raw_dual_imu_csv(device_time=True),
            "calibration.csv",
        )
        metadata = stored_profile_metadata("stored-ref", "CAL-STORE-002")
        metadata["timestamp_basis"] = "device_ms"
        created = self.service.create_job(
            metadata,
            raw_dual_imu_csv(duration_s=4.0, device_time=True, calibration_motion=False),
            "task.csv",
        )
        self.service.run_job(created["job_id"])
        result = self.service.get_result("stored-ref")
        SessionResult(**result)
        self.assertEqual(result["analysis_status"], "accepted")
        self.assertEqual(result["calibration"]["application_mode"], "stored_calibration_profile")
        self.assertEqual(result["calibration"]["source_calibration_id"], "CAL-STORE-002")
        self.assertFalse(result["calibration"]["task_neutral_reestimated"])

    def test_create_calibration_rejects_duplicate_id(self):
        self.service.create_calibration(
            calibration_metadata("CAL-DUP"),
            raw_dual_imu_csv(device_time=True),
            "calibration.csv",
        )
        with self.assertRaises(BackendError) as context:
            self.service.create_calibration(
                calibration_metadata("CAL-DUP"),
                raw_dual_imu_csv(device_time=True),
                "calibration.csv",
            )
        self.assertEqual(context.exception.code, "CALIBRATION_EXISTS")

    def test_create_calibration_rejects_moving_neutral(self):
        with self.assertRaises(BackendError) as context:
            self.service.create_calibration(
                calibration_metadata("CAL-BAD"),
                raw_dual_imu_csv(bad_neutral=True, device_time=True),
                "calibration.csv",
            )
        self.assertEqual(context.exception.code, "CALIBRATION_QUALITY_FAILED")

    def test_stored_profile_reference_requires_existing_calibration(self):
        metadata = stored_profile_metadata("missing-ref", "CAL-DOES-NOT-EXIST")
        with self.assertRaises(BackendError) as context:
            self.service.create_job(metadata, raw_dual_imu_csv(device_time=True, calibration_motion=False), "task.csv")
        self.assertEqual(context.exception.code, "CALIBRATION_NOT_FOUND")

    def test_stored_profile_reference_forbids_calibration_file(self):
        self.service.create_calibration(
            calibration_metadata("CAL-STORE-003"),
            raw_dual_imu_csv(device_time=True),
            "calibration.csv",
        )
        metadata = stored_profile_metadata("stored-with-file", "CAL-STORE-003")
        with self.assertRaises(BackendError) as context:
            self.service.create_job(
                metadata,
                raw_dual_imu_csv(device_time=True, calibration_motion=False),
                "task.csv",
                calibration=raw_dual_imu_csv(device_time=True),
                calibration_filename="calibration.csv",
            )
        self.assertEqual(context.exception.code, "INVALID_SCHEMA")

    def test_get_missing_calibration_returns_not_found(self):
        with self.assertRaises(BackendError) as context:
            self.service.get_calibration("CAL-UNKNOWN")
        self.assertEqual(context.exception.code, "CALIBRATION_NOT_FOUND")

    def test_target_raw_dual_imu_session_reports_validated_installation(self):
        created = self.service.create_job(raw_metadata("raw-target"), raw_dual_imu_csv(), "raw.csv")
        self.service.run_job(created["job_id"])
        result = self.service.get_result("raw-target")
        self.assertEqual(result["analysis_status"], "accepted")
        self.assertTrue(result["sensor_installation"]["contract_validated"])
        self.assertEqual(result["sensor_installation"]["side"], "right")
        self.assertEqual(result["calibration"]["status"], "passed")
        self.assertTrue(result["calibration"]["quality_gate_passed"])

    def test_moving_neutral_calibration_rejects_raw_session(self):
        created = self.service.create_job(raw_metadata("bad-neutral"), raw_dual_imu_csv(bad_neutral=True), "raw.csv")
        self.service.run_job(created["job_id"])
        result = self.service.get_result("bad-neutral")
        self.assertEqual(result["analysis_status"], "rejected")
        self.assertIn("calibration_quality_gate_failed", result["rejection_reasons"])
        self.assertEqual(result["calibration"]["status"], "rejected")
        self.assertFalse(result["calibration"]["quality_gate_passed"])

    def test_mechanical_boolean_channels_require_zero_or_one(self):
        invalid = b"timestamp_ms,discomfort\n0,0\n1000,2\n"
        with self.assertRaises(BackendError) as context:
            self.service.create_job(joint_metadata("bad-bool"), joint_state_csv(), "input.csv", invalid, "mechanical.csv")
        self.assertEqual(context.exception.code, "INVALID_SCHEMA")

    def test_idempotency_reuses_identical_request_and_rejects_conflict(self):
        first = self.service.create_job(joint_metadata("idem"), joint_state_csv(), "input.csv", idempotency_key="same-key")
        second = self.service.create_job(joint_metadata("idem"), joint_state_csv(), "input.csv", idempotency_key="same-key")
        self.assertEqual(first["job_id"], second["job_id"])
        with self.assertRaises(BackendError) as context:
            self.service.create_job(joint_metadata("idem-other"), joint_state_csv(5.0), "input.csv", idempotency_key="same-key")
        self.assertEqual(context.exception.code, "IDEMPOTENCY_CONFLICT")

    def test_personal_baseline_absent_without_participant(self):
        created = self.service.create_job(joint_metadata("no-participant"), joint_state_csv(), "input.csv")
        self.service.run_job(created["job_id"])
        result = self.service.get_result("no-participant")
        SessionResult(**result)
        self.assertIsNone(result["personal_baseline"])

    def test_personal_baseline_auto_bootstrap_and_accumulates(self):
        first_meta = joint_metadata("pb-s01-1")
        first_meta["participant_id"] = "S01"
        created = self.service.create_job(first_meta, joint_state_csv(duration_s=120.0, sample_rate_hz=20.0), "input.csv")
        self.service.run_job(created["job_id"])
        result = self.service.get_result("pb-s01-1")
        SessionResult(**result)
        pb = result["personal_baseline"]
        self.assertIsNotNone(pb)
        self.assertEqual(pb["participant_id"], "S01")
        self.assertEqual(pb["status"], "provisional")
        self.assertTrue(pb["update_applied"])
        self.assertEqual(pb["session_count"], 1)
        self.assertEqual(pb["control_effect"], "none")
        self.assertIn("abs_fe_deg_p90", pb["relative_exposure"])
        self.assertEqual(pb["symptom_association"]["status"], "not_evaluable")
        self.assertTrue(any(item["name"] == "personal_baseline.json" for item in result["artifacts"]))
        artifact = self.service.get_artifact("pb-s01-1", "personal_baseline.json").read_text(encoding="utf-8")
        self.assertIn('"control_authority": "none"', artifact)

        second_meta = joint_metadata("pb-s01-2")
        second_meta["participant_id"] = "S01"
        created2 = self.service.create_job(second_meta, joint_state_csv(duration_s=120.0, sample_rate_hz=20.0), "input.csv")
        self.service.run_job(created2["job_id"])
        result2 = self.service.get_result("pb-s01-2")
        self.assertEqual(result2["personal_baseline"]["session_count"], 2)

    def test_personal_baseline_enrollment_rejects_over_long_session(self):
        meta = joint_metadata("pb-enroll-long")
        meta["participant_id"] = "S02"
        meta["options"]["personal_baseline_role"] = "enroll"
        created = self.service.create_job(meta, joint_state_csv(duration_s=360.0, sample_rate_hz=10.0), "input.csv")
        self.service.run_job(created["job_id"])
        result = self.service.get_result("pb-enroll-long")
        SessionResult(**result)
        pb = result["personal_baseline"]
        self.assertEqual(pb["status"], "rejected")
        artifact = self.service.get_artifact("pb-enroll-long", "personal_baseline.json").read_text(encoding="utf-8")
        self.assertIn("enrollment_session_exceeds_max_minutes", artifact)

    def test_artifact_download_is_allow_listed(self):
        created = self.service.create_job(joint_metadata("artifact-test"), joint_state_csv(), "input.csv")
        self.service.run_job(created["job_id"])
        with self.assertRaises(BackendError) as context:
            self.service.get_artifact("artifact-test", "metadata.json")
        self.assertEqual(context.exception.code, "ARTIFACT_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()