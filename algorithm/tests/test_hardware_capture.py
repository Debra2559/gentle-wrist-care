from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from shewrist.backend import parse_raw_dual_imu
from shewrist.hardware_capture import (
    G_TO_MPS2,
    audit_and_convert_capture,
    import_capture_directory,
)


def write_wide_capture(
    path: Path,
    *,
    sample_count: int = 12,
    timestamp_field: str = "device_us",
    interval: float = 2500.0,
    gap_index: int | None = None,
    zero_forearm_start: int | None = None,
    saturated_pressure_start: int | None = None,
) -> bytes:
    fields = [
        timestamp_field,
        "pressure_adc_raw",
        "pressure_adc_filtered",
        "pressure_load_adc",
        "hand_ax_g",
        "hand_ay_g",
        "hand_az_g",
        "hand_gx_dps",
        "hand_gy_dps",
        "hand_gz_dps",
        "arm_ax_g",
        "arm_ay_g",
        "arm_az_g",
        "arm_gx_dps",
        "arm_gy_dps",
        "arm_gz_dps",
    ]
    timestamp = 1000000.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for index in range(sample_count):
            if index:
                timestamp += interval * (5.0 if gap_index == index else 1.0)
            hand = [0.2 + index * 0.001, 0.1, 1.0, 1.0 + index * 0.01, 2.0, 3.0]
            forearm = [0.1 + index * 0.001, -0.1, 1.0, -1.0 - index * 0.01, -2.0, -3.0]
            if zero_forearm_start is not None and index >= zero_forearm_start:
                forearm = [0.0] * 6
            pressure = 4095 if saturated_pressure_start is not None and index >= saturated_pressure_start else 1200 + index
            writer.writerow([timestamp, pressure, pressure, 0, *hand, *forearm])
    return path.read_bytes()


def write_processed_angles(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["device_ms", "flex_deg", "deviation_deg"])
        deviations = [0.0, 170.0, -170.0, -160.0, -150.0, -140.0, -130.0, -120.0, -110.0, -100.0, -90.0, -80.0]
        for index, deviation in enumerate(deviations):
            writer.writerow([index * 200.0, float(index), deviation])


class HardwareCaptureImportTests(unittest.TestCase):
    def test_wide_capture_converts_units_nodes_and_fsr_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "datasets" / "imu_pressure_test.csv"
            destination = root / "outputs" / "capture.csv"
            source.parent.mkdir()
            original = write_wide_capture(source)
            original_hash = hashlib.sha256(original).hexdigest()

            report = audit_and_convert_capture(source, destination)

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report["source_sha256"], original_hash)
            self.assertTrue(report["source_unchanged"])
            self.assertFalse(report["analysis_ready"])
            self.assertEqual(report["output_row_count"], 24)
            with destination.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([rows[0]["sensor_id"], rows[1]["sensor_id"]], ["forearm", "hand"])
            self.assertAlmostEqual(float(rows[0]["ax"]), 0.1 * G_TO_MPS2)
            self.assertAlmostEqual(float(rows[0]["gx"]), -3.141592653589793 / 180.0)
            self.assertEqual(rows[0]["fsr_raw_adc"], "1200")
            self.assertEqual(rows[1]["fsr_raw_adc"], "")
            raw, _, rate = parse_raw_dual_imu(destination.read_bytes())
            self.assertEqual(raw["forearm_accel"].shape, (12, 3))
            self.assertEqual(len(raw["fsr_raw_adc"]), 12)
            self.assertAlmostEqual(rate, 400.0)

    def test_audit_flags_gaps_zero_forearm_and_majority_pressure_saturation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.csv"
            destination = root / "canonical.csv"
            write_wide_capture(
                source,
                sample_count=20,
                gap_index=5,
                zero_forearm_start=8,
                saturated_pressure_start=6,
            )

            report = audit_and_convert_capture(source, destination)

            self.assertEqual(report["timing"]["gap_count"], 1)
            self.assertGreaterEqual(
                report["quality"]["forearm"]["reason_counts"]["zero_accelerometer_vector"],
                12,
            )
            self.assertIn("timestamp_gaps_detected", report["warnings"])
            self.assertIn("forearm_invalid_samples_detected", report["warnings"])
            self.assertIn("pressure_majority_saturated", report["warnings"])

    def test_low_rate_capture_is_preserved_but_marked_not_analysis_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "capture.csv"
            destination = root / "canonical.csv"
            write_wide_capture(source, timestamp_field="device_ms", interval=200.0)

            report = audit_and_convert_capture(source, destination)

            self.assertAlmostEqual(report["timing"]["sample_rate_hz_median"], 5.0)
            self.assertIn("sample_rate_below_minimum_analysis_rate", report["warnings"])
            self.assertFalse(report["analysis_ready"])
            self.assertIn("missing_separate_calibration_recording", report["analysis_blockers"])

    def test_directory_import_classifies_unlabeled_data_and_audits_legacy_angles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "datasets"
            output_dir = root / "outputs"
            source_dir.mkdir()
            write_wide_capture(source_dir / "imu_pressure_20260828_000000.csv")
            write_processed_angles(source_dir / "wrist_20260828_000000.csv")

            report = import_capture_directory(source_dir, output_dir)

            self.assertEqual(report["dataset_classification"], "unlabeled_wired_hardware_pilot")
            self.assertEqual(report["summary"]["converted_raw_capture_count"], 1)
            self.assertEqual(report["summary"]["processed_angle_file_count"], 1)
            self.assertFalse(report["analysis_eligibility"]["abc_effect_comparison"])
            self.assertFalse(report["processed_angle_captures"][0]["promoted_to_joint_state"])
            self.assertEqual(
                report["processed_angle_captures"][0]["deviation_deg"]["wrap_jump_count_over_180deg"],
                1,
            )
            self.assertTrue((output_dir / "audit_report.json").is_file())
            self.assertTrue(
                (output_dir / "standardized/imu_pressure_20260828_000000_canonical.csv").is_file()
            )
            with self.assertRaises(ValueError):
                import_capture_directory(source_dir, source_dir / "generated")


if __name__ == "__main__":
    unittest.main()
