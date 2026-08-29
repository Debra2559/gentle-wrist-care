import unittest

import numpy as np

from shewrist.faults import FaultSpec, inject_faults
from shewrist.quality import sensor_fault_quality, timestamp_quality
from shewrist.session import synchronize_dual_imu


class FaultInjectionTests(unittest.TestCase):
    def _raw(self):
        t = np.arange(100, dtype=float) / 100.0
        accel = np.tile([0.0, 0.0, 9.80665], (100, 1))
        gyro = np.column_stack((0.01 * np.sin(t), np.zeros(100), np.zeros(100)))
        return {
            "forearm_timestamp_s": t.copy(),
            "forearm_accel": accel.copy(),
            "forearm_gyro": gyro.copy(),
            "hand_timestamp_s": t.copy(),
            "hand_accel": accel.copy(),
            "hand_gyro": gyro.copy(),
        }

    def test_dropout_creates_an_auditable_timestamp_gap(self):
        corrupted, _, audit = inject_faults(self._raw(), [FaultSpec("dropout", duration_fraction=0.05)])
        quality = timestamp_quality(corrupted["hand_timestamp_s"])
        self.assertGreater(quality["gap_count"], 0)
        self.assertEqual(audit[0]["expected_response"], "affected_samples_rejected_or_degraded")

    def test_out_of_order_is_explicitly_rejected(self):
        corrupted, _, _ = inject_faults(self._raw(), [FaultSpec("out_of_order")])
        with self.assertRaises(ValueError):
            timestamp_quality(corrupted["hand_timestamp_s"])

    def test_silence_and_saturation_are_rejected(self):
        corrupted, _, _ = inject_faults(
            self._raw(),
            [FaultSpec("silence", duration_fraction=0.20), FaultSpec("saturation", start_fraction=0.20, duration_fraction=0.05)],
        )
        quality, reasons = sensor_fault_quality(corrupted["hand_accel"], corrupted["hand_gyro"])
        flat_reasons = {reason for sample in reasons for reason in sample}
        self.assertIn("sensor_silence", flat_reasons)
        self.assertIn("sensor_saturation", flat_reasons)
        self.assertGreater(np.count_nonzero(quality == 0.0), 0)

    def test_node_offset_fails_sync_gate_and_quality(self):
        corrupted, _, _ = inject_faults(self._raw(), [FaultSpec("timestamp_offset", magnitude=50.0)])
        aligned, audit = synchronize_dual_imu(corrupted, 100.0, {"gap_factor": 1.5}, 20.0)
        self.assertFalse(audit["sync_gate_passed"])
        self.assertTrue(np.all(aligned["hand_quality"] == 0.0))
        self.assertTrue(np.all(aligned["forearm_quality"] == 0.0))

    def test_internal_clock_mismatch_fails_even_when_boundaries_match(self):
        raw = self._raw()
        keep = np.r_[np.arange(40), np.arange(80, 100)]
        raw["hand_timestamp_s"] = raw["hand_timestamp_s"][keep]
        raw["hand_accel"] = raw["hand_accel"][keep]
        raw["hand_gyro"] = raw["hand_gyro"][keep]
        aligned, audit = synchronize_dual_imu(raw, 100.0, {"gap_factor": 1.5}, 20.0)
        self.assertAlmostEqual(audit["boundary_offset_ms"], 0.0)
        self.assertGreater(audit["nearest_sync"]["max_sync_error_ms"], 20.0)
        self.assertFalse(audit["sync_gate_passed"])
        self.assertEqual(audit["sync_gate_basis"], "boundary_offset_and_bidirectional_p95_and_max")
        self.assertTrue(np.all(aligned["hand_quality"] == 0.0))
        self.assertTrue(np.all(aligned["forearm_quality"] == 0.0))


if __name__ == "__main__":
    unittest.main()
