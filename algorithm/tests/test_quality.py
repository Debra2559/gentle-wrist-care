import unittest

import numpy as np

from shewrist.quality import detect_stationary, nearest_sync_error_ms, sample_quality, timestamp_quality


class QualityTests(unittest.TestCase):
    def test_timestamp_gap_and_missing_sample_estimate(self):
        t = np.array([0.00, 0.01, 0.02, 0.05, 0.06])
        result = timestamp_quality(t)
        self.assertAlmostEqual(result["nominal_rate_hz"], 100.0)
        self.assertEqual(result["gap_count"], 1)
        self.assertEqual(result["estimated_missing_samples"], 2)

    def test_sample_quality_reasons(self):
        t = np.array([0.00, 0.01, 0.02, 0.05, 0.06])
        accel = np.tile([0.0, 0.0, 9.80665], (len(t), 1))
        accel[1] = [0.0, 0.0, 20.0]
        gyro = np.zeros((len(t), 3))
        mag = np.tile([30.0, 0.0, 0.0], (len(t), 1))
        mag[2] = [100.0, 0.0, 0.0]
        quality, reasons = sample_quality(t, accel, gyro, mag)
        self.assertIn("dynamic_acceleration", reasons[1])
        self.assertIn("magnetic_disturbance", reasons[2])
        self.assertIn("timestamp_gap", reasons[3])
        self.assertEqual(quality[3], 0.0)
        self.assertLess(quality[1], 1.0)
        self.assertLess(quality[2], 1.0)

    def test_stationarity_and_sync_error(self):
        accel = np.array([[0.0, 0.0, 9.80665], [0.0, 0.0, 11.0]])
        gyro = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.2]])
        stationary = detect_stationary(accel, gyro)
        np.testing.assert_array_equal(stationary, [True, False])
        sync = nearest_sync_error_ms(np.array([0.0, 0.01, 0.02]), np.array([0.001, 0.011, 0.021]))
        self.assertAlmostEqual(sync["max_sync_error_ms"], 1.0)

    def test_timestamp_quality_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            timestamp_quality(np.array([0.0, 0.01, 0.01]))


if __name__ == "__main__":
    unittest.main()
