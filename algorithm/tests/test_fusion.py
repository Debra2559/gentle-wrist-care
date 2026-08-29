import unittest

import numpy as np

from shewrist.fusion import estimate_orientation
from shewrist.quaternion import distance_degrees, from_rotation_vector


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0.0, 1.0, 101)
        self.accel = np.tile([0.0, 0.0, 9.80665], (len(self.t), 1))

    def test_constant_yaw_rate_for_both_filters(self):
        gyro = np.tile([0.0, 0.0, 1.0], (len(self.t), 1))
        expected = from_rotation_vector(np.array([0.0, 0.0, 1.0]))
        for algorithm in ("madgwick", "mahony"):
            with self.subTest(algorithm=algorithm):
                result = estimate_orientation(self.t, self.accel, gyro, algorithm=algorithm)
                self.assertEqual(result.algorithm, algorithm)
                self.assertLess(float(distance_degrees(result.quaternion[-1], expected)), 0.02)
                np.testing.assert_allclose(np.linalg.norm(result.quaternion, axis=1), 1.0, atol=1e-12)

    def test_supplied_gyro_bias_is_removed(self):
        bias = np.array([0.1, -0.05, 0.02])
        gyro = np.tile(bias, (len(self.t), 1))
        result = estimate_orientation(self.t, self.accel, gyro, gyro_bias=bias)
        np.testing.assert_allclose(result.quaternion, np.tile([1.0, 0.0, 0.0, 0.0], (len(self.t), 1)), atol=1e-12)
        np.testing.assert_allclose(result.gyro_bias, bias)

    def test_invalid_magnetic_field_degrades_quality_but_stays_finite(self):
        gyro = np.zeros((len(self.t), 3))
        mag = np.tile([200.0, 0.0, 0.0], (len(self.t), 1))
        result = estimate_orientation(self.t, self.accel, gyro, magnetometer_uT=mag)
        self.assertTrue(np.all(result.quality[1:] < 1.0))
        self.assertTrue(np.all(np.isfinite(result.quaternion)))

    def test_rejects_non_increasing_timestamps_and_unknown_algorithm(self):
        gyro = np.zeros((len(self.t), 3))
        bad_t = self.t.copy()
        bad_t[20] = bad_t[19]
        with self.assertRaises(ValueError):
            estimate_orientation(bad_t, self.accel, gyro)
        with self.assertRaises(ValueError):
            estimate_orientation(self.t, self.accel, gyro, algorithm="unknown")


if __name__ == "__main__":
    unittest.main()
