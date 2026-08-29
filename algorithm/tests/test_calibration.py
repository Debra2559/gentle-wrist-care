import unittest

import numpy as np

from shewrist.calibration import (
    estimate_functional_axes,
    neutral_zero,
    project_angles,
    relative_quaternion,
)
from shewrist.quaternion import from_rotation_vector


class CalibrationTests(unittest.TestCase):
    def test_relative_quaternion_and_neutral_zero(self):
        forearm = np.repeat(from_rotation_vector(np.array([[0.0, 0.0, 0.2]])), 5, axis=0)
        wrist_offset = from_rotation_vector(np.array([0.3, 0.0, 0.0]))
        hand = np.array([
            from_rotation_vector(np.array([0.0, 0.0, 0.2 + 0.0]))
            for _ in range(5)
        ])
        from shewrist.quaternion import multiply

        hand = multiply(forearm, wrist_offset)
        relative = relative_quaternion(forearm, hand)
        np.testing.assert_allclose(relative, np.repeat(wrist_offset[None, :], 5, axis=0), atol=1e-12)
        zeroed, neutral = neutral_zero(relative, np.ones(5, dtype=bool))
        np.testing.assert_allclose(zeroed, np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)), atol=1e-12)
        np.testing.assert_allclose(neutral, wrist_offset, atol=1e-12)

    def test_functional_axes_follow_documented_signs(self):
        t = np.arange(0.0, 5.0, 0.01)
        rotation_vectors = np.zeros((len(t), 3))
        intervals = [
            ("Flexion", 0.8, 1.4, np.array([-0.30, 0.0, 0.0])),
            ("Extension", 1.6, 2.2, np.array([0.35, 0.0, 0.0])),
            ("Radial Deviation", 2.5, 3.1, np.array([0.0, -0.20, 0.0])),
            ("Ulnar Deviation", 3.3, 3.9, np.array([0.0, 0.25, 0.0])),
        ]
        annotations = []
        for label, start, end, vector in intervals:
            rotation_vectors[(t >= start) & (t <= end)] = vector
            annotations.append({"Type": label, "Init": start, "End": end})
        quaternions = from_rotation_vector(rotation_vectors)
        axes = estimate_functional_axes(quaternions, t, annotations)
        self.assertGreater(float(np.dot(axes.flexion_extension, [1.0, 0.0, 0.0])), 0.999)
        self.assertGreater(float(np.dot(axes.radial_ulnar, [0.0, 1.0, 0.0])), 0.999)
        self.assertGreater(float(np.dot(axes.pronation_supination, [0.0, 0.0, 1.0])), 0.999)
        fe, rud, _ = project_angles(quaternions, axes)
        self.assertLess(float(np.median(fe[(t >= 0.8) & (t <= 1.4)])), 0.0)
        self.assertGreater(float(np.median(fe[(t >= 1.6) & (t <= 2.2)])), 0.0)
        self.assertLess(float(np.median(rud[(t >= 2.5) & (t <= 3.1)])), 0.0)
        self.assertGreater(float(np.median(rud[(t >= 3.3) & (t <= 3.9)])), 0.0)

    def test_ulnar_only_pose_can_orient_deviation_axis(self):
        t = np.arange(0.0, 4.0, 0.01)
        rotation_vectors = np.zeros((len(t), 3))
        intervals = [
            ("Flexion", 0.5, 1.0, np.array([-0.30, 0.0, 0.0])),
            ("Extension", 1.2, 1.7, np.array([0.35, 0.0, 0.0])),
            ("Ulnar Deviation", 2.0, 2.5, np.array([0.0, 0.25, 0.0])),
        ]
        annotations = []
        for label, start, end, vector in intervals:
            rotation_vectors[(t >= start) & (t <= end)] = vector
            annotations.append({"Type": label, "Init": start, "End": end})
        axes = estimate_functional_axes(from_rotation_vector(rotation_vectors), t, annotations)
        self.assertGreater(float(np.dot(axes.flexion_extension, [1.0, 0.0, 0.0])), 0.999)
        self.assertGreater(float(np.dot(axes.radial_ulnar, [0.0, 1.0, 0.0])), 0.999)

    def test_neutral_zero_requires_three_samples(self):
        q = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        with self.assertRaises(ValueError):
            neutral_zero(q, np.array([True, True, False]))


if __name__ == "__main__":
    unittest.main()