import unittest

import numpy as np

from shewrist.quaternion import (
    distance_degrees,
    from_rotation_vector,
    integrate_gyro,
    inverse,
    make_continuous,
    multiply,
    normalize,
    rotate_vector,
    to_rotation_vector,
)


class QuaternionTests(unittest.TestCase):
    def test_identity_inverse_and_zero_norm(self):
        q = from_rotation_vector(np.array([0.3, -0.2, 0.1]))
        identity = multiply(q, inverse(q))
        np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        with self.assertRaises(ValueError):
            normalize(np.zeros(4))

    def test_rotation_vector_round_trip(self):
        vectors = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.1, 0.3],
                [-0.4, 0.5, 0.1],
            ]
        )
        recovered = to_rotation_vector(from_rotation_vector(vectors))
        np.testing.assert_allclose(recovered, vectors, atol=1e-12)

    def test_rotate_vector(self):
        q = from_rotation_vector(np.array([0.0, 0.0, np.pi / 2.0]))
        rotated = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)

    def test_make_continuous_removes_sign_flips(self):
        q = from_rotation_vector(np.array([0.1, 0.2, 0.3]))
        sequence = np.stack((q, -q, q, -q))
        continuous = make_continuous(sequence)
        self.assertTrue(np.all(np.sum(continuous[:-1] * continuous[1:], axis=1) > 0.0))
        np.testing.assert_allclose(continuous, np.repeat(q[None, :], 4, axis=0), atol=1e-12)

    def test_constant_gyro_integration(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(100):
            q = integrate_gyro(q, np.array([0.0, 0.0, 1.0]), 0.01)
        expected = from_rotation_vector(np.array([0.0, 0.0, 1.0]))
        self.assertLess(float(distance_degrees(q, expected)), 0.01)


if __name__ == "__main__":
    unittest.main()
