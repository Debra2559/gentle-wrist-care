import unittest

import numpy as np

from shewrist.hmm import TemporalHMM


class HMMTests(unittest.TestCase):
    def test_probabilities_and_viterbi_suppress_isolated_flip(self):
        hmm = TemporalHMM.fit(
            [np.array([0, 0, 0, 1, 1, 1, 0, 0])],
            n_classes=2,
            self_transition_prior=30.0,
        )
        self.assertTrue(np.isclose(hmm.start_probability.sum(), 1.0))
        np.testing.assert_allclose(hmm.transition_probability.sum(axis=1), 1.0)
        emissions = np.array(
            [
                [0.95, 0.05],
                [0.95, 0.05],
                [0.45, 0.55],
                [0.95, 0.05],
                [0.95, 0.05],
            ]
        )
        np.testing.assert_array_equal(hmm.decode(emissions), np.zeros(5, dtype=int))

    def test_invalid_shape_is_rejected(self):
        hmm = TemporalHMM(np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]))
        with self.assertRaises(ValueError):
            hmm.decode(np.ones((4, 3)))


if __name__ == "__main__":
    unittest.main()
