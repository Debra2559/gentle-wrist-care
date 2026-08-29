import unittest

import numpy as np

from shewrist.hmm import TemporalHMM
from shewrist.ml import NumpyTemporalCNN, ShadowActivityPipeline
from shewrist.ml_data import FEATURE_NAMES
from shewrist.replay import replay_mapping, verify_chunked_replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.algorithm_config = {
            "angle_degrees": {
                "flexion_extension": {"yellow_abs": 15.0, "red_abs": 30.0},
                "radial_ulnar": {"yellow_abs": 10.0, "red_abs": 20.0},
            },
            "compound_posture": {"extension_min": 15.0, "ulnar_min": 10.0},
            "pressure_kpa": {"yellow": 3.0, "red": 4.4},
            "duration_seconds": {"continuous_alert": 1.0, "rolling_window": 30.0, "rolling_high_exposure": 2.0, "cooldown": 5.0},
            "cycles_per_minute": {"cycle_amplitude_deg": 8.0},
        }
        self.ml_config = {"window": {"sample_rate_hz": 10.0, "window_seconds": 1.0, "step_seconds": 0.5, "label_strategy": "center_sample", "min_window_quality": 0.5}}
        cnn = NumpyTemporalCNN(len(FEATURE_NAMES), 5, 4, 3, seed=9)
        training = np.zeros((5, 10, len(FEATURE_NAMES)), dtype=np.float32)
        training[:, :, -1] = 1.0
        cnn.standardizer.fit(training)
        hmm = TemporalHMM.fit([np.zeros(20, dtype=int)], 5)
        self.pipeline = ShadowActivityPipeline(cnn, hmm, ("background", "extension", "flexion", "radial_deviation", "ulnar_deviation"), FEATURE_NAMES, 1.0, 0.5)
        self.joint = {
            "timestamp_ms": np.arange(41) * 100.0,
            "theta_FE": np.r_[np.zeros(20), np.full(21, 35.0)],
            "theta_RUD": np.zeros(41),
            "quality": np.ones(41),
        }

    def test_chunked_ingest_reconstructs_arrays(self):
        replayed, audit = replay_mapping(self.joint, 7)
        self.assertTrue(audit["input_reconstruction_equal"])
        self.assertEqual(audit["chunk_count"], 6)
        np.testing.assert_array_equal(replayed["theta_FE"], self.joint["theta_FE"])

    def test_chunked_and_batch_analysis_are_equal(self):
        result, audit = verify_chunked_replay(
            self.joint,
            self.algorithm_config,
            self.ml_config,
            self.pipeline,
            "replay-test",
            "simulation",
            7,
        )
        self.assertTrue(audit["deterministic_state_equal"])
        self.assertTrue(audit["final_analysis_equal"])
        self.assertEqual(result["control_policy"]["ml_control_authority"], "none")

    def test_replay_rejects_cross_chunk_timestamp_regression(self):
        joint = {key: value.copy() for key, value in self.joint.items()}
        joint["timestamp_ms"][10] = joint["timestamp_ms"][9]
        with self.assertRaises(ValueError):
            replay_mapping(joint, 7)


if __name__ == "__main__":
    unittest.main()
