import unittest

import numpy as np

from shewrist.hmm import TemporalHMM
from shewrist.ml import NumpyTemporalCNN, ShadowActivityPipeline
from shewrist.ml_data import FEATURE_NAMES
from shewrist.pipeline import analyze_with_shadow


class CombinedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.algorithm_config = {
            "angle_degrees": {
                "flexion_extension": {"yellow_abs": 15.0, "red_abs": 30.0},
                "radial_ulnar": {"yellow_abs": 10.0, "red_abs": 20.0},
            },
            "compound_posture": {"extension_min": 15.0, "ulnar_min": 10.0},
            "pressure_kpa": {"yellow": 3.0, "red": 4.4},
            "duration_seconds": {
                "continuous_alert": 10.0,
                "rolling_window": 300.0,
                "rolling_high_exposure": 60.0,
                "cooldown": 300.0,
            },
            "cycles_per_minute": {"cycle_amplitude_deg": 8.0},
        }
        self.ml_config = {
            "window": {
                "sample_rate_hz": 10.0,
                "window_seconds": 1.0,
                "step_seconds": 0.5,
                "label_strategy": "center_sample",
                "min_window_quality": 0.5,
            }
        }

    def _pipeline(self):
        cnn = NumpyTemporalCNN(len(FEATURE_NAMES), 5, 4, 3, seed=7)
        training = np.zeros((5, 10, len(FEATURE_NAMES)), dtype=np.float32)
        training[:, :, -1] = 1.0
        cnn.standardizer.fit(training)
        hmm = TemporalHMM.fit([np.zeros(10, dtype=int)], 5)
        return ShadowActivityPipeline(cnn, hmm, ("background", "extension", "flexion", "radial_deviation", "ulnar_deviation"), FEATURE_NAMES, 1.0, 0.5)

    def test_pressure_stop_bypasses_rejected_shadow_predictions(self):
        timestamp_ms = np.arange(31) * 100.0
        joint = {
            "timestamp_ms": timestamp_ms,
            "theta_FE": np.zeros(31),
            "theta_RUD": np.zeros(31),
            "quality": np.zeros(31),
        }
        pressure = np.full(31, 4.5)
        result = analyze_with_shadow(
            joint,
            self.algorithm_config,
            self.ml_config,
            self._pipeline(),
            "test",
            "bench",
            pressure_kpa=pressure,
        )
        self.assertGreater(len(result["deterministic_control"]["alerts"]), 0)
        self.assertEqual(
            result["control_policy"]["pressure_stop_authority"],
            "deterministic_calibrated_pressure_or_safety_symptom",
        )
        self.assertEqual(result["control_policy"]["ml_control_authority"], "none")
        self.assertEqual(result["ml_shadow"]["accepted_window_count"], 0)
        self.assertEqual(result["ml_shadow"]["mechanical_control_effect"], "none")


if __name__ == "__main__":
    unittest.main()