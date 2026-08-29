import json
import unittest
from pathlib import Path

import numpy as np

from shewrist.validation import angle_error_metrics, evaluate_go_no_go, paired_condition_comparison


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_angle_error_metrics(self):
        reference = np.array([0.0, 1.0, 2.0, 3.0])
        estimate = reference + 1.0
        result = angle_error_metrics(reference, estimate, sample_rate_hz=100.0)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mae_deg"], 1.0)
        self.assertAlmostEqual(result["rmse_deg"], 1.0)
        self.assertAlmostEqual(result["bias_deg"], 1.0)
        self.assertAlmostEqual(result["rom_error_deg"], 0.0)
        self.assertAlmostEqual(result["correlation"], 1.0)

    def test_pairing_never_crosses_participants(self):
        records = [
            {"participant_id": "P1", "condition_id": "A", "dose": 100.0},
            {"participant_id": "P2", "condition_id": "C", "dose": 100.0},
            {"participant_id": "P1", "condition_id": "C", "dose": 50.0},
            {"participant_id": "P2", "condition_id": "A", "dose": 200.0},
            {"participant_id": "P3", "condition_id": "A", "dose": 999.0},
            {"participant_id": "P4", "condition_id": "C", "dose": 1.0},
        ]
        result = paired_condition_comparison(records, "dose", "A", "C")
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["participants"], ["P1", "P2"])
        self.assertAlmostEqual(result["baseline_mean"], 150.0)
        self.assertAlmostEqual(result["comparison_mean"], 75.0)
        self.assertAlmostEqual(result["paired_difference_mean"], -75.0)
        self.assertAlmostEqual(result["reduction_pct_mean"], 50.0)

    def test_go_no_go_uses_document_boundaries(self):
        ac = {"reduction_pct_mean": 20.0}
        bc = {"paired_difference_mean": -0.01}
        result = evaluate_go_no_go(
            ac,
            bc,
            4.4,
            4.99,
            5.0,
            self.config,
            pressure_discomfort=False,
            effective_alert_acceptance_pct=70.0,
        )
        self.assertEqual(result["decision"], "GO")
        failed = evaluate_go_no_go(
            ac,
            bc,
            4.401,
            5.0,
            4.99,
            self.config,
            pressure_discomfort=True,
            effective_alert_acceptance_pct=69.99,
        )
        self.assertEqual(failed["decision"], "NO-GO")
        self.assertFalse(failed["checks"]["pressure_screening_pass"])
        self.assertFalse(failed["checks"]["no_pressure_discomfort"])
        self.assertFalse(failed["checks"]["task_performance_pass"])
        self.assertFalse(failed["checks"]["comfort_pass"])
        self.assertFalse(failed["checks"]["effective_alert_acceptance_pass"])

    def test_go_no_go_marks_missing_human_outcomes_not_evaluable(self):
        result = evaluate_go_no_go(
            {"reduction_pct_mean": 30.0},
            {"paired_difference_mean": -1.0},
            4.0,
            2.0,
            6.0,
            self.config,
        )
        self.assertEqual(result["decision"], "NOT-EVALUABLE")
        self.assertIsNone(result["checks"]["no_pressure_discomfort"])
        self.assertIsNone(result["checks"]["effective_alert_acceptance_pass"])
        self.assertEqual(
            result["not_evaluable_checks"],
            ["effective_alert_acceptance_pass", "no_pressure_discomfort"],
        )

    def test_missing_calibrated_pressure_is_not_evaluable(self):
        result = evaluate_go_no_go(
            {"reduction_pct_mean": 30.0},
            {"paired_difference_mean": -1.0},
            None,
            2.0,
            6.0,
            self.config,
            pressure_discomfort=False,
            effective_alert_acceptance_pct=80.0,
        )
        self.assertEqual(result["decision"], "NOT-EVALUABLE")
        self.assertIsNone(result["checks"]["pressure_screening_pass"])
        self.assertEqual(result["not_evaluable_checks"], ["pressure_screening_pass"])


if __name__ == "__main__":
    unittest.main()