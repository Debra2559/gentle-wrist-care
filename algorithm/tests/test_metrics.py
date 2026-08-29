import json
import unittest
from pathlib import Path

import numpy as np

from shewrist.metrics import count_complete_cycles, exposure_metrics, intervention_efficiency


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_exposure_dose_and_longest_run(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        fe = np.array([0.0, 20.0, 20.0, 0.0])
        rud = np.array([0.0, 0.0, 15.0, 0.0])
        metrics = exposure_metrics(t, fe, rud, self.config)
        self.assertAlmostEqual(metrics["task_duration_s"], 4.0)
        self.assertAlmostEqual(metrics["P_high_pct"], 50.0)
        self.assertAlmostEqual(metrics["D_FE_deg_s"], 10.0)
        self.assertAlmostEqual(metrics["D_RUD_deg_s"], 5.0)
        self.assertAlmostEqual(metrics["D_total_deg_s"], 15.0)
        self.assertAlmostEqual(metrics["L_max_s"], 2.0)

    def test_pressure_and_external_assist_torque(self):
        t = np.array([0.0, 1.0, 2.0])
        zero = np.zeros(3)
        metrics = exposure_metrics(
            t,
            zero,
            zero,
            self.config,
            pressure_kpa=np.array([4.4, 4.5, 3.0]),
            cable_tension_n=np.array([2.0, 4.0, 6.0]),
            lever_arm_m=0.02,
        )
        self.assertAlmostEqual(metrics["max_pressure_kPa"], 4.5)
        self.assertAlmostEqual(metrics["pressure_over_screening_s"], 1.0)
        self.assertAlmostEqual(metrics["mean_external_assist_torque_Nm"], 0.08)
        self.assertAlmostEqual(metrics["max_external_assist_torque_Nm"], 0.12)

    def test_cycles_and_intervention_efficiency(self):
        cycles = count_complete_cycles(np.array([-10.0, 10.0, -10.0, 10.0, -10.0]))
        self.assertEqual(cycles, 2.0)
        shifted_cycles = count_complete_cycles(np.array([10.0, 30.0, 10.0, 30.0, 10.0]))
        self.assertEqual(shifted_cycles, 2.0)
        self.assertAlmostEqual(intervention_efficiency(100.0, 60.0), 40.0)
        self.assertIsNone(intervention_efficiency(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
