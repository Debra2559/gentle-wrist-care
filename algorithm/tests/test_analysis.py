import json
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from shewrist.analysis import analyze_condition


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_pressure_yellow_does_not_inflate_angle_exposure(self):
        t = np.array([0.0, 1.0, 2.0])
        angles = np.zeros(3)
        pressure = np.full(3, 3.5)
        metrics, alerts = analyze_condition(t, angles, angles, self.config, pressure_kpa=pressure)
        self.assertAlmostEqual(metrics["P_high_pct"], 0.0)
        self.assertAlmostEqual(metrics["D_total_deg_s"], 0.0)
        self.assertEqual(metrics["max_pressure_kPa"], 3.5)
        self.assertEqual(alerts, [])

    def test_continuous_red_pressure_is_one_safety_event(self):
        t = np.array([0.0, 0.5, 1.0, 1.5])
        angles = np.zeros(4)
        pressure = np.full(4, 4.5)
        metrics, alerts = analyze_condition(t, angles, angles, self.config, pressure_kpa=pressure)
        self.assertEqual(metrics["safety_stop_count"], 1)
        self.assertTrue(all(alert["safety_stop"] for alert in alerts))

    def test_user_continues_gates_manual_mechanical_recommendation(self):
        config = deepcopy(self.config)
        config["duration_seconds"].update(
            {"continuous_alert": 1.0, "rolling_window": 30.0, "rolling_high_exposure": 2.0, "cooldown": 5.0}
        )
        timestamp = np.arange(0.0, 4.1, 0.1)
        angles = np.full(len(timestamp), 35.0)
        stopped, stopped_alerts = analyze_condition(
            timestamp,
            angles,
            np.zeros(len(timestamp)),
            config,
            user_continues=np.zeros(len(timestamp), dtype=bool),
        )
        continued, continued_alerts = analyze_condition(
            timestamp,
            angles,
            np.zeros(len(timestamp)),
            config,
            user_continues=np.ones(len(timestamp), dtype=bool),
        )
        self.assertGreater(stopped["alert_count"], 0)
        self.assertEqual(stopped["mechanical_recommendation_count"], 0)
        self.assertEqual(sum(item["recommend_mechanical"] for item in stopped_alerts), 0)
        self.assertEqual(continued["mechanical_recommendation_count"], 1)
        self.assertEqual(sum(item["recommend_mechanical"] for item in continued_alerts), 1)

    def test_trial_reminder_switch_records_silent_and_actual_events(self):
        config = deepcopy(self.config)
        config["duration_seconds"].update({"continuous_alert": 1.0, "cooldown": 30.0})
        timestamp = np.arange(0.0, 3.1, 0.1)
        angles = np.full(len(timestamp), 35.0)
        silent_metrics, silent_alerts = analyze_condition(
            timestamp,
            angles,
            np.zeros(len(timestamp)),
            config,
            angle_alerts_enabled=False,
            mechanical_recommendations_enabled=False,
        )
        active_metrics, active_alerts = analyze_condition(
            timestamp,
            angles,
            np.zeros(len(timestamp)),
            config,
            angle_alerts_enabled=True,
            mechanical_recommendations_enabled=False,
        )
        self.assertEqual(silent_metrics["alert_count"], 0)
        self.assertEqual(silent_metrics["would_alert_count"], 1)
        self.assertEqual(silent_metrics["mechanical_recommendation_count"], 0)
        self.assertEqual(silent_alerts, [])
        self.assertEqual(active_metrics["alert_count"], 1)
        self.assertEqual(active_metrics["would_alert_count"], 1)
        self.assertEqual(len(active_alerts), 1)


if __name__ == "__main__":
    unittest.main()