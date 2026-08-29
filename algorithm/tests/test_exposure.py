import json
import unittest
from pathlib import Path

from shewrist.exposure import ExposureEngine, classify_zone


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


class ExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_angle_and_pressure_boundaries(self):
        self.assertEqual(classify_zone(14.999, 0.0, self.config)["zone"], "green")
        self.assertEqual(classify_zone(15.0, 0.0, self.config)["zone"], "yellow")
        self.assertEqual(classify_zone(30.0, 0.0, self.config)["zone"], "yellow")
        self.assertEqual(classify_zone(30.001, 0.0, self.config)["zone"], "red")
        self.assertEqual(classify_zone(0.0, 10.0, self.config)["zone"], "yellow")
        self.assertEqual(classify_zone(0.0, 20.0, self.config)["zone"], "yellow")
        self.assertEqual(classify_zone(0.0, 20.001, self.config)["zone"], "red")
        compound = classify_zone(15.0, 10.0, self.config)
        self.assertTrue(compound["compound"])
        self.assertEqual(compound["zone"], "red")
        self.assertEqual(classify_zone(0.0, 0.0, self.config, pressure_kpa=3.0)["zone"], "yellow")
        self.assertEqual(classify_zone(0.0, 0.0, self.config, pressure_kpa=4.4)["zone"], "yellow")
        self.assertEqual(classify_zone(0.0, 0.0, self.config, pressure_kpa=4.401)["zone"], "red")

    def test_continuous_alert_and_cooldown(self):
        engine = ExposureEngine(self.config)
        alerts = []
        for second in range(311):
            state = engine.update(float(second), 31.0, 0.0)
            if state.alert:
                alerts.append(second)
        self.assertEqual(alerts, [10, 310])

    def test_rolling_mechanical_recommendation_is_edge_event(self):
        engine = ExposureEngine(self.config)
        recommendation_times = []
        for second in range(62):
            state = engine.update(float(second), 31.0, 0.0)
            if state.recommend_mechanical:
                recommendation_times.append(second)
        self.assertEqual(recommendation_times, [60])

    def test_pressure_or_discomfort_stops_immediately(self):
        pressure = ExposureEngine(self.config).update(0.0, 0.0, 0.0, pressure_kpa=4.5)
        self.assertTrue(pressure.safety_stop)
        self.assertTrue(pressure.alert)
        self.assertEqual(pressure.alert_reason, "release_and_stop_calibrated_pressure_or_safety_symptom")
        discomfort = ExposureEngine(self.config).update(0.0, 0.0, 0.0, discomfort=True)
        self.assertTrue(discomfort.safety_stop)
        self.assertTrue(discomfort.alert)

    def test_pressure_alert_is_not_blocked_by_angle_cooldown(self):
        engine = ExposureEngine(self.config)
        engine.update(0.0, 31.0, 0.0)
        angle_alert = engine.update(10.0, 31.0, 0.0)
        pressure_alert = engine.update(10.1, 31.0, 0.0, pressure_kpa=4.5)
        self.assertTrue(angle_alert.alert)
        self.assertTrue(pressure_alert.alert)
        self.assertTrue(pressure_alert.safety_stop)

    def test_invalid_quality_never_accumulates_or_alerts(self):
        engine = ExposureEngine(self.config)
        first = engine.update(0.0, 45.0, 25.0, quality_valid=False)
        second = engine.update(20.0, 45.0, 25.0, quality_valid=False)
        self.assertEqual(first.zone, "invalid")
        self.assertEqual(second.high_duration_s, 0.0)
        self.assertFalse(second.alert)

    def test_pressure_stop_survives_invalid_angle_quality(self):
        state = ExposureEngine(self.config).update(
            0.0,
            float("nan"),
            float("nan"),
            pressure_kpa=4.5,
            quality_valid=False,
        )
        self.assertEqual(state.angle_zone, "invalid")
        self.assertEqual(state.pressure_zone, "red")
        self.assertTrue(state.safety_stop)
        self.assertTrue(state.alert)


if __name__ == "__main__":
    unittest.main()