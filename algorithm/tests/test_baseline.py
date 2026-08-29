import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shewrist.baseline import (
    build_personal_report,
    estimate_exposure_tolerance,
    goal_line,
    init_personal_baseline,
    load_personal_baseline,
    relative_exposure,
    save_personal_baseline,
    session_exposure_summary,
    symptom_exposure_association,
    update_personal_baseline,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


def _make_session(peak_fe: float, peak_rud: float, seconds: float = 120.0, rate: float = 50.0):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    phase = np.linspace(0.0, 8.0 * np.pi, n)
    fe = peak_fe * np.abs(np.sin(phase))
    rud = peak_rud * np.abs(np.sin(phase + 0.5))
    return t, fe, rud


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_session_summary_reports_tracked_metrics(self):
        t, fe, rud = _make_session(25.0, 15.0)
        summary = session_exposure_summary(t, fe, rud, self.config)
        self.assertGreater(summary["valid_minutes"], 1.9)
        for name in self.config["personal_baseline"]["tracked_metrics"]:
            self.assertIn(name, summary["metrics"])
        self.assertIsNotNone(summary["metrics"]["dose_rate_deg_s_per_min"])
        self.assertGreater(summary["metrics"]["abs_fe_deg_p90"], 0.0)

    def test_enrollment_rejects_too_short_session(self):
        t, fe, rud = _make_session(20.0, 12.0, seconds=30.0)  # 0.5 min < 1 min min
        summary = session_exposure_summary(t, fe, rud, self.config)
        baseline = init_personal_baseline("p1", summary, self.config)
        self.assertEqual(baseline.status, "rejected")
        self.assertIn("insufficient_valid_minutes", baseline.reasons)

    def test_enrollment_and_ewma_update(self):
        t, fe, rud = _make_session(20.0, 12.0)
        base_summary = session_exposure_summary(t, fe, rud, self.config)
        baseline = init_personal_baseline("p1", base_summary, self.config)
        self.assertEqual(baseline.status, "provisional")
        self.assertEqual(baseline.session_count, 1)

        t2, fe2, rud2 = _make_session(40.0, 24.0)
        high_summary = session_exposure_summary(t2, fe2, rud2, self.config)
        before = baseline.metrics["abs_fe_deg_p90"]
        updated = update_personal_baseline(baseline, high_summary, self.config)
        after = updated.metrics["abs_fe_deg_p90"]
        self.assertGreater(after, before)  # EWMA moved toward the higher session
        self.assertEqual(updated.session_count, 2)

    def test_relative_and_goal(self):
        t, fe, rud = _make_session(20.0, 12.0)
        baseline = init_personal_baseline(
            "p1", session_exposure_summary(t, fe, rud, self.config), self.config
        )
        t2, fe2, rud2 = _make_session(30.0, 18.0)
        today = session_exposure_summary(t2, fe2, rud2, self.config)
        rel = relative_exposure(today, baseline)
        self.assertGreater(rel["abs_fe_deg_p90"]["pct_vs_baseline"], 0.0)
        goal = goal_line(baseline, 20.0)
        self.assertAlmostEqual(
            goal["abs_fe_deg_p90"], baseline.metrics["abs_fe_deg_p90"] * 0.8, places=6
        )

    def test_symptom_association_gating_and_positive_case(self):
        few = symptom_exposure_association([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], self.config)
        self.assertEqual(few["status"], "not_evaluable")
        self.assertIn("insufficient_paired_days", few["reasons"])

        rng = np.random.default_rng(0)
        exposure = np.linspace(10.0, 30.0, 20)
        pain = 0.2 * exposure + rng.normal(0.0, 0.3, size=20)
        result = symptom_exposure_association(exposure, pain, self.config, lag_days=0)
        self.assertEqual(result["status"], "evaluable")
        self.assertGreater(result["pearson_r"], 0.5)
        self.assertEqual(len(result["pearson_r_ci95"]), 2)

    def test_tolerance_gating_and_estimate(self):
        not_ready = estimate_exposure_tolerance([1.0, 2.0], [1.0, 2.0], self.config)
        self.assertEqual(not_ready["status"], "not_evaluable")

        rng = np.random.default_rng(1)
        exposure = np.concatenate([np.full(10, 12.0), np.full(10, 28.0)]) + rng.normal(0, 0.5, 20)
        pain = np.concatenate([np.full(10, 1.0), np.full(10, 5.0)])
        tol = estimate_exposure_tolerance(exposure, pain, self.config, lag_days=0)
        self.assertEqual(tol["status"], "evaluable")
        self.assertLess(tol["tolerance_exposure"], tol["elevated_median_exposure"])

    def test_enrollment_rejects_session_over_max_minutes(self):
        # max_session_minutes default is 5.0; make a ~6 min session.
        t, fe, rud = _make_session(20.0, 12.0, seconds=360.0)
        summary = session_exposure_summary(t, fe, rud, self.config)
        baseline = init_personal_baseline("p1", summary, self.config)
        self.assertEqual(baseline.status, "rejected")
        self.assertIn("enrollment_session_exceeds_max_minutes", baseline.reasons)

    def test_persistence_round_trip(self):
        t, fe, rud = _make_session(20.0, 12.0)
        baseline = init_personal_baseline(
            "p1", session_exposure_summary(t, fe, rud, self.config), self.config
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "p1.json"
            save_personal_baseline(path, baseline)
            loaded = load_personal_baseline(path)
        self.assertEqual(loaded.participant_id, baseline.participant_id)
        self.assertEqual(loaded.status, baseline.status)
        self.assertEqual(loaded.session_count, baseline.session_count)
        self.assertAlmostEqual(
            loaded.metrics["abs_fe_deg_p90"], baseline.metrics["abs_fe_deg_p90"]
        )

    def test_build_report_has_no_control_authority(self):
        t, fe, rud = _make_session(20.0, 12.0)
        baseline = init_personal_baseline(
            "p1", session_exposure_summary(t, fe, rud, self.config), self.config
        )
        t2, fe2, rud2 = _make_session(35.0, 21.0)
        today = session_exposure_summary(t2, fe2, rud2, self.config)
        report = build_personal_report(baseline, today, self.config)
        self.assertEqual(report["control_effect"], "none")
        self.assertFalse(report["evidence_limits"]["ml_used"])
        self.assertEqual(report["evidence_limits"]["control_authority"], "none")
        self.assertIn("disease risk", report["evidence_limits"]["not_claimed"])


if __name__ == "__main__":
    unittest.main()
