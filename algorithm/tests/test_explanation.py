import unittest

from shewrist.explanation import (
    TemplateExplanationProvider,
    build_explanation_request,
    explain_analysis,
    provider_from_config,
    validate_explanation_response,
)


class ExplanationTests(unittest.TestCase):
    def _analysis(self):
        return {
            "session_id": "S1",
            "evidence_type": "replay",
            "control_policy": {"llm_control_authority": "none"},
            "deterministic_control": {
                "metrics": {"valid_sample_pct": 98.0, "P_high_pct": 12.0, "alert_count": 1},
                "alerts": [{"timestamp_s": 2.0}],
            },
            "ml_shadow": {
                "operating_mode": "shadow",
                "accepted_window_count": 3,
                "rejected_window_count": 2,
                "tokens": [{"event_type": "extension", "duration_ms": 1000, "confidence": 0.8, "mean_quality": 0.9, "evidence_type": "replay"}],
            },
        }

    def test_template_provider_never_calls_an_api(self):
        result = explain_analysis(self._analysis(), {"provider": "template", "model": "replace-me", "language": "zh-CN"})
        self.assertEqual(result["provider"], "local_template")
        self.assertEqual(result["model"], "replace-me")
        self.assertFalse(result["api_called"])
        self.assertEqual(result["control_authority"], "none")
        self.assertEqual(result["response"]["safety_effect"], "none")

    def test_request_contains_structured_facts_not_raw_samples(self):
        request = build_explanation_request(self._analysis(), "model-placeholder")
        payload = request.to_dict()
        self.assertEqual(payload["model"], "model-placeholder")
        self.assertNotIn("raw_sensor", payload["facts"])
        self.assertEqual(payload["facts"]["alert_count"], 1)

    def test_external_provider_is_disabled_by_default(self):
        with self.assertRaises(ValueError):
            provider_from_config({"provider": "openai_compatible", "enabled": False, "model": "placeholder"})

    def test_provider_cannot_claim_safety_authority(self):
        with self.assertRaises(ValueError):
            validate_explanation_response({"summary": "x", "safety_effect": "alarm"})

    def test_template_protocol_is_concrete(self):
        provider = TemplateExplanationProvider("model-placeholder")
        response = provider.generate(build_explanation_request(self._analysis(), provider.model))
        self.assertTrue(response.summary)
        self.assertTrue(response.limitations)


if __name__ == "__main__":
    unittest.main()
