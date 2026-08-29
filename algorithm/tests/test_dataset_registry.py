from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shewrist.dataset_registry import (
    DatasetCapabilityError,
    DatasetRegistry,
)
from shewrist.experts import (
    ExpertContract,
    ExpertPrediction,
    ReservedExpert,
    ValidatedFusionWeights,
    fuse_expert_probabilities,
)
from shewrist.ml_data import FEATURE_NAMES, WindowDataset, concatenate_datasets
from shewrist.ml_evaluation import leave_one_dataset_out_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_CONFIG = PROJECT_ROOT / "config/datasets.json"


def make_dataset(dataset_id: str, count: int = 3) -> WindowDataset:
    return WindowDataset(
        windows=np.zeros((count, 5, len(FEATURE_NAMES)), dtype=np.float32),
        labels=np.arange(count, dtype=int) % 5,
        subject_ids=np.asarray([f"s{index}" for index in range(count)], dtype=object),
        session_ids=np.asarray([f"session{index}" for index in range(count)], dtype=object),
        sequence_ids=np.asarray([f"seq{index}" for index in range(count)], dtype=object),
        start_s=np.arange(count, dtype=float),
        end_s=np.arange(count, dtype=float) + 0.5,
        mean_quality=np.ones(count, dtype=float),
        dataset_ids=np.full(count, dataset_id, dtype=object),
        missing_fraction=np.linspace(0.0, 0.2, count),
    )


class DatasetRegistryTests(unittest.TestCase):
    def test_registry_reports_reserved_sources_without_treating_them_as_trainable(self):
        registry = DatasetRegistry.from_config(REGISTRY_CONFIG, PROJECT_ROOT)
        ultra = registry.inspect("ultra_mocap")
        self.assertEqual(ultra.status, "not_installed")
        self.assertNotIn("activity_training", ultra.usable_capabilities)
        with self.assertRaises(DatasetCapabilityError):
            registry.build_activity_dataset("ultra_mocap", {}, {})

    def test_upper_body_inspection_requires_all_six_streams_and_annotations(self):
        registry = DatasetRegistry.from_config(REGISTRY_CONFIG, PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "annotations.csv").write_text("Subject,Set\n", encoding="utf-8")
            trial = root / "subject01/set2"
            for segment in ("forearm", "hand"):
                directory = trial / segment
                directory.mkdir(parents=True)
                for filename in ("Accelerometer.txt", "Gyroscope.txt", "Magnetometer.txt"):
                    (directory / filename).write_text("0,0,0,0\n1,0,0,0\n", encoding="utf-8")
            ready = registry.inspect("upper_body_movements", root)
            self.assertEqual(ready.status, "ready")
            (trial / "hand/Magnetometer.txt").unlink()
            incomplete = registry.inspect("upper_body_movements", root)
            self.assertEqual(incomplete.status, "incomplete")
            self.assertIn("missing_required_forearm_or_hand_streams", incomplete.blockers)

    def test_opto_adapter_computes_reference_metrics_from_declared_tasks(self):
        registry = DatasetRegistry.from_config(REGISTRY_CONFIG, PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aligned = root / "Python/4.OPTO_IMU_Joint_Angles_aligned.pkl"
            aligned.parent.mkdir(parents=True)
            opto = {5: {"Right Wrist": np.zeros((5, 3))}, 6: {"Right Wrist": np.zeros((5, 3))}}
            imu = {5: {"Right Wrist": np.ones((5, 3))}, 6: {"Right Wrist": np.ones((5, 3)) * 2.0}}
            with aligned.open("wb") as handle:
                pickle.dump((opto, imu), handle)
            result = registry.evaluate_angle_reference(
                "comparison_imu_optotrak",
                {"acceptance": {"dynamic_mae_deg_max": 8.0}},
                root_override=root,
            )
            self.assertEqual(result["dataset_id"], "comparison_imu_optotrak")
            self.assertAlmostEqual(result["results"]["wrist_flexion_extension"]["mae_deg"], 1.0)
            self.assertAlmostEqual(result["results"]["wrist_radial_ulnar_deviation"]["mae_deg"], 2.0)

    def test_window_dataset_preserves_source_and_missing_metadata(self):
        first = make_dataset("one", 2)
        second = make_dataset("two", 3)
        combined = concatenate_datasets((first, second))
        self.assertEqual(combined.dataset_counts(), {"one": 2, "two": 3})
        subset = combined.subset(np.array([1, 3]))
        self.assertEqual(subset.dataset_ids.tolist(), ["one", "two"])
        np.testing.assert_allclose(subset.missing_fraction, [0.2, 0.1])

    def test_leave_one_dataset_out_requires_distinct_sources(self):
        with self.assertRaises(ValueError):
            leave_one_dataset_out_splits(np.array(["one", "one"], dtype=object))
        splits = leave_one_dataset_out_splits(np.array(["one", "two", "three"], dtype=object))
        self.assertEqual(len(splits), 3)
        for split in splits:
            self.assertNotIn(split.test_dataset_id, split.train_dataset_ids)

    def test_reserved_expert_returns_unavailable_prediction(self):
        dataset = make_dataset("one")
        contract = ExpertContract(
            expert_id="future",
            task="future_task",
            dataset_ids=("future",),
            output_labels=("background", "extension"),
            required_features=FEATURE_NAMES,
        )
        prediction = ReservedExpert(contract, "dataset_not_ready").predict(dataset)
        self.assertFalse(np.any(prediction.available))
        self.assertTrue(np.all(prediction.rejection_reason == "dataset_not_ready"))
        self.assertTrue(np.all(prediction.probabilities == 0.0))

    def test_fusion_requires_target_validated_weights_and_ignores_missing_expert(self):
        with self.assertRaises(ValueError):
            ValidatedFusionWeights({"a": 1.0}, "", "macro_f1")
        available = ExpertPrediction(
            expert_id="a",
            labels=("background", "extension"),
            probabilities=np.asarray([[0.25, 0.75], [0.8, 0.2]]),
            confidence=np.asarray([0.75, 0.8]),
            quality=np.ones(2),
            compatibility=np.ones(2),
            available=np.ones(2, dtype=bool),
            rejection_reason=np.asarray(["accepted", "accepted"], dtype=object),
        )
        unavailable = ExpertPrediction(
            expert_id="b",
            labels=("background", "extension", "other"),
            probabilities=np.zeros((2, 3)),
            confidence=np.zeros(2),
            quality=np.ones(2),
            compatibility=np.zeros(2),
            available=np.zeros(2, dtype=bool),
            rejection_reason=np.asarray(["missing_modality", "missing_modality"], dtype=object),
        )
        weights = ValidatedFusionWeights(
            {"a": 0.7, "b": 0.3},
            "shewrist_target_validation_v1",
            "participant_disjoint_macro_f1",
        )
        fused = fuse_expert_probabilities((available, unavailable), weights)
        np.testing.assert_allclose(fused.probabilities, available.probabilities)
        np.testing.assert_array_equal(fused.accepted_labels, [1, 0])
        np.testing.assert_array_equal(fused.contributing_experts, [1, 1])
        self.assertEqual(fused.control_authority, "none")

    def test_registry_config_is_valid_json_and_fusion_is_disabled(self):
        payload = json.loads(REGISTRY_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(payload["fusion_policy"]["status"], "disabled_until_target_hardware_validation")
        self.assertIsNone(payload["fusion_policy"]["validated_weights"])
        registry = DatasetRegistry.from_config(REGISTRY_CONFIG, PROJECT_ROOT)
        report = registry.readiness_summary()
        upper = next(item for item in report["datasets"] if item["dataset_id"] == "upper_body_movements")
        self.assertEqual(upper["license"], "CC BY 4.0")
        self.assertEqual(upper["source_label_map"]["Extension"], "extension")


if __name__ == "__main__":
    unittest.main()