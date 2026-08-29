import tempfile
import unittest
from pathlib import Path

import numpy as np

from shewrist.hmm import TemporalHMM
from shewrist.ml import NumpyTemporalCNN, ShadowActivityPipeline, augment_windows
from shewrist.ml_data import FEATURE_NAMES, WindowDataset, extract_windows
from shewrist.ml_evaluation import loso_splits
from shewrist.tokens import build_inertial_tokens


class MachineLearningTests(unittest.TestCase):
    def _dataset(self, quality=1.0):
        rng = np.random.default_rng(2)
        windows = rng.normal(size=(10, 20, len(FEATURE_NAMES))).astype(np.float32)
        windows[:, :, -1] = quality
        return WindowDataset(
            windows=windows,
            labels=np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
            subject_ids=np.array(["s1"] * 10, dtype=object),
            session_ids=np.array(["session"] * 10, dtype=object),
            sequence_ids=np.array(["seq"] * 10, dtype=object),
            start_s=np.arange(10, dtype=float),
            end_s=np.arange(10, dtype=float) + 1.0,
            mean_quality=np.full(10, quality),
        )

    def test_loso_never_overlaps_participants(self):
        ids = np.array(["s1", "s2", "s3", "s4"], dtype=object)
        for split in loso_splits(ids):
            self.assertNotIn(split.test_subject, split.train_subjects)
            self.assertNotIn(split.validation_subject, split.train_subjects)
            self.assertNotEqual(split.test_subject, split.validation_subject)

    def test_center_label_keeps_continuous_sequence(self):
        t = np.arange(20) / 10.0
        features = np.zeros((20, len(FEATURE_NAMES)), dtype=np.float32)
        features[:, -1] = 1.0
        labels = np.zeros(20, dtype=int)
        labels[8:13] = 1
        dataset = extract_windows(
            t,
            features,
            labels,
            np.zeros(20, dtype=bool),
            "s1",
            "session",
            {"sample_rate_hz": 10.0, "window_seconds": 0.5, "step_seconds": 0.2, "label_strategy": "center_sample"},
        )
        self.assertEqual(len(set(dataset.sequence_ids.tolist())), 1)
        self.assertIn(1, dataset.labels)
        self.assertIn(0, dataset.labels)

    def test_augmentation_marks_missing_samples_low_quality(self):
        dataset = self._dataset()
        config = {
            "enabled": True,
            "time_mask_probability": 1.0,
            "time_mask_fraction_max": 0.2,
            "channel_mask_probability": 0.0,
        }
        augmented = augment_windows(dataset.windows, dataset.feature_names, config, np.random.default_rng(3))
        self.assertTrue(np.any(augmented[:, :, -1] == 0.0))

    def test_low_quality_is_rejected_and_model_round_trips(self):
        dataset = self._dataset(quality=0.1)
        cnn = NumpyTemporalCNN(len(FEATURE_NAMES), 5, 4, 3, seed=4)
        cnn.standardizer.fit(dataset.windows)
        hmm = TemporalHMM.fit([dataset.labels], 5)
        pipeline = ShadowActivityPipeline(cnn, hmm, dataset.class_names, dataset.feature_names, 0.0, 0.5)
        prediction = pipeline.predict(dataset)
        self.assertTrue(np.all(prediction.accepted_labels == -1))
        self.assertTrue(np.all(prediction.rejection_reason == "low_quality"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            pipeline.save(path)
            loaded = ShadowActivityPipeline.load(path)
            loaded_prediction = loaded.predict(dataset)
        np.testing.assert_array_equal(prediction.accepted_labels, loaded_prediction.accepted_labels)
        np.testing.assert_allclose(prediction.probabilities, loaded_prediction.probabilities)
        self.assertFalse(bool(loaded.metadata.get("allow_alarm_control", False)))
        self.assertFalse(bool(loaded.metadata.get("allow_mechanical_control", False)))
        self.assertAlmostEqual(loaded.max_missing_fraction, 0.1)

    def test_excessive_missing_fraction_is_rejected_and_round_trips(self):
        source = self._dataset(quality=1.0)
        dataset = WindowDataset(
            windows=source.windows,
            labels=source.labels,
            subject_ids=source.subject_ids,
            session_ids=source.session_ids,
            sequence_ids=source.sequence_ids,
            start_s=source.start_s,
            end_s=source.end_s,
            mean_quality=source.mean_quality,
            missing_fraction=np.full(len(source), 0.2),
        )
        cnn = NumpyTemporalCNN(len(FEATURE_NAMES), 5, 4, 3, seed=5)
        cnn.standardizer.fit(dataset.windows)
        hmm = TemporalHMM.fit([dataset.labels], 5)
        pipeline = ShadowActivityPipeline(
            cnn,
            hmm,
            dataset.class_names,
            dataset.feature_names,
            0.0,
            0.0,
            max_missing_fraction=0.1,
        )
        prediction = pipeline.predict(dataset)
        self.assertTrue(np.all(prediction.accepted_labels == -1))
        self.assertTrue(np.all(prediction.rejection_reason == "missing_data"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            pipeline.save(path)
            loaded = ShadowActivityPipeline.load(path)
        self.assertAlmostEqual(loaded.max_missing_fraction, 0.1)
        self.assertTrue(np.all(loaded.predict(dataset).rejection_reason == "missing_data"))

    def test_tokens_do_not_merge_across_sequences(self):
        dataset = self._dataset()
        labels = np.ones(len(dataset), dtype=int)
        sequences = np.array(["a"] * 5 + ["b"] * 5, dtype=object)
        tokens = build_inertial_tokens(
            labels,
            np.ones(len(dataset)),
            dataset.start_s,
            dataset.end_s,
            dataset.mean_quality,
            dataset.windows,
            dataset.class_names,
            dataset.feature_names,
            "session",
            sequence_ids=sequences,
        )
        self.assertEqual(len(tokens), 2)
        self.assertTrue(all(token.operating_mode == "shadow" for token in tokens))
        self.assertTrue(all(token.safety_effect == "none" for token in tokens))


if __name__ == "__main__":
    unittest.main()
