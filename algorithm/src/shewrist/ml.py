"""Dependency-light 1D-CNN activity model for shadow-mode experiments.

The implementation intentionally depends only on NumPy so the existing project
runs unchanged on a clean Mac.  It is an activity classifier, not a tissue-load,
pain, disease-risk, or safety model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .hmm import TemporalHMM
from .ml_data import WindowDataset


_EPS = 1e-12
_UNKNOWN = -1


@dataclass
class FeatureStandardizer:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> "FeatureStandardizer":
        x = np.asarray(windows, dtype=float)
        if x.ndim != 3:
            raise ValueError("windows must have shape (n, time, features)")
        self.mean = np.nanmean(x, axis=(0, 1)).astype(np.float32)
        scale = np.nanstd(x, axis=(0, 1))
        self.scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError("standardizer has not been fitted")
        x = np.asarray(windows, dtype=np.float32)
        transformed = (x - self.mean[None, None, :]) / self.scale[None, None, :]
        return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_windows(
    windows: np.ndarray,
    feature_names: Sequence[str],
    config: Mapping[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32).copy()
    if not bool(config.get("enabled", True)):
        return x
    quality_index = feature_names.index("quality") if "quality" in feature_names else None
    signal_indices = [index for index, name in enumerate(feature_names) if name != "quality"]
    batch = len(x)
    scale = rng.uniform(
        float(config.get("amplitude_scale_min", 1.0)),
        float(config.get("amplitude_scale_max", 1.0)),
        size=(batch, 1, 1),
    ).astype(np.float32)
    x[:, :, signal_indices] *= scale
    bias_std = float(config.get("bias_std", 0.0))
    if bias_std > 0.0:
        spread = np.std(x[:, :, signal_indices], axis=(0, 1), keepdims=True)
        bias = rng.normal(0.0, bias_std, size=(batch, 1, len(signal_indices))).astype(np.float32)
        x[:, :, signal_indices] += bias * np.maximum(spread, 1e-3)
    noise_std = float(config.get("gaussian_noise_std", 0.0))
    if noise_std > 0.0:
        spread = np.std(x[:, :, signal_indices], axis=(0, 1), keepdims=True)
        noise = rng.normal(size=(batch, x.shape[1], len(signal_indices))).astype(np.float32)
        x[:, :, signal_indices] += noise_std * noise * np.maximum(spread, 1e-3)
    max_shift = int(config.get("max_time_shift_samples", 0))
    if max_shift > 0:
        for index in range(batch):
            shift = int(rng.integers(-max_shift, max_shift + 1))
            if shift != 0:
                x[index] = np.roll(x[index], shift, axis=0)
                if shift > 0:
                    x[index, :shift] = x[index, shift]
                else:
                    x[index, shift:] = x[index, shift - 1]
    time_probability = float(config.get("time_mask_probability", 0.0))
    max_fraction = float(config.get("time_mask_fraction_max", 0.0))
    for index in range(batch):
        if rng.random() < time_probability and max_fraction > 0.0:
            length = max(1, int(round(rng.uniform(0.02, max_fraction) * x.shape[1])))
            start = int(rng.integers(0, max(1, x.shape[1] - length + 1)))
            x[index, start : start + length, signal_indices] = 0.0
            if quality_index is not None:
                x[index, start : start + length, quality_index] = 0.0
        if rng.random() < float(config.get("channel_mask_probability", 0.0)):
            channel = int(rng.choice(signal_indices))
            x[index, :, channel] = 0.0
            if quality_index is not None:
                x[index, :, quality_index] = np.minimum(x[index, :, quality_index], 0.5)
    return x


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def classification_metrics(true: np.ndarray, predicted: np.ndarray, class_names: Sequence[str]) -> dict[str, object]:
    y = np.asarray(true, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    if y.shape != pred.shape:
        raise ValueError("true and predicted labels must match")
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes + 1), dtype=int)
    for truth, estimate in zip(y, pred):
        if 0 <= truth < n_classes:
            column = estimate if 0 <= estimate < n_classes else n_classes
            confusion[truth, column] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, name in enumerate(class_names):
        tp = int(confusion[index, index])
        fp = int(np.sum(confusion[:, index]) - tp)
        fn = int(np.sum(confusion[index, :]) - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "support": int(np.sum(confusion[index])),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return {
        "n_windows": int(len(y)),
        "accuracy": float(np.mean(y == pred)) if len(y) else 0.0,
        "selective_accuracy": float(np.mean(y[pred != _UNKNOWN] == pred[pred != _UNKNOWN])) if np.any(pred != _UNKNOWN) else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "coverage": float(np.mean(pred != _UNKNOWN)) if len(y) else 0.0,
        "unknown_count": int(np.count_nonzero(pred == _UNKNOWN)),
        "confusion_columns": list(class_names) + ["unknown"],
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


class NumpyTemporalCNN:
    """One convolution, ReLU, temporal pooling, and softmax."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        filters: int,
        kernel_size: int,
        seed: int = 0,
        pooling: str = "mean",
    ) -> None:
        if n_features < 1 or n_classes < 2 or filters < 1 or kernel_size < 1:
            raise ValueError("invalid CNN dimensions")
        if pooling not in {"mean", "mean_max"}:
            raise ValueError("pooling must be 'mean' or 'mean_max'")
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.rng = np.random.default_rng(seed)
        conv_scale = np.sqrt(2.0 / (kernel_size * n_features))
        pooled_features = filters if pooling == "mean" else 2 * filters
        dense_scale = np.sqrt(2.0 / pooled_features)
        self.parameters = {
            "conv_weight": self.rng.normal(0.0, conv_scale, (filters, kernel_size, n_features)).astype(np.float32),
            "conv_bias": np.zeros(filters, dtype=np.float32),
            "dense_weight": self.rng.normal(0.0, dense_scale, (pooled_features, n_classes)).astype(np.float32),
            "dense_bias": np.zeros(n_classes, dtype=np.float32),
        }
        self.standardizer = FeatureStandardizer()
        self.temperature = 1.0
        self.history: list[dict[str, float | int]] = []
        self.best_epoch = 0

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if x.shape[1] < self.kernel_size:
            raise ValueError("window is shorter than convolution kernel")
        patches = np.lib.stride_tricks.sliding_window_view(x, self.kernel_size, axis=1).transpose(0, 1, 3, 2)
        conv = np.einsum("btkc,fkc->btf", patches, self.parameters["conv_weight"], optimize=True)
        conv += self.parameters["conv_bias"][None, None, :]
        activated = np.maximum(conv, 0.0)
        mean_pooled = np.mean(activated, axis=1)
        pooled = mean_pooled if self.pooling == "mean" else np.concatenate((mean_pooled, np.max(activated, axis=1)), axis=1)
        logits = np.einsum("bf,fc->bc", pooled, self.parameters["dense_weight"], optimize=False)
        logits += self.parameters["dense_bias"]
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("non-finite CNN logits")
        return logits, (patches, conv, activated, pooled)

    def _loss_and_gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_weight: np.ndarray,
        weight_decay: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self._forward(x)
        probabilities = _softmax(logits)
        sample_weight = class_weight[y]
        normalizer = float(np.sum(sample_weight))
        loss = -float(np.sum(sample_weight * np.log(np.clip(probabilities[np.arange(len(y)), y], _EPS, 1.0))) / normalizer)
        loss += 0.5 * weight_decay * (
            float(np.sum(self.parameters["conv_weight"] ** 2))
            + float(np.sum(self.parameters["dense_weight"] ** 2))
        )
        dlogits = probabilities
        dlogits[np.arange(len(y)), y] -= 1.0
        dlogits *= sample_weight[:, None] / normalizer
        patches, conv, activated, pooled = cache
        gradients: dict[str, np.ndarray] = {}
        gradients["dense_weight"] = np.einsum("bf,bc->fc", pooled, dlogits, optimize=False)
        gradients["dense_weight"] += weight_decay * self.parameters["dense_weight"]
        gradients["dense_bias"] = np.sum(dlogits, axis=0)
        dpooled = np.einsum("bc,fc->bf", dlogits, self.parameters["dense_weight"], optimize=False)
        if self.pooling == "mean":
            dactivated = np.broadcast_to(dpooled[:, None, :] / conv.shape[1], conv.shape).copy()
        else:
            dmean = dpooled[:, : self.filters]
            dmax = dpooled[:, self.filters :]
            dactivated = np.broadcast_to(dmean[:, None, :] / conv.shape[1], conv.shape).copy()
            max_indices = np.argmax(activated, axis=1)
            batch_indices = np.arange(len(x))
            for filter_index in range(self.filters):
                dactivated[batch_indices, max_indices[:, filter_index], filter_index] += dmax[:, filter_index]
        dconv = dactivated * (conv > 0.0)
        gradients["conv_weight"] = np.einsum("btf,btkc->fkc", dconv, patches, optimize=True)
        gradients["conv_weight"] += weight_decay * self.parameters["conv_weight"]
        gradients["conv_bias"] = np.sum(dconv, axis=(0, 1))
        return loss, gradients

    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probabilities = _softmax(self._forward(x)[0])
        return -float(np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y], _EPS, 1.0))))

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray | None,
        validation_y: np.ndarray | None,
        feature_names: Sequence[str],
        model_config: Mapping[str, object],
        augmentation_config: Mapping[str, object] | None = None,
        epochs: int | None = None,
    ) -> list[dict[str, float | int]]:
        x = np.asarray(train_x, dtype=np.float32)
        y = np.asarray(train_y, dtype=int)
        if x.ndim != 3 or x.shape[2] != self.n_features or len(x) != len(y):
            raise ValueError("invalid training shapes")
        if np.any((y < 0) | (y >= self.n_classes)):
            raise ValueError("training labels are out of range")
        self.standardizer.fit(x)
        val_x = None if validation_x is None else self.standardizer.transform(validation_x)
        val_y = None if validation_y is None else np.asarray(validation_y, dtype=int)
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        if np.any(counts == 0):
            raise ValueError("every class needs at least one training window")
        class_weight = np.sqrt(np.max(counts) / counts)
        class_weight /= np.mean(class_weight)
        learning_rate = float(model_config.get("learning_rate", 0.003))
        weight_decay = float(model_config.get("weight_decay", 0.0))
        batch_size = int(model_config.get("batch_size", 128))
        max_epochs = int(model_config.get("epochs", 30) if epochs is None else epochs)
        patience = int(model_config.get("patience", max_epochs))
        clip_norm = float(model_config.get("gradient_clip_norm", 5.0))
        augmentation = {} if augmentation_config is None else dict(augmentation_config)
        first_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        second_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        beta1, beta2 = 0.9, 0.999
        step = 0
        best_loss = np.inf
        best_parameters = {name: value.copy() for name, value in self.parameters.items()}
        remaining_patience = patience
        self.history = []
        start_time = perf_counter()
        for epoch in range(1, max_epochs + 1):
            order = self.rng.permutation(len(x))
            losses = []
            for batch_start in range(0, len(order), batch_size):
                indices = order[batch_start : batch_start + batch_size]
                raw_batch = augment_windows(x[indices], feature_names, augmentation, self.rng)
                batch = self.standardizer.transform(raw_batch)
                loss, gradients = self._loss_and_gradients(batch, y[indices], class_weight, weight_decay)
                losses.append(loss)
                total_norm = np.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients.values()))
                if total_norm > clip_norm:
                    scale = clip_norm / max(total_norm, _EPS)
                    gradients = {name: gradient * scale for name, gradient in gradients.items()}
                step += 1
                for name in self.parameters:
                    gradient = gradients[name].astype(np.float32)
                    first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradient
                    second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * (gradient * gradient)
                    corrected_first = first_moment[name] / (1.0 - beta1**step)
                    corrected_second = second_moment[name] / (1.0 - beta2**step)
                    self.parameters[name] -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            evaluation_x = self.standardizer.transform(x) if val_x is None else val_x
            evaluation_y = y if val_y is None else val_y
            validation_loss = self._loss(evaluation_x, evaluation_y)
            prediction = np.argmax(_softmax(self._forward(evaluation_x)[0]), axis=1)
            macro_f1 = float(classification_metrics(evaluation_y, prediction, tuple(str(i) for i in range(self.n_classes)))["macro_f1"])
            self.history.append(
                {
                    "epoch": epoch,
                    "training_loss": float(np.mean(losses)),
                    "validation_loss": validation_loss,
                    "validation_macro_f1": macro_f1,
                }
            )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_parameters = {name: value.copy() for name, value in self.parameters.items()}
                self.best_epoch = epoch
                remaining_patience = patience
            else:
                remaining_patience -= 1
                if val_x is not None and remaining_patience <= 0:
                    break
        self.parameters = best_parameters
        if val_x is not None and val_y is not None:
            self.calibrate_temperature(validation_x, val_y)
        if self.history:
            self.history[-1]["elapsed_s"] = perf_counter() - start_time
        return self.history

    def calibrate_temperature(self, windows: np.ndarray, labels: np.ndarray) -> float:
        x = self.standardizer.transform(windows)
        y = np.asarray(labels, dtype=int)
        logits = self._forward(x)[0]
        candidates = np.geomspace(0.25, 4.0, 81)
        losses = []
        for temperature in candidates:
            probabilities = _softmax(logits / temperature)
            losses.append(-float(np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y], _EPS, 1.0)))))
        self.temperature = float(candidates[int(np.argmin(losses))])
        return self.temperature

    def predict_proba(self, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
        x = np.asarray(windows, dtype=np.float32)
        outputs = []
        for start in range(0, len(x), batch_size):
            batch = self.standardizer.transform(x[start : start + batch_size])
            outputs.append(_softmax(self._forward(batch)[0] / self.temperature))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, self.n_classes), dtype=float)


@dataclass(frozen=True)
class ShadowPrediction:
    raw_labels: np.ndarray
    smoothed_labels: np.ndarray
    accepted_labels: np.ndarray
    confidence: np.ndarray
    probabilities: np.ndarray
    rejection_reason: np.ndarray


class ShadowActivityPipeline:
    def __init__(
        self,
        cnn: NumpyTemporalCNN,
        hmm: TemporalHMM,
        class_names: Sequence[str],
        feature_names: Sequence[str],
        confidence_threshold: float,
        min_window_quality: float,
        metadata: Mapping[str, object] | None = None,
        max_missing_fraction: float = 0.1,
    ) -> None:
        self.cnn = cnn
        self.hmm = hmm
        self.class_names = tuple(class_names)
        self.feature_names = tuple(feature_names)
        self.confidence_threshold = float(confidence_threshold)
        self.min_window_quality = float(min_window_quality)
        self.max_missing_fraction = float(max_missing_fraction)
        if not 0.0 <= self.max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must be within 0..1")
        self.metadata = {} if metadata is None else dict(metadata)

    def predict(self, dataset: WindowDataset) -> ShadowPrediction:
        probabilities = self.cnn.predict_proba(dataset.windows)
        raw = np.argmax(probabilities, axis=1)
        smoothed = raw.copy()
        for sequence in dict.fromkeys(dataset.sequence_ids.tolist()):
            indices = np.flatnonzero(dataset.sequence_ids == sequence)
            smoothed[indices] = self.hmm.decode(probabilities[indices])
        confidence = probabilities[np.arange(len(smoothed)), smoothed]
        accepted = smoothed.copy()
        reasons = np.full(len(smoothed), "accepted", dtype=object)
        low_quality = dataset.mean_quality < self.min_window_quality
        low_confidence = confidence < self.confidence_threshold
        excessive_missing = dataset.missing_fraction > self.max_missing_fraction
        accepted[low_quality | low_confidence | excessive_missing] = _UNKNOWN
        reasons[low_confidence] = "low_confidence"
        reasons[low_quality] = "low_quality"
        reasons[excessive_missing] = "missing_data"
        return ShadowPrediction(raw, smoothed, accepted, confidence, probabilities, reasons)

    def save(self, path: str | Path) -> None:
        if self.cnn.standardizer.mean is None or self.cnn.standardizer.scale is None:
            raise ValueError("cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "class_names": list(self.class_names),
            "feature_names": list(self.feature_names),
            "confidence_threshold": self.confidence_threshold,
            "min_window_quality": self.min_window_quality,
            "max_missing_fraction": self.max_missing_fraction,
            "n_features": self.cnn.n_features,
            "n_classes": self.cnn.n_classes,
            "filters": self.cnn.filters,
            "kernel_size": self.cnn.kernel_size,
            "pooling": self.cnn.pooling,
            "temperature": self.cnn.temperature,
            "operating_mode": "shadow",
            "allow_alarm_control": False,
            "allow_mechanical_control": False,
        }
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            conv_weight=self.cnn.parameters["conv_weight"],
            conv_bias=self.cnn.parameters["conv_bias"],
            dense_weight=self.cnn.parameters["dense_weight"],
            dense_bias=self.cnn.parameters["dense_bias"],
            feature_mean=self.cnn.standardizer.mean,
            feature_scale=self.cnn.standardizer.scale,
            hmm_start=self.hmm.start_probability,
            hmm_transition=self.hmm.transition_probability,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ShadowActivityPipeline":
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            cnn = NumpyTemporalCNN(
                int(metadata["n_features"]),
                int(metadata["n_classes"]),
                int(metadata["filters"]),
                int(metadata["kernel_size"]),
                pooling=str(metadata.get("pooling", "mean")),
            )
            for name in ("conv_weight", "conv_bias", "dense_weight", "dense_bias"):
                cnn.parameters[name] = np.asarray(payload[name], dtype=np.float32)
            cnn.standardizer.mean = np.asarray(payload["feature_mean"], dtype=np.float32)
            cnn.standardizer.scale = np.asarray(payload["feature_scale"], dtype=np.float32)
            cnn.temperature = float(metadata.get("temperature", 1.0))
            hmm = TemporalHMM(np.asarray(payload["hmm_start"]), np.asarray(payload["hmm_transition"]))
        return cls(
            cnn,
            hmm,
            metadata["class_names"],
            metadata["feature_names"],
            float(metadata["confidence_threshold"]),
            float(metadata["min_window_quality"]),
            metadata,
            float(metadata.get("max_missing_fraction", 0.1)),
        )


def label_sequences(dataset: WindowDataset, indices: np.ndarray) -> list[np.ndarray]:
    sequences = []
    for sequence in dict.fromkeys(dataset.sequence_ids[indices].tolist()):
        selected = indices[dataset.sequence_ids[indices] == sequence]
        if len(selected):
            sequences.append(dataset.labels[selected])
    return sequences