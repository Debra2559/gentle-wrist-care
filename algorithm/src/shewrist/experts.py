"""Task-specific expert contracts and evidence-gated probability fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .ml import ShadowActivityPipeline
from .ml_data import WindowDataset


@dataclass(frozen=True)
class ExpertContract:
    expert_id: str
    task: str
    dataset_ids: tuple[str, ...]
    output_labels: tuple[str, ...]
    required_features: tuple[str, ...]
    operating_mode: str = "shadow"
    alarm_control_authority: str = "none"
    mechanical_control_authority: str = "none"


@dataclass(frozen=True)
class ExpertPrediction:
    expert_id: str
    labels: tuple[str, ...]
    probabilities: np.ndarray
    confidence: np.ndarray
    quality: np.ndarray
    compatibility: np.ndarray
    available: np.ndarray
    rejection_reason: np.ndarray

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        n = len(probabilities)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.labels):
            raise ValueError("expert probabilities must have shape (windows, labels)")
        if any(len(np.asarray(values)) != n for values in (self.confidence, self.quality, self.compatibility, self.available, self.rejection_reason)):
            raise ValueError("expert prediction metadata lengths must match")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("expert probabilities must be finite and non-negative")
        compatibility = np.asarray(self.compatibility, dtype=float)
        if np.any(~np.isfinite(compatibility)) or np.any((compatibility < 0.0) | (compatibility > 1.0)):
            raise ValueError("expert compatibility must be finite and within 0..1")


class ExpertModel(Protocol):
    contract: ExpertContract

    def predict(self, dataset: WindowDataset) -> ExpertPrediction:
        ...


class ShadowActivityExpert:
    """Expose the existing CNN-HMM as one non-controlling expert."""

    def __init__(self, expert_id: str, pipeline: ShadowActivityPipeline, dataset_ids: Sequence[str]) -> None:
        self.pipeline = pipeline
        self.contract = ExpertContract(
            expert_id=expert_id,
            task="five_class_wrist_motion",
            dataset_ids=tuple(str(value) for value in dataset_ids),
            output_labels=tuple(pipeline.class_names),
            required_features=tuple(pipeline.feature_names),
        )

    def predict(self, dataset: WindowDataset) -> ExpertPrediction:
        prediction = self.pipeline.predict(dataset)
        return ExpertPrediction(
            expert_id=self.contract.expert_id,
            labels=self.contract.output_labels,
            probabilities=prediction.probabilities,
            confidence=prediction.confidence,
            quality=dataset.mean_quality,
            compatibility=np.ones(len(dataset), dtype=float),
            available=prediction.accepted_labels >= 0,
            rejection_reason=prediction.rejection_reason,
        )


class ReservedExpert:
    """Return explicit unavailability until a real adapter and model exist."""

    def __init__(self, contract: ExpertContract, reason: str) -> None:
        self.contract = contract
        self.reason = str(reason)

    def predict(self, dataset: WindowDataset) -> ExpertPrediction:
        count = len(dataset)
        return ExpertPrediction(
            expert_id=self.contract.expert_id,
            labels=self.contract.output_labels,
            probabilities=np.zeros((count, len(self.contract.output_labels)), dtype=float),
            confidence=np.zeros(count, dtype=float),
            quality=np.asarray(dataset.mean_quality, dtype=float),
            compatibility=np.zeros(count, dtype=float),
            available=np.zeros(count, dtype=bool),
            rejection_reason=np.full(count, self.reason, dtype=object),
        )


@dataclass(frozen=True)
class ValidatedFusionWeights:
    """Weights are valid only when tied to a labeled target-hardware validation set."""

    weights: Mapping[str, float]
    target_validation_dataset_id: str
    validation_metric: str

    def __post_init__(self) -> None:
        if not self.target_validation_dataset_id.strip():
            raise ValueError("target_validation_dataset_id is required; weights cannot be guessed")
        if not self.validation_metric.strip():
            raise ValueError("validation_metric is required")
        if not self.weights:
            raise ValueError("at least one validated expert weight is required")
        values = np.asarray(list(self.weights.values()), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("validated expert weights must be finite and positive")


@dataclass(frozen=True)
class FusedPrediction:
    labels: tuple[str, ...]
    probabilities: np.ndarray
    accepted_labels: np.ndarray
    confidence: np.ndarray
    contributing_experts: np.ndarray
    rejection_reason: np.ndarray
    control_authority: str = "none"


def fuse_expert_probabilities(
    predictions: Sequence[ExpertPrediction],
    validated_weights: ValidatedFusionWeights,
) -> FusedPrediction:
    """Fuse only the common label intersection using externally validated weights."""
    selected = [item for item in predictions if item.expert_id in validated_weights.weights]
    if not selected:
        raise ValueError("no predictions match the validated expert weights")
    count = len(selected[0].probabilities)
    if any(len(item.probabilities) != count for item in selected):
        raise ValueError("expert predictions must cover the same windows")
    common = [label for label in selected[0].labels if all(label in item.labels for item in selected[1:])]
    if not common:
        raise ValueError("experts have no common output labels to fuse")
    weighted = np.zeros((count, len(common)), dtype=float)
    denominator = np.zeros(count, dtype=float)
    contributors = np.zeros(count, dtype=int)
    for item in selected:
        columns = [item.labels.index(label) for label in common]
        probability = item.probabilities[:, columns]
        probability_sum = np.sum(probability, axis=1, keepdims=True)
        normalized = np.divide(
            probability,
            probability_sum,
            out=np.zeros_like(probability),
            where=probability_sum > 0.0,
        )
        base_weight = float(validated_weights.weights[item.expert_id])
        effective = base_weight * np.clip(np.asarray(item.quality, dtype=float), 0.0, 1.0)
        effective *= np.clip(np.asarray(item.compatibility, dtype=float), 0.0, 1.0)
        effective *= np.asarray(item.available, dtype=bool)
        effective *= probability_sum[:, 0] > 0.0
        weighted += normalized * effective[:, None]
        denominator += effective
        contributors += effective > 0.0
    probabilities = np.divide(
        weighted,
        denominator[:, None],
        out=np.zeros_like(weighted),
        where=denominator[:, None] > 0.0,
    )
    accepted = np.full(count, -1, dtype=int)
    available = denominator > 0.0
    accepted[available] = np.argmax(probabilities[available], axis=1)
    confidence = np.zeros(count, dtype=float)
    confidence[available] = np.max(probabilities[available], axis=1)
    reasons = np.full(count, "no_compatible_expert_prediction", dtype=object)
    reasons[available] = "accepted"
    return FusedPrediction(
        labels=tuple(common),
        probabilities=probabilities,
        accepted_labels=accepted,
        confidence=confidence,
        contributing_experts=contributors,
        rejection_reason=reasons,
    )