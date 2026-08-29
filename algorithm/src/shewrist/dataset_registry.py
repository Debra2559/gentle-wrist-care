"""Auditable dataset registry and adapters for SheWrist ML experiments."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ml_data import CLASS_NAMES, WindowDataset, build_public_activity_dataset, concatenate_datasets
from .validation import angle_error_metrics


class DatasetAdapterError(ValueError):
    """Base error for a dataset that cannot satisfy a requested capability."""


class DatasetUnavailableError(DatasetAdapterError):
    """Raised when required local files are absent or incomplete."""


class DatasetCapabilityError(DatasetAdapterError):
    """Raised when a dataset is present but cannot perform the requested task."""


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_id: str
    display_name: str
    root: Path
    adapter: str
    expert_id: str
    doi: str | None
    license: str
    capabilities: tuple[str, ...]
    required_modalities: tuple[str, ...]
    source_label_map: Mapping[str, str]
    mapping_status: str
    artifacts: Mapping[str, str]
    reference_tasks: tuple[Mapping[str, object], ...]
    allowed_uses: tuple[str, ...]
    evidence_limits: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "root": str(self.root),
            "adapter": self.adapter,
            "expert_id": self.expert_id,
            "doi": self.doi,
            "license": self.license,
            "capabilities": list(self.capabilities),
            "required_modalities": list(self.required_modalities),
            "source_label_map": dict(self.source_label_map),
            "mapping_status": self.mapping_status,
            "artifacts": dict(self.artifacts),
            "reference_tasks": [dict(value) for value in self.reference_tasks],
            "allowed_uses": list(self.allowed_uses),
            "evidence_limits": list(self.evidence_limits),
        }


@dataclass(frozen=True)
class DatasetInspection:
    dataset_id: str
    status: str
    installed: bool
    usable_capabilities: tuple[str, ...]
    observed: Mapping[str, object]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "installed": self.installed,
            "usable_capabilities": list(self.usable_capabilities),
            "observed": dict(self.observed),
            "blockers": list(self.blockers),
        }


class DatasetRegistry:
    """Resolve local datasets without pretending unavailable sources are trainable."""

    def __init__(
        self,
        descriptors: Sequence[DatasetDescriptor],
        canonical_activity_labels: Sequence[str],
        fusion_policy: Mapping[str, object],
        project_root: str | Path,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.canonical_activity_labels = tuple(str(value) for value in canonical_activity_labels)
        self.fusion_policy = dict(fusion_policy)
        self._descriptors = {item.dataset_id: item for item in descriptors}
        if not self._descriptors or len(self._descriptors) != len(descriptors):
            raise ValueError("dataset registry IDs must be unique and non-empty")
        if not self.canonical_activity_labels:
            raise ValueError("canonical_activity_labels must not be empty")
        if self.canonical_activity_labels != CLASS_NAMES:
            raise ValueError("canonical_activity_labels must match the model class order")
        canonical = set(self.canonical_activity_labels)
        for descriptor in descriptors:
            if "activity_training" in descriptor.capabilities:
                mapped = set(descriptor.source_label_map.values())
                if not mapped or not mapped.issubset(canonical):
                    raise ValueError(
                        f"{descriptor.dataset_id} maps labels outside the canonical activity ontology"
                    )

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        project_root: str | Path | None = None,
    ) -> "DatasetRegistry":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        root = Path(project_root).resolve() if project_root is not None else config_path.parent.parent.resolve()
        descriptors = []
        for source in payload.get("datasets", []):
            root_path = Path(str(source["root"]))
            if not root_path.is_absolute():
                root_path = root / root_path
            descriptors.append(
                DatasetDescriptor(
                    dataset_id=str(source["dataset_id"]),
                    display_name=str(source["display_name"]),
                    root=root_path.resolve(),
                    adapter=str(source["adapter"]),
                    expert_id=str(source["expert_id"]),
                    doi=None if source.get("doi") is None else str(source["doi"]),
                    license=str(source["license"]),
                    capabilities=tuple(str(value) for value in source.get("capabilities", [])),
                    required_modalities=tuple(str(value) for value in source.get("required_modalities", [])),
                    source_label_map={str(key): str(value) for key, value in source.get("source_label_map", {}).items()},
                    mapping_status=str(source.get("mapping_status", "unspecified")),
                    artifacts={str(key): str(value) for key, value in source.get("artifacts", {}).items()},
                    reference_tasks=tuple(dict(value) for value in source.get("reference_tasks", [])),
                    allowed_uses=tuple(str(value) for value in source.get("allowed_uses", [])),
                    evidence_limits=tuple(str(value) for value in source.get("evidence_limits", [])),
                )
            )
        return cls(
            descriptors,
            payload.get("canonical_activity_labels", []),
            payload.get("fusion_policy", {}),
            root,
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def descriptor(self, dataset_id: str) -> DatasetDescriptor:
        try:
            return self._descriptors[str(dataset_id)]
        except KeyError as exc:
            raise KeyError(f"unknown dataset_id: {dataset_id}") from exc

    def resolve_root(self, dataset_id: str, override: str | Path | None = None) -> Path:
        return self.descriptor(dataset_id).root if override is None else Path(override).resolve()

    def _inspect_upper_body(self, descriptor: DatasetDescriptor, root: Path) -> DatasetInspection:
        subjects = sorted(path for path in root.glob("subject*") if path.is_dir()) if root.is_dir() else []
        required_relative = tuple(
            Path(f"set2/{segment}/{filename}")
            for segment in ("forearm", "hand")
            for filename in ("Accelerometer.txt", "Gyroscope.txt", "Magnetometer.txt")
        )
        complete = [path.name for path in subjects if all((path / relative).is_file() for relative in required_relative)]
        blockers = []
        if not (root / "annotations.csv").is_file():
            blockers.append("missing_annotations_csv")
        if len(complete) != len(subjects) or not complete:
            blockers.append("missing_required_forearm_or_hand_streams")
        ready = not blockers
        return DatasetInspection(
            descriptor.dataset_id,
            "ready" if ready else "incomplete" if root.exists() else "not_installed",
            root.exists(),
            descriptor.capabilities if ready else (),
            {
                "root": str(root),
                "subject_directory_count": len(subjects),
                "complete_set2_subject_count": len(complete),
                "annotations_present": (root / "annotations.csv").is_file(),
            },
            tuple(blockers),
        )

    def _inspect_opto_sample(self, descriptor: DatasetDescriptor, root: Path) -> DatasetInspection:
        relative = descriptor.artifacts.get("aligned_pickle", "")
        aligned = root / relative if relative else root / "missing"
        imu_files = list((root / "Sample Data/IMU").glob("*.csv")) if root.exists() else []
        opto_files = list((root / "Sample Data/OPTO").glob("*.c3d")) if root.exists() else []
        available = aligned.is_file()
        blockers = [] if available else ["missing_aligned_reference_pickle"]
        if available:
            blockers.extend(("full_16_participant_dataset_not_installed", "source_toolbox_baseline_only"))
        return DatasetInspection(
            descriptor.dataset_id,
            "sample_only" if available else "incomplete" if root.exists() else "not_installed",
            root.exists(),
            ("angle_reference_validation",) if available else (),
            {
                "root": str(root),
                "aligned_pickle_present": available,
                "sample_imu_file_count": len(imu_files),
                "sample_opto_file_count": len(opto_files),
                "participant_scope": 1 if available else 0,
            },
            tuple(blockers),
        )

    def _inspect_reserved(self, descriptor: DatasetDescriptor, root: Path) -> DatasetInspection:
        file_count = sum(1 for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0
        installed = root.is_dir() and file_count > 0
        blockers = (
            ("adapter_and_label_mapping_not_implemented",)
            if installed
            else ("dataset_not_installed", "adapter_and_label_mapping_not_implemented")
        )
        return DatasetInspection(
            descriptor.dataset_id,
            "schema_pending" if installed else "not_installed",
            installed,
            (),
            {"root": str(root), "file_count": file_count},
            blockers,
        )

    def _inspect_hardware_pilot(self, descriptor: DatasetDescriptor, root: Path) -> DatasetInspection:
        raw_files = list(root.glob("imu_pressure_*.csv")) if root.is_dir() else []
        angle_files = list(root.glob("wrist_*.csv")) if root.is_dir() else []
        installed = bool(raw_files or angle_files)
        return DatasetInspection(
            descriptor.dataset_id,
            "unlabeled" if installed else "not_installed",
            installed,
            descriptor.capabilities if installed else (),
            {
                "root": str(root),
                "raw_capture_file_count": len(raw_files),
                "legacy_angle_file_count": len(angle_files),
                "supervised_training_eligible": False,
            },
            (
                "missing_participant_activity_condition_and_calibration_labels",
                "not_supervised_training_data",
            ) if installed else ("dataset_not_installed",),
        )

    def inspect(self, dataset_id: str, root_override: str | Path | None = None) -> DatasetInspection:
        descriptor = self.descriptor(dataset_id)
        root = self.resolve_root(dataset_id, root_override)
        if descriptor.adapter == "upper_body_movements":
            return self._inspect_upper_body(descriptor, root)
        if descriptor.adapter == "opto_reference_sample":
            return self._inspect_opto_sample(descriptor, root)
        if descriptor.adapter == "hardware_pilot":
            return self._inspect_hardware_pilot(descriptor, root)
        if descriptor.adapter == "reserved":
            return self._inspect_reserved(descriptor, root)
        raise DatasetAdapterError(f"unsupported adapter: {descriptor.adapter}")

    def inspect_all(self) -> list[DatasetInspection]:
        return [self.inspect(dataset_id) for dataset_id in self.ids()]

    def activity_subjects(self, dataset_id: str, root_override: str | Path | None = None) -> tuple[str, ...]:
        descriptor = self.descriptor(dataset_id)
        if descriptor.adapter != "upper_body_movements":
            raise DatasetCapabilityError(f"{dataset_id} has no implemented activity adapter")
        root = self.resolve_root(dataset_id, root_override)
        return tuple(sorted(path.name for path in root.glob("subject*") if path.is_dir()))

    def build_activity_dataset(
        self,
        dataset_id: str,
        algorithm_config: Mapping[str, object],
        ml_config: Mapping[str, object],
        subjects: Sequence[str] | None = None,
        root_override: str | Path | None = None,
    ) -> WindowDataset:
        descriptor = self.descriptor(dataset_id)
        if "activity_training" not in descriptor.capabilities or descriptor.adapter != "upper_body_movements":
            raise DatasetCapabilityError(f"{dataset_id} does not have an implemented activity-training adapter")
        inspection = self.inspect(dataset_id, root_override)
        if inspection.status != "ready":
            raise DatasetUnavailableError(f"{dataset_id} is {inspection.status}: {list(inspection.blockers)}")
        return build_public_activity_dataset(
            self.resolve_root(dataset_id, root_override),
            algorithm_config,
            ml_config,
            subjects=subjects,
            dataset_id=dataset_id,
            source_label_map=descriptor.source_label_map,
        )

    def build_activity_datasets(
        self,
        dataset_ids: Sequence[str],
        algorithm_config: Mapping[str, object],
        ml_config: Mapping[str, object],
        root_overrides: Mapping[str, str | Path] | None = None,
    ) -> WindowDataset:
        if not dataset_ids:
            raise ValueError("at least one activity dataset ID is required")
        overrides = {} if root_overrides is None else dict(root_overrides)
        datasets = [
            self.build_activity_dataset(
                dataset_id,
                algorithm_config,
                ml_config,
                root_override=overrides.get(dataset_id),
            )
            for dataset_id in dataset_ids
        ]
        if len(datasets) > 1:
            datasets = [
                WindowDataset(
                    windows=dataset.windows,
                    labels=dataset.labels,
                    subject_ids=np.asarray(
                        [f"{dataset_id}:{value}" for value in dataset.subject_ids],
                        dtype=object,
                    ),
                    session_ids=np.asarray(
                        [f"{dataset_id}:{value}" for value in dataset.session_ids],
                        dtype=object,
                    ),
                    sequence_ids=np.asarray(
                        [f"{dataset_id}:{value}" for value in dataset.sequence_ids],
                        dtype=object,
                    ),
                    start_s=dataset.start_s,
                    end_s=dataset.end_s,
                    mean_quality=dataset.mean_quality,
                    feature_names=dataset.feature_names,
                    class_names=dataset.class_names,
                    dataset_ids=dataset.dataset_ids,
                    missing_fraction=dataset.missing_fraction,
                )
                for dataset_id, dataset in zip(dataset_ids, datasets)
            ]
        return concatenate_datasets(datasets)

    def evaluate_angle_reference(
        self,
        dataset_id: str,
        algorithm_config: Mapping[str, object],
        root_override: str | Path | None = None,
        aligned_pickle_override: str | Path | None = None,
    ) -> dict[str, object]:
        descriptor = self.descriptor(dataset_id)
        if descriptor.adapter != "opto_reference_sample":
            raise DatasetCapabilityError(f"{dataset_id} has no implemented angle-reference adapter")
        inspection = self.inspect(dataset_id, root_override)
        if "angle_reference_validation" not in inspection.usable_capabilities:
            raise DatasetUnavailableError(f"{dataset_id} is {inspection.status}: {list(inspection.blockers)}")
        root = self.resolve_root(dataset_id, root_override)
        aligned = (
            Path(aligned_pickle_override).resolve()
            if aligned_pickle_override is not None
            else root / descriptor.artifacts["aligned_pickle"]
        )
        with aligned.open("rb") as handle:
            opto, imu = pickle.load(handle)
        results: dict[str, object] = {}
        for task in descriptor.reference_tasks:
            name = str(task["name"])
            task_index = int(task["task_index"])
            axis_index = int(task["axis_index"])
            reference = np.asarray(opto[task_index]["Right Wrist"], dtype=float)[:, axis_index]
            estimate = np.asarray(imu[task_index]["Right Wrist"], dtype=float)[:, axis_index]
            result = angle_error_metrics(reference, estimate, sample_rate_hz=60.0)
            centered = angle_error_metrics(
                reference - np.mean(reference),
                estimate - np.mean(estimate),
                sample_rate_hz=60.0,
            )
            result["centered_mae_deg"] = centered["mae_deg"]
            result["centered_rmse_deg"] = centered["rmse_deg"]
            result["dynamic_acceptance_mae_pass"] = result["mae_deg"] <= float(
                algorithm_config["acceptance"]["dynamic_mae_deg_max"]
            )
            results[name] = result
        return {
            "dataset_id": descriptor.dataset_id,
            "source": {
                "name": descriptor.display_name + " public sample",
                "doi": descriptor.doi,
                "license": descriptor.license,
                "local_artifact": str(aligned),
            },
            "scope": "One public participant; source-toolbox IMU angles aligned to optical motion capture.",
            "important_boundary": (
                "These numbers characterize the source toolbox baseline, not the raw-sensor "
                "SheWrist implementation. They validate the independent angle-error reporting path only."
            ),
            "results": results,
        }

    def readiness_summary(self) -> dict[str, object]:
        inspections = self.inspect_all()
        ready_activity = [
            item.dataset_id
            for item in inspections
            if "activity_training" in item.usable_capabilities and item.status == "ready"
        ]
        angle_sources = [
            item.dataset_id for item in inspections if "angle_reference_validation" in item.usable_capabilities
        ]
        cross_dataset = {
            "status": "ready" if len(ready_activity) >= 2 else "not_evaluable",
            "available_labeled_activity_dataset_ids": ready_activity,
            "minimum_required_dataset_count": 2,
            "reason": None if len(ready_activity) >= 2 else "only_one_compatible_labeled_activity_dataset_is_ready",
        }
        return {
            "schema_version": 1,
            "registered_dataset_count": len(inspections),
            "canonical_activity_labels": list(self.canonical_activity_labels),
            "datasets": [
                {
                    **self.descriptor(item.dataset_id).to_dict(),
                    **item.to_dict(),
                }
                for item in inspections
            ],
            "ready_activity_dataset_ids": ready_activity,
            "usable_angle_reference_dataset_ids": angle_sources,
            "cross_dataset_activity_evaluation": cross_dataset,
            "fusion_policy": self.fusion_policy,
        }