"""Structured, non-clinical inertial tokens and deterministic feedback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class InertialToken:
    schema_version: str
    session_id: str
    event_type: str
    source: str
    evidence_type: str
    operating_mode: str
    start_ms: int
    end_ms: int
    duration_ms: int
    confidence: float
    mean_quality: float
    peak_abs_fe_deg: float
    peak_abs_rud_deg: float
    model_name: str
    model_version: str
    threshold_version: str
    safety_effect: str = "none"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _event_runs(labels: np.ndarray, sequence_ids: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.asarray(labels, dtype=int)
    sequences = np.asarray(sequence_ids, dtype=object)
    if len(values) != len(sequences):
        raise ValueError("labels and sequence_ids must have equal length")
    runs: list[tuple[int, int, int]] = []
    start = 0
    while start < len(values):
        label = int(values[start])
        sequence = sequences[start]
        end = start + 1
        while end < len(values) and int(values[end]) == label and sequences[end] == sequence:
            end += 1
        runs.append((start, end, label))
        start = end
    return runs


def build_inertial_tokens(
    labels: np.ndarray,
    confidence: np.ndarray,
    window_start_s: np.ndarray,
    window_end_s: np.ndarray,
    mean_quality: np.ndarray,
    windows: np.ndarray,
    class_names: Sequence[str],
    feature_names: Sequence[str],
    session_id: str,
    sequence_ids: Sequence[str] | None = None,
    model_name: str = "shewrist_numpy_1dcnn_hmm",
    model_version: str = "1.0",
    threshold_version: str = "engineering_v1",
    source: str = "derived",
    evidence_type: str = "replay",
    operating_mode: str = "shadow",
    unknown_label: int = -1,
) -> list[InertialToken]:
    """Merge consecutive non-background predictions into auditable events."""
    prediction = np.asarray(labels, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    start_s = np.asarray(window_start_s, dtype=float)
    end_s = np.asarray(window_end_s, dtype=float)
    quality = np.asarray(mean_quality, dtype=float)
    sequences = np.full(len(prediction), session_id, dtype=object) if sequence_ids is None else np.asarray(sequence_ids, dtype=object)
    x = np.asarray(windows, dtype=float)
    n = len(prediction)
    if not all(len(values) == n for values in (confidence, start_s, end_s, quality, sequences, x)):
        raise ValueError("token inputs must have equal length")
    if x.ndim != 3 or x.shape[2] != len(feature_names):
        raise ValueError("windows must match feature_names")
    feature_index = {name: index for index, name in enumerate(feature_names)}
    fe_index = feature_index.get("theta_fe_deg")
    rud_index = feature_index.get("theta_rud_deg")
    tokens: list[InertialToken] = []
    for run_start, run_end, label in _event_runs(prediction, sequences):
        if label in {0, unknown_label} or label < 0 or label >= len(class_names):
            continue
        chunk = x[run_start:run_end]
        peak_fe = float(np.nanmax(np.abs(chunk[:, :, fe_index]))) if fe_index is not None else float("nan")
        peak_rud = float(np.nanmax(np.abs(chunk[:, :, rud_index]))) if rud_index is not None else float("nan")
        first_ms = int(round(1000.0 * start_s[run_start]))
        last_ms = int(round(1000.0 * end_s[run_end - 1]))
        tokens.append(
            InertialToken(
                schema_version="1.0",
                session_id=session_id,
                event_type=str(class_names[label]),
                source=source,
                evidence_type=evidence_type,
                operating_mode=operating_mode,
                start_ms=first_ms,
                end_ms=last_ms,
                duration_ms=max(0, last_ms - first_ms),
                confidence=float(np.mean(confidence[run_start:run_end])),
                mean_quality=float(np.mean(quality[run_start:run_end])),
                peak_abs_fe_deg=peak_fe,
                peak_abs_rud_deg=peak_rud,
                model_name=model_name,
                model_version=model_version,
                threshold_version=threshold_version,
            )
        )
    return tokens


def feedback_from_token(token: InertialToken) -> str:
    """Return deterministic wording; it intentionally makes no medical claim."""
    label = token.event_type.replace("_", " ")
    return (
        f"影子模型在 {token.duration_ms / 1000.0:.1f} 秒窗口内识别到 {label} 模式，"
        f"平均置信度 {token.confidence:.2f}、数据质量 {token.mean_quality:.2f}。"
        "该结果只用于公开数据动作识别演示，不参与报警、机械控制或疾病判断。"
    )