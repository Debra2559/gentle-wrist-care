"""Chunked offline replay and batch-equivalence checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Mapping

import numpy as np

from .exposure import ExposureEngine


def chunk_slices(length: int, chunk_size: int) -> list[slice]:
    if length < 1:
        raise ValueError("replay input is empty")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [slice(start, min(length, start + chunk_size)) for start in range(0, length, chunk_size)]


def _sample_length(payload: Mapping[str, object]) -> int:
    lengths = {len(np.asarray(value)) for value in payload.values() if np.asarray(value).ndim > 0}
    if len(lengths) != 1:
        raise ValueError("all replay arrays must have equal length")
    return next(iter(lengths))


def replay_mapping(payload: Mapping[str, object], chunk_size: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Ingest arrays in chunks, preserve order, and finalize a full-session buffer."""
    length = _sample_length(payload)
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in payload}
    slices = chunk_slices(length, chunk_size)
    previous_timestamp: float | None = None
    for index in slices:
        for key, value in payload.items():
            chunk = np.asarray(value)[index].copy()
            pieces[key].append(chunk)
        if "timestamp_ms" in payload:
            timestamp = np.asarray(payload["timestamp_ms"], dtype=float)[index]
            if len(timestamp) and previous_timestamp is not None and timestamp[0] <= previous_timestamp:
                raise ValueError("chunked replay received non-increasing timestamps")
            if len(timestamp) and np.any(np.diff(timestamp) <= 0.0):
                raise ValueError("chunked replay received non-increasing timestamps")
            if len(timestamp):
                previous_timestamp = float(timestamp[-1])
    reconstructed = {key: np.concatenate(values) for key, values in pieces.items()}
    equal = True
    for key, value in payload.items():
        original = np.asarray(value)
        restored = reconstructed[key]
        if np.issubdtype(original.dtype, np.number):
            equal = equal and bool(np.array_equal(original, restored, equal_nan=True))
        else:
            equal = equal and bool(np.array_equal(original, restored))
    return reconstructed, {
        "mode": "chunked_ingest_full_session_finalize",
        "chunk_size_samples": int(chunk_size),
        "chunk_count": len(slices),
        "sample_count": length,
        "input_reconstruction_equal": bool(equal),
        "online_ml_semantics": "ML/HMM finalizes after the complete replay buffer; no claim of causal real-time HMM output.",
    }


def stream_exposure_states(
    joint_state: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    chunk_size: int,
    pressure_kpa: np.ndarray | None = None,
    discomfort: np.ndarray | None = None,
    user_continues: np.ndarray | None = None,
    angle_alerts_enabled: bool = True,
    mechanical_recommendations_enabled: bool = True,
) -> list[dict[str, object]]:
    """Run the stateful deterministic engine across chunk boundaries without reset."""
    length = _sample_length(joint_state)
    timestamp = np.asarray(joint_state["timestamp_ms"], dtype=float) / 1000.0
    fe = np.asarray(joint_state["theta_FE"], dtype=float)
    rud = np.asarray(joint_state["theta_RUD"], dtype=float)
    quality = np.asarray(joint_state.get("quality", np.ones(length)), dtype=float)
    pressure = np.full(length, np.nan) if pressure_kpa is None else np.asarray(pressure_kpa, dtype=float)
    discomfort_values = np.zeros(length, dtype=bool) if discomfort is None else np.asarray(discomfort, dtype=bool)
    continues_values = np.ones(length, dtype=bool) if user_continues is None else np.asarray(user_continues, dtype=bool)
    if len(pressure) != length or len(discomfort_values) != length or len(continues_values) != length:
        raise ValueError("optional replay arrays must match joint_state")
    engine = ExposureEngine(config)
    rows: list[dict[str, object]] = []
    for span in chunk_slices(length, chunk_size):
        for index in range(span.start, span.stop):
            state = engine.update(
                timestamp[index],
                fe[index],
                rud[index],
                None if not np.isfinite(pressure[index]) else float(pressure[index]),
                bool(discomfort_values[index]),
                bool(np.isfinite(quality[index]) and quality[index] >= 0.2),
                bool(continues_values[index]),
                angle_alerts_enabled,
                mechanical_recommendations_enabled,
            )
            rows.append(asdict(state))
    return rows


def canonical_fingerprint(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_chunked_replay(
    joint_state: Mapping[str, np.ndarray],
    algorithm_config: Mapping[str, object],
    ml_config: Mapping[str, object],
    activity_pipeline,
    session_id: str,
    evidence_type: str,
    chunk_size: int,
    pressure_kpa: np.ndarray | None = None,
    discomfort: np.ndarray | None = None,
    cable_tension_n: np.ndarray | None = None,
    lever_arm_m: float | np.ndarray | None = None,
    user_continues: np.ndarray | None = None,
    angle_alerts_enabled: bool = True,
    mechanical_recommendations_enabled: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    from .pipeline import analyze_with_shadow

    reconstructed, ingest = replay_mapping(joint_state, chunk_size)
    batch = analyze_with_shadow(
        joint_state,
        algorithm_config,
        ml_config,
        activity_pipeline,
        session_id,
        evidence_type,
        pressure_kpa=pressure_kpa,
        discomfort=discomfort,
        cable_tension_n=cable_tension_n,
        lever_arm_m=lever_arm_m,
        user_continues=user_continues,
        angle_alerts_enabled=angle_alerts_enabled,
        mechanical_recommendations_enabled=mechanical_recommendations_enabled,
    )
    replayed = analyze_with_shadow(
        reconstructed,
        algorithm_config,
        ml_config,
        activity_pipeline,
        session_id,
        evidence_type,
        pressure_kpa=pressure_kpa,
        discomfort=discomfort,
        cable_tension_n=cable_tension_n,
        lever_arm_m=lever_arm_m,
        user_continues=user_continues,
        angle_alerts_enabled=angle_alerts_enabled,
        mechanical_recommendations_enabled=mechanical_recommendations_enabled,
    )
    batch_states = stream_exposure_states(
        joint_state,
        algorithm_config,
        len(np.asarray(joint_state["timestamp_ms"])),
        pressure_kpa,
        discomfort,
        user_continues,
        angle_alerts_enabled,
        mechanical_recommendations_enabled,
    )
    stream_states = stream_exposure_states(
        joint_state,
        algorithm_config,
        chunk_size,
        pressure_kpa,
        discomfort,
        user_continues,
        angle_alerts_enabled,
        mechanical_recommendations_enabled,
    )
    batch_fingerprint = canonical_fingerprint(batch)
    replay_fingerprint = canonical_fingerprint(replayed)
    ingest.update(
        {
            "deterministic_state_equal": canonical_fingerprint(batch_states) == canonical_fingerprint(stream_states),
            "final_analysis_equal": batch_fingerprint == replay_fingerprint,
            "batch_analysis_sha256": batch_fingerprint,
            "replay_analysis_sha256": replay_fingerprint,
        }
    )
    return batch, ingest