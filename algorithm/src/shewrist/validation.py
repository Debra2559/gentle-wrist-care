"""Validation metrics and within-participant condition comparisons."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def estimate_lag_samples(reference: np.ndarray, estimate: np.ndarray, max_lag: int | None = None) -> int:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    n = min(len(reference), len(estimate))
    if n < 3:
        raise ValueError("signals need at least three samples")
    x = reference[:n] - np.mean(reference[:n])
    y = estimate[:n] - np.mean(estimate[:n])
    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-n + 1, n)
    if max_lag is not None:
        keep = np.abs(lags) <= int(max_lag)
        corr, lags = corr[keep], lags[keep]
    return int(lags[np.argmax(corr)])


def align_by_lag(reference: np.ndarray, estimate: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if lag >= 0:
        reference = reference[lag:]
    else:
        estimate = estimate[-lag:]
    n = min(len(reference), len(estimate))
    return reference[:n], estimate[:n]


def angle_error_metrics(
    reference_deg: np.ndarray,
    estimate_deg: np.ndarray,
    sample_rate_hz: float | None = None,
    align: bool = False,
    max_lag_s: float = 2.0,
) -> dict[str, float | int | None]:
    reference = np.asarray(reference_deg, dtype=float)
    estimate = np.asarray(estimate_deg, dtype=float)
    lag = 0
    if align:
        if sample_rate_hz is None:
            raise ValueError("sample_rate_hz is required when align=True")
        lag = estimate_lag_samples(reference, estimate, int(round(max_lag_s * sample_rate_hz)))
        reference, estimate = align_by_lag(reference, estimate, lag)
    else:
        n = min(len(reference), len(estimate))
        reference, estimate = reference[:n], estimate[:n]
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if np.count_nonzero(valid) < 3:
        raise ValueError("fewer than three valid paired values")
    reference, estimate = reference[valid], estimate[valid]
    error = estimate - reference
    correlation = float(np.corrcoef(reference, estimate)[0, 1]) if np.std(reference) > 0 and np.std(estimate) > 0 else None
    return {
        "n": int(len(error)),
        "mae_deg": float(np.mean(np.abs(error))),
        "rmse_deg": float(np.sqrt(np.mean(error * error))),
        "bias_deg": float(np.mean(error)),
        "p95_abs_error_deg": float(np.percentile(np.abs(error), 95)),
        "rom_reference_deg": float(np.ptp(reference)),
        "rom_estimate_deg": float(np.ptp(estimate)),
        "rom_error_deg": float(np.ptp(estimate) - np.ptp(reference)),
        "correlation": correlation,
        "lag_samples": int(lag),
        "lag_ms": None if sample_rate_hz is None else 1000.0 * lag / sample_rate_hz,
    }


def bootstrap_mean_ci(values: np.ndarray, confidence: float = 0.95, iterations: int = 10000, seed: int = 20260826) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("no finite values")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    batch = 1000
    for start in range(0, iterations, batch):
        size = min(batch, iterations - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def paired_condition_comparison(
    records: Iterable[Mapping[str, object]],
    metric: str,
    baseline: str = "A",
    comparison: str = "C",
) -> dict[str, object]:
    by_participant: dict[str, dict[str, float]] = {}
    for row in records:
        participant = str(row["participant_id"])
        condition = str(row["condition_id"])
        value = row.get(metric)
        if value is not None and np.isfinite(float(value)):
            by_participant.setdefault(participant, {})[condition] = float(value)
    pairs = [
        (participant, values[baseline], values[comparison])
        for participant, values in sorted(by_participant.items())
        if baseline in values and comparison in values
    ]
    if not pairs:
        raise ValueError(f"no participant has both {baseline} and {comparison} for {metric}")
    base_values = np.array([row[1] for row in pairs])
    comparison_values = np.array([row[2] for row in pairs])
    differences = comparison_values - base_values
    reductions = np.where(np.abs(base_values) > 1e-12, 100.0 * (base_values - comparison_values) / np.abs(base_values), np.nan)
    diff_ci = bootstrap_mean_ci(differences)
    finite_reductions = reductions[np.isfinite(reductions)]
    reduction_ci = bootstrap_mean_ci(finite_reductions) if len(finite_reductions) else (None, None)
    return {
        "metric": metric,
        "baseline": baseline,
        "comparison": comparison,
        "n_pairs": len(pairs),
        "participants": [row[0] for row in pairs],
        "baseline_mean": float(np.mean(base_values)),
        "comparison_mean": float(np.mean(comparison_values)),
        "paired_difference_mean": float(np.mean(differences)),
        "paired_difference_ci95": list(diff_ci),
        "reduction_pct_mean": float(np.nanmean(reductions)) if len(finite_reductions) else None,
        "reduction_pct_ci95": list(reduction_ci),
    }


def evaluate_go_no_go(
    comparison_ac: Mapping[str, object] | None,
    comparison_bc: Mapping[str, object] | None,
    max_pressure_kpa: float | None,
    task_performance_drop_pct: float | None,
    comfort_score: float | None,
    config: Mapping[str, object],
    pressure_discomfort: bool | None = None,
    effective_alert_acceptance_pct: float | None = None,
) -> dict[str, object]:
    criteria = config["acceptance"]
    checks: dict[str, bool | None] = {
        "condition_c_reduces_exposure_vs_a": (
            None
            if not comparison_ac or comparison_ac.get("reduction_pct_mean") is None
            else bool(float(comparison_ac["reduction_pct_mean"]) >= float(criteria["condition_c_exposure_reduction_pct_min"]))
        ),
        "condition_c_improves_vs_b": (
            None
            if not comparison_bc or comparison_bc.get("paired_difference_mean") is None
            else bool(float(comparison_bc["paired_difference_mean"]) < 0.0)
        ),
        "pressure_screening_pass": (
            None
            if max_pressure_kpa is None
            else bool(max_pressure_kpa <= float(config["pressure_kpa"]["red"]))
        ),
        "no_pressure_discomfort": None if pressure_discomfort is None else bool(pressure_discomfort is False),
        "task_performance_pass": (
            None
            if task_performance_drop_pct is None
            else bool(task_performance_drop_pct < float(criteria["task_performance_drop_pct_max"]))
        ),
        "comfort_pass": None if comfort_score is None else bool(comfort_score >= float(criteria["comfort_score_min"])),
        "effective_alert_acceptance_pass": (
            None
            if effective_alert_acceptance_pct is None
            else bool(effective_alert_acceptance_pct >= float(criteria["effective_alert_acceptance_pct_min"]))
        ),
    }
    failed = sorted(key for key, value in checks.items() if value is False)
    not_evaluable = sorted(key for key, value in checks.items() if value is None)
    decision = "NO-GO" if failed else "NOT-EVALUABLE" if not_evaluable else "GO"
    return {
        "decision": decision,
        "checks": checks,
        "failed_checks": failed,
        "not_evaluable_checks": not_evaluable,
    }