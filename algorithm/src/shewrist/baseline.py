"""Deterministic per-person exposure baseline and single-subject symptom association.

Scope and boundaries (read before use):

* This module is **advisory and descriptive only**. It quantifies how a single
  person's own wrist exposure changes over time relative to that person's own
  history, and — when enough paired self-reported pain data exist — an
  observational (n-of-1) association between exposure and pain.
* It uses **no machine learning** and reads only deterministic angle exposure
  (from the two IMUs) plus optional self-reported pain scores.
* It has **no control authority**: it never raises alerts, never changes safety
  thresholds, and never drives any mechanical action (the hardware has none).
* It does **not** estimate disease risk, diagnose tenosynovitis, prescribe
  treatment, or claim clinical validity. The "tolerance" estimate is an
  observational personal statistic, not a medical threshold.

Layering:

* L1 descriptive baseline: adaptive percentiles of a person's own exposure
  ("today vs your usual").
* L2 goal line: a user/clinician-chosen behavioural target relative to baseline.
* L3 symptom linkage: n-of-1 correlation and an observational exposure-tolerance
  estimate, both gated to ``not_evaluable`` until enough paired days exist.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .metrics import exposure_metrics, sample_durations

SCHEMA_VERSION = "1.0"

ASSOCIATION_INTERPRETATION = (
    "Single-subject (n-of-1) association between this person's own exposure and "
    "their self-reported pain. It is descriptive, not causal, and is not a "
    "disease-risk or diagnostic estimate."
)
TOLERANCE_INTERPRETATION = (
    "Observational personal exposure level below which this person's own pain "
    "reports tended not to be elevated. It is an engineering statistic for "
    "reflection, not a validated clinical or safety threshold."
)


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


# ---------------------------------------------------------------------------
# Session exposure summary
# ---------------------------------------------------------------------------


def session_exposure_summary(
    timestamp_s: np.ndarray,
    theta_fe_deg: np.ndarray,
    theta_rud_deg: np.ndarray,
    config: Mapping[str, object],
    quality: np.ndarray | None = None,
) -> dict[str, object]:
    """Summarise one session into the exposure metrics tracked by the baseline.

    Returns valid-minutes, valid-sample percentage, and a flat ``metrics`` dict
    keyed by the names in ``config['personal_baseline']['tracked_metrics']``.
    """

    t = np.asarray(timestamp_s, dtype=float)
    fe = np.asarray(theta_fe_deg, dtype=float)
    rud = np.asarray(theta_rud_deg, dtype=float)
    if not (len(t) == len(fe) == len(rud)):
        raise ValueError("timestamps and angles must have equal length")

    pae = config["personal_baseline"]  # type: ignore[index]
    percentiles = [float(p) for p in pae["percentiles"]]  # type: ignore[index]

    dt = sample_durations(t)
    valid = np.isfinite(fe) & np.isfinite(rud)
    if quality is not None:
        q = np.asarray(quality, dtype=float)
        if len(q) != len(t):
            raise ValueError("quality must match timestamps")
        valid &= q >= 0.2

    valid_minutes = float(np.sum(dt[valid])) / 60.0
    base_metrics = exposure_metrics(t, fe, rud, config, quality=quality)

    metrics: dict[str, float | None] = {}
    if np.any(valid):
        abs_fe = np.abs(fe[valid])
        abs_rud = np.abs(rud[valid])
        for p in percentiles:
            metrics[f"abs_fe_deg_p{int(p)}"] = float(np.percentile(abs_fe, p))
            metrics[f"abs_rud_deg_p{int(p)}"] = float(np.percentile(abs_rud, p))
    else:
        for p in percentiles:
            metrics[f"abs_fe_deg_p{int(p)}"] = None
            metrics[f"abs_rud_deg_p{int(p)}"] = None

    dose_total = _finite_or_none(base_metrics.get("D_total_deg_s"))
    metrics["dose_rate_deg_s_per_min"] = (
        dose_total / valid_minutes if dose_total is not None and valid_minutes > 0 else None
    )
    metrics["P_high_pct"] = _finite_or_none(base_metrics.get("P_high_pct"))

    return {
        "valid_minutes": valid_minutes,
        "valid_sample_pct": float(base_metrics.get("valid_sample_pct", 0.0) or 0.0),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Adaptive personal baseline (L1)
# ---------------------------------------------------------------------------


@dataclass
class PersonalBaseline:
    """A single person's adaptive exposure reference.

    ``metrics`` holds one running estimate per tracked metric. ``status`` is one
    of ``provisional`` (not enough observed minutes yet), ``established``, or
    ``rejected`` (enrollment session failed quality gates).
    """

    participant_id: str
    metrics: dict[str, float | None]
    observed_minutes: float
    session_count: int
    status: str
    reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonalBaseline":
        return cls(
            participant_id=str(payload["participant_id"]),
            metrics=dict(payload.get("metrics", {})),  # type: ignore[arg-type]
            observed_minutes=float(payload.get("observed_minutes", 0.0)),  # type: ignore[arg-type]
            session_count=int(payload.get("session_count", 0)),  # type: ignore[arg-type]
            status=str(payload.get("status", "provisional")),
            reasons=list(payload.get("reasons", [])),  # type: ignore[arg-type]
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            updated_at=payload.get("updated_at"),  # type: ignore[arg-type]
        )


def _tracked_metrics(config: Mapping[str, object]) -> list[str]:
    return [str(name) for name in config["personal_baseline"]["tracked_metrics"]]  # type: ignore[index]


def init_personal_baseline(
    participant_id: str,
    summary: Mapping[str, object],
    config: Mapping[str, object],
    updated_at: str | None = None,
) -> PersonalBaseline:
    """Create a baseline from an enrollment session (<=5 min of relaxed work)."""

    pae = config["personal_baseline"]  # type: ignore[index]
    enrollment = pae["enrollment"]  # type: ignore[index]
    adaptive = pae["adaptive"]  # type: ignore[index]

    valid_minutes = float(summary["valid_minutes"])  # type: ignore[arg-type]
    valid_pct = float(summary["valid_sample_pct"])  # type: ignore[arg-type]
    reasons: list[str] = []
    if valid_minutes < float(enrollment["min_valid_minutes"]):  # type: ignore[index]
        reasons.append("insufficient_valid_minutes")
    if valid_minutes > float(enrollment["max_session_minutes"]):  # type: ignore[index]
        reasons.append("enrollment_session_exceeds_max_minutes")
    if valid_pct < float(enrollment["min_valid_sample_pct"]):  # type: ignore[index]
        reasons.append("valid_sample_pct_below_minimum")

    tracked = _tracked_metrics(config)
    session_metrics = summary["metrics"]  # type: ignore[index]
    metrics = {name: _finite_or_none(session_metrics.get(name)) for name in tracked}  # type: ignore[union-attr]

    if reasons:
        return PersonalBaseline(
            participant_id=participant_id,
            metrics={name: None for name in tracked},
            observed_minutes=0.0,
            session_count=0,
            status="rejected",
            reasons=reasons,
            updated_at=updated_at,
        )

    established = valid_minutes >= float(adaptive["established_minutes_min"])  # type: ignore[index]
    return PersonalBaseline(
        participant_id=participant_id,
        metrics=metrics,
        observed_minutes=valid_minutes,
        session_count=1,
        status="established" if established else "provisional",
        reasons=[],
        updated_at=updated_at,
    )


def update_personal_baseline(
    baseline: PersonalBaseline,
    summary: Mapping[str, object],
    config: Mapping[str, object],
    updated_at: str | None = None,
) -> PersonalBaseline:
    """Refine a baseline with a new session using an EWMA ("keep measuring, keep correcting")."""

    pae = config["personal_baseline"]  # type: ignore[index]
    adaptive = pae["adaptive"]  # type: ignore[index]
    alpha = float(adaptive["ewma_alpha"])  # type: ignore[index]

    session_metrics = summary["metrics"]  # type: ignore[index]
    new_metrics = dict(baseline.metrics)
    for name in baseline.metrics:
        value = _finite_or_none(session_metrics.get(name))  # type: ignore[union-attr]
        if value is None:
            continue
        previous = new_metrics.get(name)
        new_metrics[name] = value if previous is None else (1.0 - alpha) * previous + alpha * value

    observed_minutes = baseline.observed_minutes + float(summary["valid_minutes"])  # type: ignore[arg-type]
    established = observed_minutes >= float(adaptive["established_minutes_min"])  # type: ignore[index]
    return PersonalBaseline(
        participant_id=baseline.participant_id,
        metrics=new_metrics,
        observed_minutes=observed_minutes,
        session_count=baseline.session_count + 1,
        status="established" if established else "provisional",
        reasons=[],
        updated_at=updated_at,
    )


# ---------------------------------------------------------------------------
# Relative exposure (L1) and goal line (L2)
# ---------------------------------------------------------------------------


def relative_exposure(
    today_summary: Mapping[str, object],
    baseline: PersonalBaseline,
) -> dict[str, dict[str, float | None]]:
    """"Today vs your usual" for each tracked metric."""

    today_metrics = today_summary["metrics"]  # type: ignore[index]
    out: dict[str, dict[str, float | None]] = {}
    for name, base in baseline.metrics.items():
        today = _finite_or_none(today_metrics.get(name))  # type: ignore[union-attr]
        base_value = _finite_or_none(base)
        if today is None or base_value is None or base_value == 0.0:
            out[name] = {"today": today, "baseline": base_value, "ratio": None, "pct_vs_baseline": None}
        else:
            out[name] = {
                "today": today,
                "baseline": base_value,
                "ratio": today / base_value,
                "pct_vs_baseline": 100.0 * (today - base_value) / base_value,
            }
    return out


def goal_line(baseline: PersonalBaseline, target_reduction_pct: float) -> dict[str, float | None]:
    """Behavioural target = baseline reduced by ``target_reduction_pct`` percent."""

    factor = 1.0 - float(target_reduction_pct) / 100.0
    return {
        name: (None if _finite_or_none(base) is None else float(base) * factor)
        for name, base in baseline.metrics.items()
    }


# ---------------------------------------------------------------------------
# n-of-1 symptom association and exposure tolerance (L3)
# ---------------------------------------------------------------------------


def _pairwise(
    exposure_values: Sequence[float],
    pain_values: Sequence[float],
    lag_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    exposure = np.asarray(exposure_values, dtype=float)
    pain = np.asarray(pain_values, dtype=float)
    if len(exposure) != len(pain):
        raise ValueError("exposure and pain series must have equal length")
    if lag_days > 0:
        x = exposure[:-lag_days]
        y = pain[lag_days:]
    elif lag_days < 0:
        shift = -lag_days
        x = exposure[shift:]
        y = pain[:-shift]
    else:
        x = exposure
        y = pain
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    n = len(x)
    samples: list[float] = []
    for _ in range(int(iterations)):
        idx = rng.integers(0, n, n)
        r = _pearson(x[idx], y[idx])
        if r is not None:
            samples.append(r)
    if not samples:
        return None
    lo, hi = np.quantile(samples, [(1.0 - confidence) / 2.0, 1.0 - (1.0 - confidence) / 2.0])
    return [float(lo), float(hi)]


def symptom_exposure_association(
    exposure_values: Sequence[float],
    pain_values: Sequence[float],
    config: Mapping[str, object],
    lag_days: int | None = None,
) -> dict[str, object]:
    """Single-subject association between daily exposure and (lagged) pain."""

    symptom = config["personal_baseline"]["symptom"]  # type: ignore[index]
    lag = int(symptom["lag_days"]) if lag_days is None else int(lag_days)  # type: ignore[index]
    min_pairs = int(symptom["min_paired_days"])  # type: ignore[index]
    iterations = int(symptom["bootstrap_iterations"])  # type: ignore[index]
    seed = int(symptom["seed"])  # type: ignore[index]

    x, y = _pairwise(exposure_values, pain_values, lag)
    n = int(len(x))
    reasons: list[str] = []
    if n < min_pairs:
        reasons.append("insufficient_paired_days")
    if n >= 2 and (np.std(x) == 0.0 or np.std(y) == 0.0):
        reasons.append("no_variation_in_series")

    if reasons:
        return {
            "status": "not_evaluable",
            "reasons": reasons,
            "n_pairs": n,
            "lag_days": lag,
            "pearson_r": None,
            "spearman_r": None,
            "pearson_r_ci95": None,
            "interpretation": ASSOCIATION_INTERPRETATION,
        }

    pearson = _pearson(x, y)
    spearman = _pearson(_average_ranks(x), _average_ranks(y))
    ci = _bootstrap_corr_ci(x, y, iterations, seed)
    return {
        "status": "evaluable",
        "reasons": [],
        "n_pairs": n,
        "lag_days": lag,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "pearson_r_ci95": ci,
        "interpretation": ASSOCIATION_INTERPRETATION,
    }


def estimate_exposure_tolerance(
    exposure_values: Sequence[float],
    pain_values: Sequence[float],
    config: Mapping[str, object],
    lag_days: int | None = None,
) -> dict[str, object]:
    """Observational personal exposure "tolerance" (advisory, not clinical).

    Splits paired days into elevated vs non-elevated pain (relative to the
    person's own median pain) and reports the exposure percentile on
    non-elevated days. Returns ``not_evaluable`` unless both groups have enough
    days.
    """

    symptom = config["personal_baseline"]["symptom"]  # type: ignore[index]
    lag = int(symptom["lag_days"]) if lag_days is None else int(lag_days)  # type: ignore[index]
    min_pairs = int(symptom["min_paired_days"])  # type: ignore[index]
    min_group = int(symptom["min_group_days"])  # type: ignore[index]
    percentile = float(symptom["tolerance_percentile"])  # type: ignore[index]
    primary_metric = str(symptom["primary_metric"])  # type: ignore[index]

    x, y = _pairwise(exposure_values, pain_values, lag)
    n = int(len(x))
    reasons: list[str] = []
    if n < min_pairs:
        reasons.append("insufficient_paired_days")

    median_pain = float(np.median(y)) if n > 0 else None
    low = x[y <= median_pain] if median_pain is not None else np.array([])
    elevated = x[y > median_pain] if median_pain is not None else np.array([])
    if len(low) < min_group or len(elevated) < min_group:
        reasons.append("insufficient_group_days")

    if reasons:
        return {
            "status": "not_evaluable",
            "reasons": reasons,
            "primary_metric": primary_metric,
            "n_pairs": n,
            "lag_days": lag,
            "tolerance_exposure": None,
            "elevated_median_exposure": None,
            "non_elevated_day_count": int(len(low)),
            "elevated_day_count": int(len(elevated)),
            "interpretation": TOLERANCE_INTERPRETATION,
        }

    return {
        "status": "evaluable",
        "reasons": [],
        "primary_metric": primary_metric,
        "n_pairs": n,
        "lag_days": lag,
        "tolerance_exposure": float(np.percentile(low, percentile)),
        "elevated_median_exposure": float(np.median(elevated)),
        "non_elevated_day_count": int(len(low)),
        "elevated_day_count": int(len(elevated)),
        "interpretation": TOLERANCE_INTERPRETATION,
    }


# ---------------------------------------------------------------------------
# Advisory suggestions and combined report
# ---------------------------------------------------------------------------


def _suggestion(code: str, message: str) -> dict[str, object]:
    return {
        "code": code,
        "message": message,
        "control_effect": "none",
        "requires_human_action": True,
    }


def advisory_suggestions(
    relative: Mapping[str, Mapping[str, float | None]],
    association: Mapping[str, object],
    tolerance: Mapping[str, object],
    primary_metric: str,
    config: Mapping[str, object],
    high_exposure_pct: float = 25.0,
) -> list[dict[str, object]]:
    """Behavioural, non-clinical suggestions only. No control, no diagnosis."""

    suggestions: list[dict[str, object]] = []

    primary = relative.get(primary_metric, {})
    pct = primary.get("pct_vs_baseline")
    if pct is not None and pct >= high_exposure_pct:
        suggestions.append(
            _suggestion(
                "exposure_above_personal_baseline",
                f"Today's wrist exposure is about {pct:.0f}% above your personal baseline. "
                "Consider more micro-breaks or reducing continuous use.",
            )
        )

    if tolerance.get("status") == "evaluable":
        tol = tolerance.get("tolerance_exposure")
        today = primary.get("today")
        if tol is not None and today is not None and today > tol:
            suggestions.append(
                _suggestion(
                    "exposure_above_personal_tolerance",
                    "Today's exposure is above your observed personal comfort level. "
                    "This is a personal statistic, not a medical limit; consider resting the wrist.",
                )
            )

    r = association.get("pearson_r")
    if association.get("status") == "evaluable" and r is not None and r >= 0.4:
        suggestions.append(
            _suggestion(
                "exposure_pain_association_observed",
                "Your higher-exposure days tend to be followed by higher self-reported pain. "
                "If pain persists or worsens, consider a clinical follow-up. This tool does not diagnose.",
            )
        )

    return suggestions


def build_personal_report(
    baseline: PersonalBaseline,
    today_summary: Mapping[str, object],
    config: Mapping[str, object],
    exposure_series: Sequence[float] | None = None,
    pain_series: Sequence[float] | None = None,
    target_reduction_pct: float | None = None,
) -> dict[str, object]:
    """Combine L1 (relative), L2 (goal), and L3 (symptom) into one advisory report."""

    pae = config["personal_baseline"]  # type: ignore[index]
    symptom = pae["symptom"]  # type: ignore[index]
    primary_metric = str(symptom["primary_metric"])  # type: ignore[index]
    if target_reduction_pct is None:
        target_reduction_pct = float(pae["goal"]["default_target_reduction_pct"])  # type: ignore[index]

    relative = relative_exposure(today_summary, baseline)
    goal = goal_line(baseline, target_reduction_pct)

    if exposure_series is not None and pain_series is not None:
        association = symptom_exposure_association(exposure_series, pain_series, config)
        tolerance = estimate_exposure_tolerance(exposure_series, pain_series, config)
    else:
        association = {"status": "not_evaluable", "reasons": ["no_symptom_series_provided"]}
        tolerance = {"status": "not_evaluable", "reasons": ["no_symptom_series_provided"]}

    suggestions = advisory_suggestions(relative, association, tolerance, primary_metric, config)

    return {
        "schema_version": SCHEMA_VERSION,
        "participant_id": baseline.participant_id,
        "baseline_status": baseline.status,
        "baseline": baseline.to_dict(),
        "relative_exposure": relative,
        "goal_line": {"target_reduction_pct": target_reduction_pct, "targets": goal},
        "symptom_association": association,
        "exposure_tolerance": tolerance,
        "suggestions": suggestions,
        "control_effect": "none",
        "evidence_limits": {
            "control_authority": "none",
            "ml_used": False,
            "claims": "advisory personal exposure tracking and single-subject symptom association",
            "not_claimed": [
                "disease risk",
                "diagnosis of tenosynovitis",
                "clinical efficacy",
                "safety guarantee",
                "tissue strain",
            ],
            "note": str(pae["evidence"]["note"]),  # type: ignore[index]
        },
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_personal_baseline(path: str | Path, baseline: PersonalBaseline) -> None:
    """Write a baseline profile as UTF-8 JSON (creates parent directories)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(baseline.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_personal_baseline(path: str | Path) -> PersonalBaseline:
    """Load a baseline profile previously written by :func:`save_personal_baseline`."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PersonalBaseline.from_dict(payload)
