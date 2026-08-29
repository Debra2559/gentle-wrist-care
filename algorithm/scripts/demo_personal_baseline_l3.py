"""Demo: mock longitudinal data and exercise the personal-baseline L3 path.

Part A runs the deterministic baseline module directly on synthetic per-day
sessions to show when the single-subject (n-of-1) symptom association and the
exposure-tolerance estimate become evaluable.

Part B replays the SAME synthetic days through the real backend
``AnalysisService`` (joint-state CSV + mechanical CSV with ``discomfort_nrs``)
to show the ``personal_baseline`` block evolving in the public API result.

Nothing here is clinical: the algorithm is advisory, deterministic, uses no ML
for the baseline, and has no control authority.

Run:
    PYTHONPATH=src .venv/bin/python scripts/demo_personal_baseline_l3.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "demo" / ".mplcache"))
(PROJECT_ROOT / "outputs" / "demo" / ".mplcache").mkdir(parents=True, exist_ok=True)

from shewrist.backend import AnalysisService, BackendSettings  # noqa: E402
from shewrist.baseline import (  # noqa: E402
    build_personal_report,
    session_exposure_summary,
)
from shewrist.data import load_config  # noqa: E402

SEED = 7
N_DAYS = 14
SAMPLE_RATE_HZ = 10.0
DURATION_S = 90.0
PARTICIPANT_ID = "DEMO01"


def synthetic_session(amplitude_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One day of wrist angles whose exposure scales with ``amplitude_deg``."""

    n = int(DURATION_S * SAMPLE_RATE_HZ) + 1
    t = np.arange(n) / SAMPLE_RATE_HZ
    fe = amplitude_deg * np.sin(2.0 * np.pi * 0.5 * t)
    rud = 0.6 * amplitude_deg * np.sin(2.0 * np.pi * 0.35 * t)
    return t, fe, rud


def build_days(config: dict) -> dict:
    """Generate deterministic per-day amplitude, exposure, and (lagged) pain."""

    rng = np.random.default_rng(SEED)
    amplitudes = 22.0 + 7.0 * np.sin(np.arange(N_DAYS) / 2.0) + rng.uniform(-5.0, 9.0, N_DAYS)
    amplitudes = np.clip(amplitudes, 12.0, 45.0)

    exposures = []
    for amplitude in amplitudes:
        t, fe, rud = synthetic_session(amplitude)
        summary = session_exposure_summary(t, fe, rud, config)
        exposures.append(float(summary["metrics"]["dose_rate_deg_s_per_min"]))
    exposures = np.asarray(exposures, dtype=float)

    lo, hi = float(exposures.min()), float(exposures.max())
    slope = 7.0 / (hi - lo) if hi > lo else 0.0
    noise = np.random.default_rng(SEED + 1).normal(0.0, 0.4, N_DAYS)
    pains = np.empty(N_DAYS, dtype=float)
    for day in range(N_DAYS):
        driver = exposures[day - 1] if day > 0 else exposures[0]
        pains[day] = np.clip(1.0 + slope * (driver - lo) + noise[day], 0.0, 10.0)
    pains = np.round(pains, 1)

    return {"amplitudes": amplitudes, "exposures": exposures, "pains": pains}


def part_a_module_demo(config: dict, days: dict) -> dict:
    print("=" * 74)
    print("PART A — module-level L3 on synthetic longitudinal data")
    print("=" * 74)
    exposures = days["exposures"]
    pains = days["pains"]

    print(f"{'day':>3} {'exposure(dose_rate)':>20} {'pain_NRS':>9}")
    for day in range(N_DAYS):
        print(f"{day + 1:>3} {exposures[day]:>20.2f} {pains[day]:>9.1f}")

    print("\nL3 evaluability as days accumulate (lag=1 day):")
    print(f"{'days':>5} {'assoc_status':>14} {'pearson_r':>10} {'tol_status':>14}")
    last_summary = None
    for k in range(3, N_DAYS + 1):
        exposure_series = list(exposures[:k])
        pain_series = list(pains[:k])
        t, fe, rud = synthetic_session(days["amplitudes"][k - 1])
        last_summary = session_exposure_summary(t, fe, rud, config)
        report = build_personal_report(
            _dummy_baseline(config, exposures[:k]),
            last_summary,
            config,
            exposure_series,
            pain_series,
        )
        assoc = report["symptom_association"]
        tol = report["exposure_tolerance"]
        r = assoc.get("pearson_r")
        print(
            f"{k:>5} {assoc['status']:>14} {('%.2f' % r) if r is not None else '   -':>10} "
            f"{tol['status']:>14}"
        )

    print("\nFull L3 block at day", N_DAYS, ":")
    final = build_personal_report(
        _dummy_baseline(config, exposures),
        last_summary,
        config,
        list(exposures),
        list(pains),
    )
    print(json.dumps(final["symptom_association"], ensure_ascii=False, indent=2))
    print(json.dumps(final["exposure_tolerance"], ensure_ascii=False, indent=2))
    return final


def _dummy_baseline(config: dict, exposures: np.ndarray):
    from shewrist.baseline import PersonalBaseline

    tracked = list(config["personal_baseline"]["tracked_metrics"])
    metrics = {name: None for name in tracked}
    metrics["dose_rate_deg_s_per_min"] = float(np.mean(exposures))
    return PersonalBaseline(PARTICIPANT_ID, metrics, float(len(exposures)), len(exposures), "established", [])


def _joint_csv(amplitude_deg: float) -> bytes:
    t, fe, rud = synthetic_session(amplitude_deg)
    speed = np.sqrt(np.gradient(fe, t) ** 2 + np.gradient(rud, t) ** 2)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality"])
    for i, ti in enumerate(t):
        writer.writerow([ti * 1000.0, fe[i], rud[i], "", speed[i], "DEMO-CAL", 1.0])
    return out.getvalue().encode("utf-8")


def _mechanical_csv(pain_nrs: float) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["timestamp_ms", "discomfort_nrs", "user_continues"])
    for second in np.arange(0.0, DURATION_S + 1.0, 1.0):
        writer.writerow([second * 1000.0, pain_nrs, 1])
    return out.getvalue().encode("utf-8")


def _day_metadata(day: int) -> dict:
    return {
        "schema_version": "1.0",
        "session_id": f"{PARTICIPANT_ID}-d{day:02d}",
        "participant_id": PARTICIPANT_ID,
        "input_type": "joint_state",
        "evidence_type": "simulation",
        "timestamp_basis": "session_relative_ms",
        "options": {
            "enable_ml_shadow": True,
            "threshold_version": "engineering_v1",
            "explanation_provider": "local_template",
            "enable_external_api": False,
            "generate_charts": False,
            "personal_baseline_role": "auto",
        },
    }


def part_b_backend_demo(days: dict) -> dict:
    print("\n" + "=" * 74)
    print("PART B — end-to-end backend flow (one session per day)")
    print("=" * 74)
    temporary = tempfile.TemporaryDirectory()
    defaults = BackendSettings.default(PROJECT_ROOT)
    settings = BackendSettings(
        project_root=PROJECT_ROOT,
        output_root=Path(temporary.name) / "api",
        algorithm_config=defaults.algorithm_config,
        ml_config=defaults.ml_config,
        explanation_config=defaults.explanation_config,
        model_path=defaults.model_path,
    )
    service = AnalysisService(settings)

    print(
        f"{'day':>3} {'status':>12} {'sessions':>8} {'today_vs_base%':>14} "
        f"{'assoc':>14} {'r':>6} {'tol_exposure':>13} {'#sugg':>6}"
    )
    last_result = None
    for day in range(1, N_DAYS + 1):
        created = service.create_job(
            _day_metadata(day),
            _joint_csv(days["amplitudes"][day - 1]),
            "input.csv",
            _mechanical_csv(float(days["pains"][day - 1])),
            "mechanical.csv",
        )
        service.run_job(created["job_id"])
        result = service.get_result(f"{PARTICIPANT_ID}-d{day:02d}")
        pb = result["personal_baseline"]
        assoc = pb["symptom_association"]
        tol = pb["exposure_tolerance"]
        rel = pb["relative_exposure"].get("dose_rate_deg_s_per_min", {})
        pct = rel.get("pct_vs_baseline")
        r = assoc.get("pearson_r")
        tol_val = tol.get("tolerance_exposure")
        print(
            f"{day:>3} {str(pb['status']):>12} {int(pb['session_count']):>8} "
            f"{('%.1f' % pct) if pct is not None else '   -':>14} "
            f"{assoc['status']:>14} {('%.2f' % r) if r is not None else '  -':>6} "
            f"{('%.2f' % tol_val) if tol_val is not None else '   -':>13} "
            f"{len(pb['suggestions']):>6}"
        )
        last_result = result

    print("\nFinal personal_baseline block (day", N_DAYS, ") from the API result:")
    print(json.dumps(last_result["personal_baseline"], ensure_ascii=False, indent=2))
    temporary.cleanup()
    return last_result["personal_baseline"]


def make_plot(days: dict, final_tolerance: dict) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plot skipped: {exc}]")
        return None

    exposures = days["exposures"]
    pains = days["pains"]
    x = exposures[:-1]
    y = pains[1:]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(x, y, c="tab:blue", s=48, label="day exposure -> next-day pain")
    if x.size >= 2:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="tab:orange", lw=2, label="linear trend")
    tol = final_tolerance.get("tolerance_exposure")
    if tol is not None:
        ax.axvline(tol, color="tab:green", ls="--", lw=2, label=f"tolerance ~ {tol:.1f}")
    ax.set_xlabel("Daily exposure (dose_rate, deg*s/min)")
    ax.set_ylabel("Next-day pain (NRS 0-10)")
    ax.set_title("Personal n-of-1 exposure vs next-day pain (synthetic)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = PROJECT_ROOT / "outputs" / "demo" / "personal_baseline_l3.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output


def main() -> None:
    config = load_config(PROJECT_ROOT / "config" / "thresholds.yaml")
    days = build_days(config)
    part_a_module_demo(config, days)
    final_block = part_b_backend_demo(days)
    plot_path = make_plot(days, final_block["exposure_tolerance"])
    if plot_path is not None:
        print(f"\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()
