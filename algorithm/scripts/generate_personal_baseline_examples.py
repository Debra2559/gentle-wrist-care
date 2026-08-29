"""Generate a ready-to-POST personal-baseline example bundle for handoff.

Writes, under ``examples/personal_baseline/``:

* ``metadata_dNN.json`` / ``joint_state_dNN.csv`` / ``mechanical_dNN.csv`` for
  each mock day (exactly the three multipart parts of one analysis job).
* ``post_order.json`` listing the days and their files in submission order.
* ``sample_result_dNN.json`` — a real ``GET /api/v1/sessions/{id}`` response
  captured by running the mock days through the actual backend, so the
  software-side integrator sees the concrete ``personal_baseline`` payload.
* ``personal_baseline_evolution.json`` — the per-day personal_baseline blocks.

Deterministic (fixed seed). Re-run to regenerate.

Run:
    PYTHONPATH=src .venv/bin/python scripts/generate_personal_baseline_examples.py
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "examples" / "personal_baseline"

from shewrist.backend import AnalysisService, BackendSettings  # noqa: E402
from shewrist.baseline import session_exposure_summary  # noqa: E402
from shewrist.data import load_config  # noqa: E402

SEED = 7
N_DAYS = 10
SAMPLE_RATE_HZ = 10.0
DURATION_S = 90.0
PARTICIPANT_ID = "DEMO01"


def synthetic_session(amplitude_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(DURATION_S * SAMPLE_RATE_HZ) + 1
    t = np.arange(n) / SAMPLE_RATE_HZ
    fe = amplitude_deg * np.sin(2.0 * np.pi * 0.5 * t)
    rud = 0.6 * amplitude_deg * np.sin(2.0 * np.pi * 0.35 * t)
    return t, fe, rud


def build_days(config: dict) -> dict:
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
    return {"amplitudes": amplitudes, "pains": pains}


def joint_state_bytes(amplitude_deg: float) -> bytes:
    t, fe, rud = synthetic_session(amplitude_deg)
    speed = np.sqrt(np.gradient(fe, t) ** 2 + np.gradient(rud, t) ** 2)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["timestamp_ms", "theta_FE", "theta_RUD", "theta_thumb", "angular_velocity", "calibration_id", "quality"])
    for i, ti in enumerate(t):
        writer.writerow([round(ti * 1000.0, 3), round(float(fe[i]), 6), round(float(rud[i]), 6), "", round(float(speed[i]), 6), "DEMO-CAL", 1.0])
    return out.getvalue().encode("utf-8")


def mechanical_bytes(pain_nrs: float) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["timestamp_ms", "discomfort_nrs", "user_continues"])
    for second in np.arange(0.0, DURATION_S + 1.0, 1.0):
        writer.writerow([round(second * 1000.0, 3), pain_nrs, 1])
    return out.getvalue().encode("utf-8")


def day_metadata(day: int) -> dict:
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
            "language": "zh-CN",
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT_ROOT / "config" / "thresholds.yaml")
    days = build_days(config)

    post_order = []
    for day in range(1, N_DAYS + 1):
        metadata = day_metadata(day)
        meta_name = f"metadata_d{day:02d}.json"
        joint_name = f"joint_state_d{day:02d}.csv"
        mech_name = f"mechanical_d{day:02d}.csv"
        (OUTPUT_DIR / meta_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (OUTPUT_DIR / joint_name).write_bytes(joint_state_bytes(days["amplitudes"][day - 1]))
        (OUTPUT_DIR / mech_name).write_bytes(mechanical_bytes(float(days["pains"][day - 1])))
        post_order.append(
            {
                "day": day,
                "session_id": metadata["session_id"],
                "metadata": meta_name,
                "data_file": joint_name,
                "mechanical_file": mech_name,
            }
        )
    (OUTPUT_DIR / "post_order.json").write_text(
        json.dumps({"participant_id": PARTICIPANT_ID, "days": post_order}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Capture real backend responses so the integrator sees the exact payload.
    with tempfile.TemporaryDirectory() as tmp:
        defaults = BackendSettings.default(PROJECT_ROOT)
        service = AnalysisService(
            BackendSettings(
                project_root=PROJECT_ROOT,
                output_root=Path(tmp) / "api",
                algorithm_config=defaults.algorithm_config,
                ml_config=defaults.ml_config,
                explanation_config=defaults.explanation_config,
                model_path=defaults.model_path,
            )
        )
        evolution = []
        last_result = None
        for entry in post_order:
            metadata = json.loads((OUTPUT_DIR / entry["metadata"]).read_text(encoding="utf-8"))
            created = service.create_job(
                metadata,
                (OUTPUT_DIR / entry["data_file"]).read_bytes(),
                entry["data_file"],
                (OUTPUT_DIR / entry["mechanical_file"]).read_bytes(),
                entry["mechanical_file"],
            )
            service.run_job(created["job_id"])
            result = service.get_result(entry["session_id"])
            evolution.append({"day": entry["day"], "personal_baseline": result["personal_baseline"]})
            last_result = result

        (OUTPUT_DIR / f"sample_result_d{N_DAYS:02d}.json").write_text(
            json.dumps(last_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (OUTPUT_DIR / "personal_baseline_evolution.json").write_text(
            json.dumps({"participant_id": PARTICIPANT_ID, "days": evolution}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {N_DAYS} mock days + sample responses to {OUTPUT_DIR}")
    print("Files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
