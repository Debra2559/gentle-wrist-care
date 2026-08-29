"""Interpretable exposure zones and alert state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np


ZONE_ORDER = {"invalid": -1, "green": 0, "yellow": 1, "red": 2}


def _band(abs_angle: float, yellow: float, red: float) -> str:
    if not np.isfinite(abs_angle):
        return "invalid"
    if abs_angle > red:
        return "red"
    if abs_angle >= yellow:
        return "yellow"
    return "green"


def classify_zone(
    theta_fe_deg: float,
    theta_rud_deg: float,
    config: Mapping[str, object],
    pressure_kpa: float | None = None,
    discomfort: bool = False,
    quality_valid: bool = True,
) -> dict[str, object]:
    pressure_zone = "green"
    if discomfort:
        pressure_zone = "red"
    elif pressure_kpa is not None and np.isfinite(pressure_kpa):
        pressure_cfg = config["pressure_kpa"]
        pressure_zone = _band(abs(float(pressure_kpa)), float(pressure_cfg["yellow"]), float(pressure_cfg["red"]))
    if not quality_valid or not np.isfinite(theta_fe_deg) or not np.isfinite(theta_rud_deg):
        reason = "pressure_or_discomfort" if pressure_zone == "red" else "pressure_screening_yellow" if pressure_zone == "yellow" else "invalid_quality"
        return {
            "zone": pressure_zone if pressure_zone != "green" else "invalid",
            "angle_zone": "invalid",
            "pressure_zone": pressure_zone,
            "compound": False,
            "reason": reason,
        }
    angles = config["angle_degrees"]
    fe_cfg = angles["flexion_extension"]
    rud_cfg = angles["radial_ulnar"]
    fe_zone = _band(abs(theta_fe_deg), float(fe_cfg["yellow_abs"]), float(fe_cfg["red_abs"]))
    rud_zone = _band(abs(theta_rud_deg), float(rud_cfg["yellow_abs"]), float(rud_cfg["red_abs"]))
    compound_cfg = config["compound_posture"]
    compound = theta_fe_deg >= float(compound_cfg["extension_min"]) and theta_rud_deg >= float(compound_cfg["ulnar_min"])
    angle_zone = max((fe_zone, rud_zone), key=lambda value: ZONE_ORDER[value])
    if compound:
        angle_zone = "red"
    zone = max((angle_zone, pressure_zone), key=lambda value: ZONE_ORDER[value])
    reasons = []
    if abs(theta_fe_deg) > float(fe_cfg["red_abs"]):
        reasons.append("fe_red")
    if abs(theta_rud_deg) > float(rud_cfg["red_abs"]):
        reasons.append("rud_red")
    if compound:
        reasons.append("extension_ulnar_compound")
    if pressure_zone == "red":
        reasons.append("pressure_or_discomfort")
    elif pressure_zone == "yellow":
        reasons.append("pressure_screening_yellow")
    if not reasons and zone == "yellow":
        reasons.append("non_neutral")
    return {
        "zone": zone,
        "angle_zone": angle_zone,
        "pressure_zone": pressure_zone,
        "compound": compound,
        "reason": "+".join(reasons) if reasons else "neutral",
    }


@dataclass(frozen=True)
class ExposureSample:
    timestamp_s: float
    zone: str
    angle_zone: str
    pressure_zone: str
    compound: bool
    high_duration_s: float
    rolling_high_s: float
    alert: bool
    would_alert: bool
    alert_reason: str
    recommend_mechanical: bool
    safety_stop: bool


class ExposureEngine:
    """Stateful online-equivalent alert engine.

    Angle alerts require continuous red/compound exposure. Pressure above the
    screening line or reported discomfort bypasses the delay and requests an
    immediate release/stop. A mechanical recommendation is never an automatic
    tightening command.
    """

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = config
        duration = config["duration_seconds"]
        self.continuous_alert_s = float(duration["continuous_alert"])
        self.rolling_window_s = float(duration["rolling_window"])
        self.rolling_high_target_s = float(duration["rolling_high_exposure"])
        self.cooldown_s = float(duration["cooldown"])
        self.reset()

    def reset(self) -> None:
        self.last_timestamp: float | None = None
        self.high_duration_s = 0.0
        self.last_angle_alert_s = -np.inf
        self.last_safety_alert_s = -np.inf
        self.alert_count = 0
        self.was_angle_red = False
        self.mechanical_recommendation_active = False
        self.window: deque[tuple[float, float]] = deque()
        self.rolling_high_s = 0.0

    def update(
        self,
        timestamp_s: float,
        theta_fe_deg: float,
        theta_rud_deg: float,
        pressure_kpa: float | None = None,
        discomfort: bool = False,
        quality_valid: bool = True,
        user_continues: bool = True,
        angle_alerts_enabled: bool = True,
        mechanical_recommendations_enabled: bool = True,
    ) -> ExposureSample:
        timestamp = float(timestamp_s)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        dt = 0.0 if self.last_timestamp is None else timestamp - self.last_timestamp
        state = classify_zone(theta_fe_deg, theta_rud_deg, self.config, pressure_kpa, discomfort, quality_valid)
        angle_red = state["angle_zone"] == "red"
        high = state["angle_zone"] in {"yellow", "red"}
        if angle_red:
            self.high_duration_s += dt
        else:
            self.high_duration_s = 0.0
        high_dt = dt if high else 0.0
        self.window.append((timestamp, high_dt))
        self.rolling_high_s += high_dt
        cutoff = timestamp - self.rolling_window_s
        while self.window and self.window[0][0] < cutoff:
            _, expired = self.window.popleft()
            self.rolling_high_s -= expired
        safety_stop = state["pressure_zone"] == "red"
        alert = False
        would_alert = False
        alert_reason = ""
        if safety_stop:
            if timestamp - self.last_safety_alert_s >= 1.0:
                alert = True
                alert_reason = "release_and_stop_calibrated_pressure_or_safety_symptom"
                self.last_safety_alert_s = timestamp
        else:
            became_eligible = angle_red and self.high_duration_s >= self.continuous_alert_s
            cooldown_ready = timestamp - self.last_angle_alert_s >= self.cooldown_s
            if became_eligible and cooldown_ready:
                would_alert = True
                alert_reason = str(state["reason"])
                if angle_alerts_enabled:
                    alert = True
                self.last_angle_alert_s = timestamp
                self.alert_count += 1
        mechanical_eligible = bool(
            mechanical_recommendations_enabled
            and user_continues
            and not safety_stop
            and self.alert_count >= 1
            and self.rolling_high_s >= self.rolling_high_target_s
        )
        recommend_mechanical = mechanical_eligible and not self.mechanical_recommendation_active
        self.mechanical_recommendation_active = mechanical_eligible
        self.last_timestamp = timestamp
        self.was_angle_red = angle_red
        return ExposureSample(
            timestamp_s=timestamp,
            zone=str(state["zone"]),
            angle_zone=str(state["angle_zone"]),
            pressure_zone=str(state["pressure_zone"]),
            compound=bool(state["compound"]),
            high_duration_s=float(self.high_duration_s),
            rolling_high_s=float(max(0.0, self.rolling_high_s)),
            alert=alert,
            would_alert=would_alert,
            alert_reason=alert_reason,
            recommend_mechanical=recommend_mechanical,
            safety_stop=safety_stop,
        )

    def process(
        self,
        timestamp_s: np.ndarray,
        theta_fe_deg: np.ndarray,
        theta_rud_deg: np.ndarray,
        pressure_kpa: np.ndarray | None = None,
        discomfort: np.ndarray | None = None,
        quality: np.ndarray | None = None,
        quality_threshold: float = 0.2,
        user_continues: np.ndarray | None = None,
        angle_alerts_enabled: bool = True,
        mechanical_recommendations_enabled: bool = True,
    ) -> list[ExposureSample]:
        t = np.asarray(timestamp_s, dtype=float)
        fe = np.asarray(theta_fe_deg, dtype=float)
        rud = np.asarray(theta_rud_deg, dtype=float)
        if not (len(t) == len(fe) == len(rud)):
            raise ValueError("time and angle arrays must have equal length")
        pressure = np.full(len(t), np.nan) if pressure_kpa is None else np.asarray(pressure_kpa, dtype=float)
        discomfort_values = np.zeros(len(t), dtype=bool) if discomfort is None else np.asarray(discomfort, dtype=bool)
        continues_values = np.ones(len(t), dtype=bool) if user_continues is None else np.asarray(user_continues, dtype=bool)
        quality_values = np.ones(len(t)) if quality is None else np.asarray(quality, dtype=float)
        if not (len(pressure) == len(discomfort_values) == len(continues_values) == len(quality_values) == len(t)):
            raise ValueError("optional sample arrays must match timestamps")
        self.reset()
        return [
            self.update(
                t[i],
                fe[i],
                rud[i],
                None if not np.isfinite(pressure[i]) else float(pressure[i]),
                bool(discomfort_values[i]),
                bool(np.isfinite(quality_values[i]) and quality_values[i] >= quality_threshold),
                bool(continues_values[i]),
                angle_alerts_enabled,
                mechanical_recommendations_enabled,
            )
            for i in range(len(t))
        ]