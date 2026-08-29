"""Dual-IMU wrist kinematics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .calibration import (
    FunctionalAxes,
    estimate_functional_axes,
    interval_mask,
    neutral_zero,
    project_angles,
    relative_quaternion,
)
from .fusion import OrientationResult, estimate_orientation
from .quaternion import conjugate, multiply, normalize


@dataclass(frozen=True)
class WristKinematics:
    timestamp_s: np.ndarray
    forearm_quaternion: np.ndarray
    hand_quaternion: np.ndarray
    relative_quaternion: np.ndarray
    theta_fe_deg: np.ndarray
    theta_rud_deg: np.ndarray
    theta_ps_deg: np.ndarray
    angular_velocity_deg_s: np.ndarray
    quality: np.ndarray
    axes: FunctionalAxes
    neutral_quaternion: np.ndarray
    forearm_gyro_bias_rad_s: np.ndarray
    hand_gyro_bias_rad_s: np.ndarray
    algorithm: str


def moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window_samples <= 1:
        return values.copy()
    window_samples = min(int(window_samples), len(values))
    left = window_samples // 2
    right = window_samples - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.convolve(padded, kernel, mode="valid")


def _orientation(
    timestamp_s: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    mag: np.ndarray | None,
    bias: np.ndarray,
    algorithm: str,
    fusion_config: Mapping[str, object],
    quality_config: Mapping[str, object],
    initialize_from_accel: bool = False,
) -> OrientationResult:
    return estimate_orientation(
        timestamp_s,
        accel,
        gyro,
        mag,
        algorithm=algorithm,
        beta=float(fusion_config.get("madgwick_beta", 0.08)),
        kp=float(fusion_config.get("mahony_kp", 0.5)),
        ki=float(fusion_config.get("mahony_ki", 0.0)),
        gyro_bias=bias,
        gravity=float(quality_config.get("gravity_mps2", 9.80665)),
        accel_tolerance=float(quality_config.get("acceleration_norm_tolerance_mps2", 3.0)),
        mag_bounds=(
            float(quality_config.get("magnetic_field_min_uT", 15.0)),
            float(quality_config.get("magnetic_field_max_uT", 80.0)),
        ),
        initialize_from_accel=initialize_from_accel,
    )


def compute_wrist_kinematics(
    timestamp_s: np.ndarray,
    forearm_accel_mps2: np.ndarray,
    forearm_gyro_rad_s: np.ndarray,
    hand_accel_mps2: np.ndarray,
    hand_gyro_rad_s: np.ndarray,
    neutral_interval_s: tuple[float, float],
    functional_annotations: Iterable[Mapping[str, object]],
    forearm_mag_uT: np.ndarray | None = None,
    hand_mag_uT: np.ndarray | None = None,
    algorithm: str = "madgwick",
    fusion_config: Mapping[str, object] | None = None,
    quality_config: Mapping[str, object] | None = None,
    smoothing_seconds: float = 0.10,
    neutral_sample_mask: np.ndarray | None = None,
    initialize_from_accel: bool = False,
) -> WristKinematics:
    """Estimate wrist angles from synchronized forearm and hand MARG streams.

    The first output axis is positive for extension and the second is positive for
    ulnar deviation. Functional labels are used only to orient those axes; they
    are not angle ground truth.
    """
    t = np.asarray(timestamp_s, dtype=float)
    if len(t) < 3:
        raise ValueError("at least three samples are required")
    fusion_config = {} if fusion_config is None else dict(fusion_config)
    quality_config = {} if quality_config is None else dict(quality_config)
    neutral_start, neutral_end = neutral_interval_s
    neutral_interval = interval_mask(t, neutral_start, neutral_end)
    if np.count_nonzero(neutral_interval) < 3:
        raise ValueError("neutral interval does not overlap enough samples")
    neutral = neutral_interval if neutral_sample_mask is None else np.asarray(neutral_sample_mask, dtype=bool)
    if len(neutral) != len(t) or np.count_nonzero(neutral) < 3 or np.any(neutral & ~neutral_interval):
        raise ValueError("neutral_sample_mask must select at least three samples inside the neutral interval")
    forearm_bias = np.median(np.asarray(forearm_gyro_rad_s, dtype=float)[neutral], axis=0)
    hand_bias = np.median(np.asarray(hand_gyro_rad_s, dtype=float)[neutral], axis=0)
    forearm_result = _orientation(
        t,
        forearm_accel_mps2,
        forearm_gyro_rad_s,
        forearm_mag_uT,
        forearm_bias,
        algorithm,
        fusion_config,
        quality_config,
        initialize_from_accel,
    )
    hand_result = _orientation(
        t,
        hand_accel_mps2,
        hand_gyro_rad_s,
        hand_mag_uT,
        hand_bias,
        algorithm,
        fusion_config,
        quality_config,
        initialize_from_accel,
    )
    relative = relative_quaternion(forearm_result.quaternion, hand_result.quaternion)
    zeroed, neutral_q = neutral_zero(relative, neutral)
    axes = estimate_functional_axes(zeroed, t, functional_annotations)
    theta_fe, theta_rud, theta_ps = project_angles(zeroed, axes)
    sample_rate = 1.0 / float(np.median(np.diff(t)))
    window = max(1, int(round(smoothing_seconds * sample_rate)))
    if window % 2 == 0:
        window += 1
    theta_fe = moving_average(theta_fe, window)
    theta_rud = moving_average(theta_rud, window)
    theta_ps = moving_average(theta_ps, window)
    angular_velocity = np.sqrt(np.gradient(theta_fe, t) ** 2 + np.gradient(theta_rud, t) ** 2)
    quality = np.minimum(forearm_result.quality, hand_result.quality)
    return WristKinematics(
        timestamp_s=t,
        forearm_quaternion=forearm_result.quaternion,
        hand_quaternion=hand_result.quaternion,
        relative_quaternion=zeroed,
        theta_fe_deg=theta_fe,
        theta_rud_deg=theta_rud,
        theta_ps_deg=theta_ps,
        angular_velocity_deg_s=angular_velocity,
        quality=quality,
        axes=axes,
        neutral_quaternion=neutral_q,
        forearm_gyro_bias_rad_s=forearm_bias,
        hand_gyro_bias_rad_s=hand_bias,
        algorithm=algorithm,
    )


def compute_wrist_kinematics_from_profile(
    timestamp_s: np.ndarray,
    forearm_accel_mps2: np.ndarray,
    forearm_gyro_rad_s: np.ndarray,
    hand_accel_mps2: np.ndarray,
    hand_gyro_rad_s: np.ndarray,
    calibration_profile: Mapping[str, object],
    forearm_mag_uT: np.ndarray | None = None,
    hand_mag_uT: np.ndarray | None = None,
    algorithm: str = "madgwick",
    fusion_config: Mapping[str, object] | None = None,
    quality_config: Mapping[str, object] | None = None,
    smoothing_seconds: float = 0.10,
) -> WristKinematics:
    """Apply a frozen neutral/axis profile to a separately recorded task stream.

    Each file uses deterministic accelerometer-tilt/yaw-zero initialization. The
    sensor mounting and local-axis convention must remain unchanged after CAL.
    No axes or neutral pose are re-estimated from task data.
    """
    t = np.asarray(timestamp_s, dtype=float)
    if len(t) < 3 or np.any(np.diff(t) <= 0.0):
        raise ValueError("at least three strictly increasing samples are required")
    fusion_config = {} if fusion_config is None else dict(fusion_config)
    quality_config = {} if quality_config is None else dict(quality_config)
    axes = FunctionalAxes(
        np.asarray(calibration_profile["flexion_extension_axis"], dtype=float),
        np.asarray(calibration_profile["radial_ulnar_axis"], dtype=float),
        np.asarray(calibration_profile["pronation_supination_axis"], dtype=float),
    )
    neutral_q = normalize(np.asarray(calibration_profile["neutral_quaternion"], dtype=float))
    forearm_bias = np.asarray(calibration_profile["forearm_gyro_bias_rad_s"], dtype=float)
    hand_bias = np.asarray(calibration_profile["hand_gyro_bias_rad_s"], dtype=float)
    forearm_result = _orientation(
        t,
        forearm_accel_mps2,
        forearm_gyro_rad_s,
        forearm_mag_uT,
        forearm_bias,
        algorithm,
        fusion_config,
        quality_config,
        True,
    )
    hand_result = _orientation(
        t,
        hand_accel_mps2,
        hand_gyro_rad_s,
        hand_mag_uT,
        hand_bias,
        algorithm,
        fusion_config,
        quality_config,
        True,
    )
    relative = relative_quaternion(forearm_result.quaternion, hand_result.quaternion)
    zeroed = normalize(multiply(conjugate(neutral_q), relative))
    theta_fe, theta_rud, theta_ps = project_angles(zeroed, axes)
    sample_rate = 1.0 / float(np.median(np.diff(t)))
    window = max(1, int(round(smoothing_seconds * sample_rate)))
    if window % 2 == 0:
        window += 1
    theta_fe = moving_average(theta_fe, window)
    theta_rud = moving_average(theta_rud, window)
    theta_ps = moving_average(theta_ps, window)
    angular_velocity = np.sqrt(np.gradient(theta_fe, t) ** 2 + np.gradient(theta_rud, t) ** 2)
    quality = np.minimum(forearm_result.quality, hand_result.quality)
    return WristKinematics(
        timestamp_s=t,
        forearm_quaternion=forearm_result.quaternion,
        hand_quaternion=hand_result.quaternion,
        relative_quaternion=zeroed,
        theta_fe_deg=theta_fe,
        theta_rud_deg=theta_rud,
        theta_ps_deg=theta_ps,
        angular_velocity_deg_s=angular_velocity,
        quality=quality,
        axes=axes,
        neutral_quaternion=neutral_q,
        forearm_gyro_bias_rad_s=forearm_bias,
        hand_gyro_bias_rad_s=hand_bias,
        algorithm=algorithm,
    )


def kinematics_rows(result: WristKinematics, calibration_id: str) -> list[dict[str, object]]:
    rows = []
    for i, timestamp in enumerate(result.timestamp_s):
        rows.append(
            {
                "timestamp_ms": int(round(timestamp * 1000.0)),
                "theta_FE": float(result.theta_fe_deg[i]),
                "theta_RUD": float(result.theta_rud_deg[i]),
                "theta_thumb": "",
                "angular_velocity": float(result.angular_velocity_deg_s[i]),
                "calibration_id": calibration_id,
                "quality": float(result.quality[i]),
            }
        )
    return rows