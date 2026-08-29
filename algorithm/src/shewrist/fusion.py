"""Orientation estimation for 6-axis or 9-axis IMU streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quaternion import multiply, normalize

_EPS = 1e-12


@dataclass(frozen=True)
class OrientationResult:
    quaternion: np.ndarray
    quality: np.ndarray
    gyro_bias: np.ndarray
    algorithm: str


def _gyro_derivative(q: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    return 0.5 * multiply(q, np.array([0.0, gyro[0], gyro[1], gyro[2]]))


def _madgwick_gradient_imu(q: np.ndarray, accel: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    ax, ay, az = accel
    f = np.array(
        [
            2.0 * (q1 * q3 - q0 * q2) - ax,
            2.0 * (q0 * q1 + q2 * q3) - ay,
            2.0 * (0.5 - q1 * q1 - q2 * q2) - az,
        ]
    )
    jacobian = np.array(
        [
            [-2.0 * q2, 2.0 * q3, -2.0 * q0, 2.0 * q1],
            [2.0 * q1, 2.0 * q0, 2.0 * q3, 2.0 * q2],
            [0.0, -4.0 * q1, -4.0 * q2, 0.0],
        ]
    )
    return np.sum(jacobian * f[:, None], axis=0)


def _madgwick_gradient_marg(q: np.ndarray, accel: np.ndarray, mag: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    ax, ay, az = accel
    mx, my, mz = mag
    hx = 2.0 * mx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * my * (q1 * q2 - q0 * q3) + 2.0 * mz * (q1 * q3 + q0 * q2)
    hy = 2.0 * mx * (q1 * q2 + q0 * q3) + 2.0 * my * (0.5 - q1 * q1 - q3 * q3) + 2.0 * mz * (q2 * q3 - q0 * q1)
    bx = np.sqrt(hx * hx + hy * hy)
    bz = 2.0 * mx * (q1 * q3 - q0 * q2) + 2.0 * my * (q2 * q3 + q0 * q1) + 2.0 * mz * (0.5 - q1 * q1 - q2 * q2)
    f1 = 2.0 * (q1 * q3 - q0 * q2) - ax
    f2 = 2.0 * (q0 * q1 + q2 * q3) - ay
    f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - az
    f4 = 2.0 * bx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * bz * (q1 * q3 - q0 * q2) - mx
    f5 = 2.0 * bx * (q1 * q2 - q0 * q3) + 2.0 * bz * (q0 * q1 + q2 * q3) - my
    f6 = 2.0 * bx * (q0 * q2 + q1 * q3) + 2.0 * bz * (0.5 - q1 * q1 - q2 * q2) - mz
    return np.array(
        [
            -2.0 * q2 * f1 + 2.0 * q1 * f2 - 2.0 * bz * q2 * f4 + (-2.0 * bx * q3 + 2.0 * bz * q1) * f5 + 2.0 * bx * q2 * f6,
            2.0 * q3 * f1 + 2.0 * q0 * f2 - 4.0 * q1 * f3 + 2.0 * bz * q3 * f4 + (2.0 * bx * q2 + 2.0 * bz * q0) * f5 + (2.0 * bx * q3 - 4.0 * bz * q1) * f6,
            -2.0 * q0 * f1 + 2.0 * q3 * f2 - 4.0 * q2 * f3 + (-4.0 * bx * q2 - 2.0 * bz * q0) * f4 + (2.0 * bx * q1 + 2.0 * bz * q3) * f5 + (2.0 * bx * q0 - 4.0 * bz * q2) * f6,
            2.0 * q1 * f1 + 2.0 * q2 * f2 + (-4.0 * bx * q3 + 2.0 * bz * q1) * f4 + (-2.0 * bx * q0 + 2.0 * bz * q2) * f5 + 2.0 * bx * q1 * f6,
        ]
    )


def madgwick_update(q: np.ndarray, gyro: np.ndarray, accel: np.ndarray, mag: np.ndarray | None, dt: float, beta: float) -> np.ndarray:
    q = normalize(q)
    q_dot = _gyro_derivative(q, gyro)
    accel_norm = np.linalg.norm(accel)
    if accel_norm > _EPS:
        a = accel / accel_norm
        if mag is not None and np.linalg.norm(mag) > _EPS:
            gradient = _madgwick_gradient_marg(q, a, mag / np.linalg.norm(mag))
        else:
            gradient = _madgwick_gradient_imu(q, a)
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > _EPS:
            q_dot -= beta * gradient / gradient_norm
    return normalize(q + q_dot * dt)


def mahony_update(
    q: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    mag: np.ndarray | None,
    dt: float,
    kp: float,
    ki: float,
    integral_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = normalize(q)
    accel_norm = np.linalg.norm(accel)
    if accel_norm < _EPS:
        return normalize(q + _gyro_derivative(q, gyro) * dt), integral_error
    ax, ay, az = accel / accel_norm
    q0, q1, q2, q3 = q
    half_vx = q1 * q3 - q0 * q2
    half_vy = q0 * q1 + q2 * q3
    half_vz = q0 * q0 - 0.5 + q3 * q3
    half_ex = ay * half_vz - az * half_vy
    half_ey = az * half_vx - ax * half_vz
    half_ez = ax * half_vy - ay * half_vx
    if mag is not None and np.linalg.norm(mag) > _EPS:
        mx, my, mz = mag / np.linalg.norm(mag)
        hx = 2.0 * mx * (0.5 - q2 * q2 - q3 * q3) + 2.0 * my * (q1 * q2 - q0 * q3) + 2.0 * mz * (q1 * q3 + q0 * q2)
        hy = 2.0 * mx * (q1 * q2 + q0 * q3) + 2.0 * my * (0.5 - q1 * q1 - q3 * q3) + 2.0 * mz * (q2 * q3 - q0 * q1)
        bx = np.sqrt(hx * hx + hy * hy)
        bz = 2.0 * mx * (q1 * q3 - q0 * q2) + 2.0 * my * (q2 * q3 + q0 * q1) + 2.0 * mz * (0.5 - q1 * q1 - q2 * q2)
        half_wx = bx * (0.5 - q2 * q2 - q3 * q3) + bz * (q1 * q3 - q0 * q2)
        half_wy = bx * (q1 * q2 - q0 * q3) + bz * (q0 * q1 + q2 * q3)
        half_wz = bx * (q0 * q2 + q1 * q3) + bz * (0.5 - q1 * q1 - q2 * q2)
        half_ex += my * half_wz - mz * half_wy
        half_ey += mz * half_wx - mx * half_wz
        half_ez += mx * half_wy - my * half_wx
    error = np.array([half_ex, half_ey, half_ez])
    integral_error = integral_error + 2.0 * ki * error * dt if ki > 0.0 else np.zeros(3)
    corrected = gyro + integral_error + 2.0 * kp * error
    return normalize(q + _gyro_derivative(q, corrected) * dt), integral_error


def _initial_orientation_from_accel(
    timestamp_s: np.ndarray,
    accel_mps2: np.ndarray,
    gravity: float,
    accel_tolerance: float,
) -> np.ndarray:
    """Return a yaw-zero sensor-to-world quaternion from robust startup gravity."""
    t = np.asarray(timestamp_s, dtype=float)
    accel = np.asarray(accel_mps2, dtype=float)
    startup = t <= t[0] + min(0.5, max(0.0, float(t[-1] - t[0])))
    norms = np.linalg.norm(accel, axis=1)
    valid = (
        startup
        & np.all(np.isfinite(accel), axis=1)
        & np.isfinite(norms)
        & (norms > _EPS)
        & (np.abs(norms - gravity) <= 2.0 * accel_tolerance)
    )
    if not np.any(valid):
        return np.array([1.0, 0.0, 0.0, 0.0])
    gravity_body = np.median(accel[valid] / norms[valid, None], axis=0)
    gravity_body /= np.linalg.norm(gravity_body)
    roll = np.arctan2(gravity_body[1], gravity_body[2])
    pitch = np.arctan2(-gravity_body[0], np.hypot(gravity_body[1], gravity_body[2]))
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    return normalize(np.array([cr * cp, sr * cp, cr * sp, -sr * sp]))


def estimate_orientation(
    timestamp_s: np.ndarray,
    accel_mps2: np.ndarray,
    gyro_rad_s: np.ndarray,
    magnetometer_uT: np.ndarray | None = None,
    algorithm: str = "madgwick",
    beta: float = 0.08,
    kp: float = 0.5,
    ki: float = 0.0,
    gyro_bias: np.ndarray | None = None,
    gravity: float = 9.80665,
    accel_tolerance: float = 3.0,
    mag_bounds: tuple[float, float] = (15.0, 80.0),
    initialize_from_accel: bool = False,
) -> OrientationResult:
    t = np.asarray(timestamp_s, dtype=float)
    accel = np.asarray(accel_mps2, dtype=float)
    gyro = np.asarray(gyro_rad_s, dtype=float)
    mag = None if magnetometer_uT is None else np.asarray(magnetometer_uT, dtype=float)
    if len(t) < 2 or accel.shape != gyro.shape or accel.shape != (len(t), 3):
        raise ValueError("time, accel and gyro must describe at least two aligned 3-axis samples")
    if mag is not None and mag.shape != accel.shape:
        raise ValueError("magnetometer must match accel shape")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float)
    gyro = gyro - bias
    q = np.empty((len(t), 4), dtype=float)
    q[0] = _initial_orientation_from_accel(t, accel, gravity, accel_tolerance) if initialize_from_accel else np.array([1.0, 0.0, 0.0, 0.0])
    quality = np.ones(len(t), dtype=float)
    nominal_dt = float(np.median(np.diff(t)))
    integral = np.zeros(3)
    method = algorithm.lower()
    if method not in {"madgwick", "mahony"}:
        raise ValueError("algorithm must be 'madgwick' or 'mahony'")
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        acc_error = abs(np.linalg.norm(accel[i]) - gravity)
        accel_valid = np.isfinite(acc_error) and acc_error <= 2.0 * accel_tolerance
        quality[i] *= max(0.0, 1.0 - acc_error / max(2.0 * accel_tolerance, _EPS))
        if dt > 1.5 * nominal_dt:
            quality[i] *= max(0.0, nominal_dt / dt)
        m = None
        if mag is not None:
            mag_norm = np.linalg.norm(mag[i])
            if mag_bounds[0] <= mag_norm <= mag_bounds[1] and np.all(np.isfinite(mag[i])):
                m = mag[i]
            else:
                quality[i] *= 0.7
        a = accel[i] if accel_valid else np.zeros(3)
        if method == "madgwick":
            q[i] = madgwick_update(q[i - 1], gyro[i], a, m, dt, beta)
        else:
            q[i], integral = mahony_update(q[i - 1], gyro[i], a, m, dt, kp, ki, integral)
    return OrientationResult(q, np.clip(quality, 0.0, 1.0), bias, method)