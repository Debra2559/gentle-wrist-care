"""Quaternion helpers using scalar-first order [w, x, y, z]."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < _EPS):
        raise ValueError("zero-norm quaternion")
    return q / n


def conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    q[..., 1:] *= -1.0
    return q


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def inverse(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm_sq = np.sum(q * q, axis=-1, keepdims=True)
    if np.any(norm_sq < _EPS):
        raise ValueError("zero-norm quaternion")
    return conjugate(q) / norm_sq


def rotate_vector(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize(q)
    vector = np.asarray(vector, dtype=float)
    zeros = np.zeros(vector.shape[:-1] + (1,), dtype=float)
    pure = np.concatenate((zeros, vector), axis=-1)
    return multiply(multiply(q, pure), conjugate(q))[..., 1:]


def from_rotation_vector(rotation_vector: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotation_vector, dtype=float)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > _EPS, np.sin(half) / np.maximum(angle, _EPS), 0.5)
    return normalize(np.concatenate((np.cos(half), rv * scale), axis=-1))


def to_rotation_vector(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(q[..., :1], -1.0, 1.0))
    scale = np.where(sin_half > _EPS, angle / np.maximum(sin_half, _EPS), 2.0)
    return xyz * scale


def average(quaternions: np.ndarray) -> np.ndarray:
    quaternions = normalize(np.asarray(quaternions, dtype=float))
    if quaternions.ndim != 2 or quaternions.shape[1] != 4 or len(quaternions) == 0:
        raise ValueError("quaternions must have shape (n, 4)")
    accumulator = np.einsum("ni,nj->ij", quaternions, quaternions)
    values, vectors = np.linalg.eigh(accumulator)
    result = vectors[:, np.argmax(values)]
    if result[0] < 0.0:
        result *= -1.0
    return normalize(result)


def make_continuous(quaternions: np.ndarray) -> np.ndarray:
    result = normalize(np.asarray(quaternions, dtype=float)).copy()
    for i in range(1, len(result)):
        if np.dot(result[i - 1], result[i]) < 0.0:
            result[i] *= -1.0
    return result


def integrate_gyro(q: np.ndarray, gyro_rad_s: np.ndarray, dt: float) -> np.ndarray:
    omega = np.array([0.0, *np.asarray(gyro_rad_s, dtype=float)])
    q_dot = 0.5 * multiply(q, omega)
    return normalize(np.asarray(q, dtype=float) + q_dot * float(dt))


def to_euler_xyz(q: np.ndarray, degrees: bool = True) -> np.ndarray:
    q = normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    result = np.stack((roll, pitch, yaw), axis=-1)
    return np.degrees(result) if degrees else result


def distance_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = multiply(inverse(a), b)
    w = np.clip(np.abs(normalize(relative)[..., 0]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))