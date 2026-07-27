"""Latent failure-time and Gamma baseline updates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data_generation import IntervalCensoredData
from .interval_ph import PiecewiseBaseline, exposure_matrix


FloatArray = NDArray[np.float64]


def initialize_latent_times(data: IntervalCensoredData) -> FloatArray:
    latent = data.left.copy()
    finite = data.finite_event
    latent[finite] = 0.5 * (data.left[finite] + data.right[finite])
    return latent


def sample_latent_times(
    data: IntervalCensoredData,
    baseline: PiecewiseBaseline,
    theta: float,
    beta: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw finite latent failures by stable truncated-exponential inversion."""
    latent = data.left.copy()
    finite = data.finite_event
    if not np.any(finite):
        return latent
    risk = np.exp(np.clip(theta * data.treatment[finite] + beta * data.covariate[finite], -30.0, 30.0))
    h_left = risk * baseline.cumulative_hazard(data.left[finite])
    h_right = risk * baseline.cumulative_hazard(data.right[finite])
    width = h_right - h_left
    if np.any(width <= 0) or np.any(~np.isfinite(width)):
        raise RuntimeError("invalid cumulative-hazard width for a finite observed interval")
    uniform = rng.uniform(np.finfo(float).eps, 1.0 - np.finfo(float).eps, size=width.size)
    increment = -np.log1p(-uniform * (-np.expm1(-width)))
    target = (h_left + increment) / risk
    latent[finite] = baseline.inverse_cumulative_hazard(target)
    tolerance = 1e-10
    if np.any(latent[finite] <= data.left[finite] - tolerance) or np.any(
        latent[finite] > data.right[finite] + tolerance
    ):
        raise RuntimeError("latent failure draw escaped its observed interval")
    return latent


def complete_data_sufficient_statistics(
    data: IntervalCensoredData,
    latent_times: ArrayLike,
    interval_starts: ArrayLike,
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    latent = np.asarray(latent_times, dtype=float)
    event = data.finite_event.astype(np.int64)
    end_time = np.where(data.finite_event, latent, data.left)
    exposure = exposure_matrix(end_time, interval_starts)
    starts = np.asarray(interval_starts, dtype=float)
    event_interval = np.searchsorted(starts, latent[data.finite_event], side="right") - 1
    event_interval = np.clip(event_interval, 0, starts.size - 1)
    counts = np.bincount(event_interval, minlength=starts.size).astype(np.int64)
    return exposure, counts, event


def baseline_full_conditional(
    data: IntervalCensoredData,
    latent_times: ArrayLike,
    interval_starts: ArrayLike,
    theta: float,
    beta: float,
    prior_shape: float,
    prior_rate: float,
) -> tuple[FloatArray, FloatArray]:
    exposure, counts, _ = complete_data_sufficient_statistics(data, latent_times, interval_starts)
    risk = np.exp(np.clip(theta * data.treatment + beta * data.covariate, -30.0, 30.0))
    shape = prior_shape + counts.astype(float)
    rate = prior_rate + np.sum(risk[:, None] * exposure, axis=0)
    if np.any(shape <= 0) or np.any(rate <= 0) or np.any(~np.isfinite(rate)):
        raise RuntimeError("invalid Gamma baseline full-conditional parameters")
    return shape, rate
