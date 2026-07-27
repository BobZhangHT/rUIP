"""Piecewise-exponential proportional-hazards utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp


FloatArray = NDArray[np.float64]


def exposure_matrix(times: ArrayLike, interval_starts: ArrayLike) -> FloatArray:
    """Return time spent in intervals whose final interval extends to infinity."""
    t = np.asarray(times, dtype=float).reshape(-1)
    starts = np.asarray(interval_starts, dtype=float).reshape(-1)
    if starts.size == 0 or starts[0] != 0.0 or np.any(np.diff(starts) <= 0):
        raise ValueError("interval_starts must begin at zero and increase strictly")
    out = np.zeros((t.size, starts.size), dtype=float)
    for j, start in enumerate(starts):
        end = starts[j + 1] if j + 1 < starts.size else np.inf
        out[:, j] = np.maximum(0.0, np.minimum(t, end) - start)
    return out


@dataclass(frozen=True)
class PiecewiseBaseline:
    """Piecewise-constant hazard with the last interval extending to infinity."""

    interval_starts: FloatArray
    hazards: FloatArray

    def __post_init__(self) -> None:
        starts = np.asarray(self.interval_starts, dtype=float)
        hazards = np.asarray(self.hazards, dtype=float)
        if starts.ndim != 1 or hazards.ndim != 1 or starts.size != hazards.size:
            raise ValueError("interval_starts and hazards must be one-dimensional and equally sized")
        if starts.size == 0 or starts[0] != 0.0 or np.any(np.diff(starts) <= 0):
            raise ValueError("interval_starts must begin at zero and increase strictly")
        if np.any(~np.isfinite(hazards)) or np.any(hazards <= 0):
            raise ValueError("hazards must be finite and strictly positive")
        object.__setattr__(self, "interval_starts", starts)
        object.__setattr__(self, "hazards", hazards)

    def cumulative_hazard(self, times: ArrayLike) -> FloatArray:
        return exposure_matrix(times, self.interval_starts) @ self.hazards

    def inverse_cumulative_hazard(self, values: ArrayLike) -> FloatArray:
        values_array = np.asarray(values, dtype=float)
        flat = values_array.reshape(-1)
        if np.any(~np.isfinite(flat)) or np.any(flat < 0):
            raise ValueError("cumulative-hazard values must be finite and nonnegative")

        starts = self.interval_starts
        hazards = self.hazards
        cumulative_at_start = np.zeros(starts.size, dtype=float)
        if starts.size > 1:
            cumulative_at_start[1:] = np.cumsum(hazards[:-1] * np.diff(starts))

        interval = np.searchsorted(cumulative_at_start[1:], flat, side="left")
        result = starts[interval] + (flat - cumulative_at_start[interval]) / hazards[interval]
        return result.reshape(values_array.shape)


@dataclass(frozen=True)
class CoxSummary:
    theta_hat: float
    se: float
    converged: bool
    iterations: int


def fit_cox_summary(
    observed_time: ArrayLike,
    event: ArrayLike,
    treatment: ArrayLike,
    covariate: ArrayLike | None = None,
    max_iter: int = 80,
    tolerance: float = 1e-9,
) -> CoxSummary:
    """Fit a small Cox PH model by Newton-Raphson using Breslow ties."""
    time = np.asarray(observed_time, dtype=float)
    delta = np.asarray(event, dtype=int)
    z = np.asarray(treatment, dtype=float)
    if covariate is None:
        design = z[:, None]
    else:
        x = np.asarray(covariate, dtype=float)
        design = np.column_stack((z, x))
    if not (time.shape == delta.shape == z.shape) or design.shape[0] != time.size:
        raise ValueError("Cox inputs must have matching lengths")
    if delta.sum() == 0:
        raise ValueError("at least one event is required for a Cox summary")

    event_times = np.unique(time[delta == 1])

    def components(coef: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        linear = design @ coef
        loglik = float(np.sum(linear[delta == 1]))
        score = np.sum(design[delta == 1], axis=0)
        information = np.zeros((design.shape[1], design.shape[1]), dtype=float)
        for event_time in event_times:
            d = int(np.sum((time == event_time) & (delta == 1)))
            risk = time >= event_time
            lr = linear[risk]
            xr = design[risk]
            normalizer = logsumexp(lr)
            weights = np.exp(lr - normalizer)
            mean = weights @ xr
            second = (xr.T * weights) @ xr
            loglik -= d * float(normalizer)
            score -= d * mean
            information += d * (second - np.outer(mean, mean))
        return loglik, score, information

    coef = np.zeros(design.shape[1], dtype=float)
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        loglik, score, information = components(coef)
        regularized = information + np.eye(information.shape[0]) * 1e-9
        step = np.linalg.solve(regularized, score)
        scale = 1.0
        while scale > 1e-8 and components(coef + scale * step)[0] < loglik:
            scale *= 0.5
        coef = coef + scale * step
        if np.max(np.abs(scale * step)) < tolerance:
            converged = True
            break

    _, _, information = components(coef)
    covariance = np.linalg.inv(information + np.eye(information.shape[0]) * 1e-9)
    se = float(np.sqrt(covariance[0, 0]))
    if not np.isfinite(se) or se <= 0:
        raise RuntimeError("invalid standard error from Cox fit")
    return CoxSummary(float(coef[0]), se, converged, iteration)
