"""Data generation for exact and interval-censored PH outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .interval_ph import PiecewiseBaseline
from .interval_ph import fit_cox_summary
from .priors import HistoricalSummary


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IntervalCensoredData:
    left: FloatArray
    right: FloatArray
    treatment: FloatArray
    covariate: FloatArray
    true_event_time: FloatArray

    @property
    def n(self) -> int:
        return int(self.left.size)

    @property
    def finite_event(self) -> NDArray[np.bool_]:
        return np.isfinite(self.right)

    def validate(self) -> None:
        arrays = [self.left, self.right, self.treatment, self.covariate, self.true_event_time]
        if any(np.asarray(value).shape != (self.n,) for value in arrays):
            raise ValueError("all interval-data arrays must have the same one-dimensional shape")
        if np.any(self.left < 0) or np.any(self.left >= self.right):
            raise ValueError("observed intervals must satisfy 0 <= L < R")
        if np.any(self.true_event_time <= self.left) or np.any(self.true_event_time > self.right):
            raise ValueError("true event times must satisfy L < T <= R")


@dataclass(frozen=True)
class ExactSurvivalData:
    observed_time: FloatArray
    event: NDArray[np.int64]
    treatment: FloatArray
    covariate: FloatArray
    true_event_time: FloatArray

    @property
    def n(self) -> int:
        return int(self.observed_time.size)


def simulate_event_times(
    rng: np.random.Generator,
    baseline: PiecewiseBaseline,
    treatment: ArrayLike,
    covariate: ArrayLike,
    theta: float,
    beta: float,
) -> FloatArray:
    z = np.asarray(treatment, dtype=float)
    x = np.asarray(covariate, dtype=float)
    risk = np.exp(np.clip(theta * z + beta * x, -30.0, 30.0))
    cumulative_target = rng.exponential(scale=1.0, size=z.size) / risk
    return baseline.inverse_cumulative_hazard(cumulative_target)


def construct_intervals(event_times: ArrayLike, inspection_times: ArrayLike) -> tuple[FloatArray, FloatArray]:
    times = np.asarray(event_times, dtype=float)
    inspections = np.asarray(inspection_times, dtype=float)
    if inspections.ndim != 1 or inspections.size == 0 or np.any(np.diff(inspections) <= 0):
        raise ValueError("inspection_times must be a nonempty, strictly increasing vector")
    if inspections[0] <= 0:
        raise ValueError("inspection_times must be positive")
    index = np.searchsorted(inspections, times, side="left")
    right = np.full(times.size, np.inf, dtype=float)
    finite = index < inspections.size
    right[finite] = inspections[index[finite]]
    left = np.empty(times.size, dtype=float)
    left[index == 0] = 0.0
    middle = (index > 0) & finite
    left[middle] = inspections[index[middle] - 1]
    left[~finite] = inspections[-1]
    return left, right


def generate_interval_censored_data(
    rng: np.random.Generator,
    n: int,
    baseline: PiecewiseBaseline,
    theta: float,
    beta: float,
    inspection_times: ArrayLike,
    include_covariate: bool = True,
) -> IntervalCensoredData:
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    covariate = rng.normal(size=n) if include_covariate else np.zeros(n, dtype=float)
    event_time = simulate_event_times(rng, baseline, treatment, covariate, theta, beta)
    left, right = construct_intervals(event_time, inspection_times)
    data = IntervalCensoredData(left, right, treatment, covariate, event_time)
    data.validate()
    return data


def generate_exact_survival_data(
    rng: np.random.Generator,
    n: int,
    baseline: PiecewiseBaseline,
    theta: float,
    beta: float,
    administrative_time: float,
    include_covariate: bool = True,
) -> ExactSurvivalData:
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    covariate = rng.normal(size=n) if include_covariate else np.zeros(n, dtype=float)
    event_time = simulate_event_times(rng, baseline, treatment, covariate, theta, beta)
    observed = np.minimum(event_time, administrative_time)
    event = (event_time <= administrative_time).astype(np.int64)
    return ExactSurvivalData(observed, event, treatment, covariate, event_time)


def generate_historical_summaries(
    rng: np.random.Generator,
    n_per_study: int,
    baseline: PiecewiseBaseline,
    treatment_effects: ArrayLike,
    beta: float,
    administrative_time: float,
    include_covariate: bool = True,
) -> tuple[list[HistoricalSummary], list[ExactSurvivalData]]:
    """Generate exact/right-censored historical studies and retain Cox summaries only."""
    summaries: list[HistoricalSummary] = []
    datasets: list[ExactSurvivalData] = []
    for index, historical_theta in enumerate(np.asarray(treatment_effects, dtype=float), start=1):
        data = generate_exact_survival_data(
            rng,
            n_per_study,
            baseline,
            float(historical_theta),
            beta,
            administrative_time,
            include_covariate,
        )
        fit = fit_cox_summary(
            data.observed_time,
            data.event,
            data.treatment,
            data.covariate if include_covariate else None,
        )
        summaries.append(HistoricalSummary(f"H{index}", fit.theta_hat, fit.se, n_per_study))
        datasets.append(data)
    return summaries, datasets
