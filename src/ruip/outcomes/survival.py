"""Cox partial-likelihood utilities for a single binary treatment effect.

The baseline hazard is deliberately absent from every object in this module.
Simulation code may use any baseline hazard to generate event times, but the
analysis receives only compressed risk sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


@dataclass(frozen=True)
class CoxRiskSets:
    """Breslow-compressed risk sets at distinct event times."""

    risk_control: np.ndarray
    risk_treatment: np.ndarray
    events: np.ndarray
    treated_events: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.risk_control,
                self.risk_treatment,
                self.events,
                self.treated_events,
            )
        )
        if not arrays[0].ndim == 1 or any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("Cox risk-set arrays must be one-dimensional and aligned")
        if arrays[0].size == 0:
            raise ValueError("at least one event time is required")
        if np.any(arrays[0] < 0) or np.any(arrays[1] < 0):
            raise ValueError("risk-set counts must be nonnegative")
        if np.any(arrays[2] <= 0) or np.any(arrays[3] < 0):
            raise ValueError("event counts are invalid")
        if np.any(arrays[3] > arrays[2]):
            raise ValueError("treated events cannot exceed total events")
        if np.any(arrays[2] > arrays[0] + arrays[1]):
            raise ValueError("events cannot exceed the risk set")
        for name, value in zip(
            ("risk_control", "risk_treatment", "events", "treated_events"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def event_count(self) -> int:
        return int(self.events.sum())


def compress_risk_sets(
    time: np.ndarray, event: np.ndarray, treatment: np.ndarray
) -> CoxRiskSets:
    """Compress subject records into Breslow risk sets.

    Ties are handled by the Breslow approximation. With continuously generated
    event times, ties occur only through external rounding.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    treatment = np.asarray(treatment)
    if time.ndim != 1 or event.shape != time.shape or treatment.shape != time.shape:
        raise ValueError("time, event, and treatment must be aligned vectors")
    if time.size == 0 or np.any(~np.isfinite(time)) or np.any(time < 0):
        raise ValueError("follow-up times must be finite, nonnegative, and nonempty")
    if np.any((event != 0) & (event != 1)) or np.any(
        (treatment != 0) & (treatment != 1)
    ):
        raise ValueError("event and treatment must be binary")

    event_times = np.unique(time[event.astype(bool)])
    if event_times.size == 0:
        raise ValueError("a Cox partial likelihood requires at least one event")
    risk_control = np.empty(event_times.size, dtype=float)
    risk_treatment = np.empty(event_times.size, dtype=float)
    events = np.empty(event_times.size, dtype=float)
    treated_events = np.empty(event_times.size, dtype=float)
    for i, value in enumerate(event_times):
        at_risk = time >= value
        at_event = (time == value) & event.astype(bool)
        risk_control[i] = np.sum(at_risk & (treatment == 0))
        risk_treatment[i] = np.sum(at_risk & (treatment == 1))
        events[i] = np.sum(at_event)
        treated_events[i] = np.sum(at_event & (treatment == 1))
    return CoxRiskSets(risk_control, risk_treatment, events, treated_events)


def log_partial_likelihood(theta: np.ndarray | float, data: CoxRiskSets) -> np.ndarray:
    """Return the Cox log partial likelihood up to a theta-free constant."""
    theta_array = np.asarray(theta, dtype=float)
    flat = theta_array.reshape(-1)
    log_r0 = np.full(data.risk_control.shape, -np.inf)
    log_r1 = np.full(data.risk_treatment.shape, -np.inf)
    np.log(data.risk_control, out=log_r0, where=data.risk_control > 0)
    np.log(data.risk_treatment, out=log_r1, where=data.risk_treatment > 0)
    denominators = np.logaddexp(
        log_r0[:, None], log_r1[:, None] + flat[None, :]
    )
    values = data.treated_events.sum() * flat - np.sum(
        data.events[:, None] * denominators, axis=0
    )
    return values.reshape(theta_array.shape)


def score_information(theta: float, data: CoxRiskSets) -> tuple[float, float]:
    """Return Cox score and observed information at ``theta``."""
    log_r0 = np.full(data.risk_control.shape, -np.inf)
    log_r1e = np.full(data.risk_treatment.shape, -np.inf)
    np.log(data.risk_control, out=log_r0, where=data.risk_control > 0)
    np.log(data.risk_treatment, out=log_r1e, where=data.risk_treatment > 0)
    log_r1e += theta
    probability = np.exp(log_r1e - np.logaddexp(log_r0, log_r1e))
    score = float(data.treated_events.sum() - np.dot(data.events, probability))
    information = float(np.dot(data.events, probability * (1.0 - probability)))
    return score, information


def partial_likelihood_summary(data: CoxRiskSets) -> tuple[float, float]:
    """Return the finite Cox PL MLE and its observed information."""
    lower, upper = -40.0, 40.0
    lower_score, _ = score_information(lower, data)
    upper_score, _ = score_information(upper, data)
    if lower_score <= 0.0 or upper_score >= 0.0:
        raise ValueError("Cox partial-likelihood MLE is on the boundary")
    estimate = float(brentq(lambda x: score_information(x, data)[0], lower, upper))
    information = score_information(estimate, data)[1]
    if not np.isfinite(information) or information <= 0.0:
        raise ValueError("Cox observed information is not positive and finite")
    return estimate, information


def posterior_summary(
    current: CoxRiskSets,
    *,
    historical: tuple[CoxRiskSets, ...] = (),
    historical_weight: float = 0.0,
    prior_means: np.ndarray | None = None,
    prior_variances: np.ndarray | None = None,
    prior_log_weights: np.ndarray | None = None,
    nodes: int = 801,
) -> tuple[float, float, float, float, float]:
    """Numerically summarize a PL posterior.

    With no Gaussian prior this is the normalized current partial likelihood;
    ``historical_weight`` adds a power prior directly on historical partial
    likelihoods. Gaussian-mixture arguments instead combine the current PL with
    the existing summary-based comparator prior.
    """
    estimate, information = partial_likelihood_summary(current)
    likelihood_terms = ((1.0, current),) + tuple(
        (historical_weight, item) for item in historical if historical_weight
    )

    def combined_score(value: float) -> float:
        return float(
            sum(
                weight * score_information(value, item)[0]
                for weight, item in likelihood_terms
            )
        )

    lower_score, upper_score = combined_score(-40.0), combined_score(40.0)
    if lower_score <= 0.0 or upper_score >= 0.0:
        raise ValueError("combined Cox partial-likelihood mode is on the boundary")
    mode = float(brentq(combined_score, -40.0, 40.0))
    mode_information = float(
        sum(
            weight * score_information(mode, item)[1]
            for weight, item in likelihood_terms
        )
    )
    anchors = [
        (estimate, 1.0 / np.sqrt(information)),
        (mode, 1.0 / np.sqrt(mode_information)),
    ]
    if historical_weight:
        anchors.extend(
            (item_estimate, 1.0 / np.sqrt(item_information))
            for item_estimate, item_information in (
                partial_likelihood_summary(item) for item in historical
            )
        )
    if prior_means is not None:
        anchors.extend(
            (float(mean), float(np.sqrt(variance)))
            for mean, variance in zip(prior_means, prior_variances, strict=True)
        )
    lower = max(-40.0, min(center - 18.0 * scale for center, scale in anchors))
    upper = min(40.0, max(center + 18.0 * scale for center, scale in anchors))
    grid = np.linspace(lower, upper, nodes)
    log_kernel = log_partial_likelihood(grid, current)
    if historical_weight:
        if historical_weight < 0.0:
            raise ValueError("historical_weight must be nonnegative")
        for item in historical:
            log_kernel += historical_weight * log_partial_likelihood(grid, item)
    if prior_means is not None:
        means = np.asarray(prior_means, dtype=float)
        variances = np.asarray(prior_variances, dtype=float)
        weights = np.asarray(prior_log_weights, dtype=float)
        components = (
            weights[:, None]
            - 0.5 * np.log(2.0 * np.pi * variances[:, None])
            - 0.5 * (grid[None, :] - means[:, None]) ** 2 / variances[:, None]
        )
        log_kernel += logsumexp(components, axis=0)
    log_kernel -= float(np.max(log_kernel))

    def integrate(x: np.ndarray, log_y: np.ndarray):
        density = np.exp(log_y)
        increments = 0.5 * (density[1:] + density[:-1]) * np.diff(x)
        mass = float(increments.sum())
        cdf = np.r_[0.0, np.cumsum(increments)] / mass
        mean = float(np.trapezoid(x * density, x) / mass)
        second = float(np.trapezoid(x * x * density, x) / mass)
        quantiles = np.interp([0.025, 0.975], cdf, x)
        return mean, max(second - mean * mean, 0.0), quantiles, mass

    fine = integrate(grid, log_kernel)
    coarse = integrate(grid[::2], log_kernel[::2])
    discrepancy = abs(coarse[3] / fine[3] - 1.0)
    return fine[0], fine[1], float(fine[2][0]), float(fine[2][1]), float(discrepancy)
