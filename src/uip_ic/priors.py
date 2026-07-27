"""Unit-information-prior construction and borrowing-amount updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import gammainc, gammaincinv


@dataclass(frozen=True)
class HistoricalSummary:
    study: str
    theta_hat: float
    se: float
    n: int

    @property
    def unit_information(self) -> float:
        if self.n <= 0 or not np.isfinite(self.se) or self.se <= 0:
            raise ValueError("historical n and SE must be positive")
        return 1.0 / (self.n * self.se**2)


@dataclass(frozen=True)
class UIPSpecification:
    mean: float
    unit_information: float
    weights: np.ndarray
    m_max: float
    summaries: tuple[HistoricalSummary, ...]
    dirichlet_concentration: np.ndarray

    def precision(self, m: float) -> float:
        return float(m * self.unit_information)

    def reweight(self, weights: Sequence[float]) -> "UIPSpecification":
        updated = np.asarray(weights, dtype=float)
        if updated.shape != self.weights.shape or np.any(updated <= 0) or not np.isfinite(updated).all():
            raise ValueError("dynamic UIP weights must be finite and strictly positive")
        updated = updated / updated.sum()
        theta = np.asarray([item.theta_hat for item in self.summaries], dtype=float)
        unit = np.asarray([item.unit_information for item in self.summaries], dtype=float)
        return UIPSpecification(
            mean=float(updated @ theta),
            unit_information=float(updated @ unit),
            weights=updated,
            m_max=self.m_max,
            summaries=self.summaries,
            dirichlet_concentration=self.dirichlet_concentration,
        )


@dataclass(frozen=True)
class CommensurateSpecification:
    """Normal-summary commensurate prior used as a lightweight comparator."""

    summaries: tuple[HistoricalSummary, ...]
    precision_shape: float = 0.5
    precision_rate: float = 0.05
    historical_mean_prior_sd: float = 10.0

    def validate(self) -> None:
        if not self.summaries:
            raise ValueError("commensurate prior needs at least one historical summary")
        if min(self.precision_shape, self.precision_rate, self.historical_mean_prior_sd) <= 0:
            raise ValueError("commensurate-prior hyperparameters must be positive")

    @property
    def initial_historical_mean(self) -> float:
        precision = np.asarray([1.0 / item.se**2 for item in self.summaries], dtype=float)
        estimates = np.asarray([item.theta_hat for item in self.summaries], dtype=float)
        return float(precision @ estimates / precision.sum())


def build_uip(
    summaries: Sequence[HistoricalSummary],
    m_max: float,
    weighting: str = "equal",
    preset_weights: Sequence[float] | None = None,
    dirichlet_concentration: Sequence[float] | None = None,
) -> UIPSpecification:
    values = tuple(summaries)
    if not values:
        raise ValueError("at least one historical summary is required")
    if not np.isfinite(m_max) or m_max <= 0:
        raise ValueError("m_max must be finite and positive")
    unit = np.asarray([item.unit_information for item in values], dtype=float)
    if preset_weights is not None:
        weights = np.asarray(preset_weights, dtype=float)
    elif weighting == "equal":
        weights = np.repeat(1.0 / len(values), len(values))
    elif weighting == "unit_information":
        weights = unit / unit.sum()
    else:
        raise ValueError("weighting must be 'equal', 'unit_information', or accompanied by preset_weights")
    if weights.shape != (len(values),) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("invalid UIP weights")
    if weights.sum() <= 0:
        raise ValueError("UIP weights must have positive sum")
    weights = weights / weights.sum()
    concentration = (
        np.ones(len(values), dtype=float)
        if dirichlet_concentration is None
        else np.asarray(dirichlet_concentration, dtype=float)
    )
    if concentration.shape != (len(values),) or np.any(concentration <= 0) or not np.isfinite(concentration).all():
        raise ValueError("Dirichlet concentrations must be finite and strictly positive")
    theta = np.asarray([item.theta_hat for item in values], dtype=float)
    return UIPSpecification(
        mean=float(weights @ theta),
        unit_information=float(weights @ unit),
        weights=weights,
        m_max=float(m_max),
        summaries=values,
        dirichlet_concentration=concentration,
    )


def build_commensurate_prior(
    summaries: Sequence[HistoricalSummary],
    precision_shape: float = 0.5,
    precision_rate: float = 0.05,
    historical_mean_prior_sd: float = 10.0,
) -> CommensurateSpecification:
    specification = CommensurateSpecification(
        tuple(summaries),
        float(precision_shape),
        float(precision_rate),
        float(historical_mean_prior_sd),
    )
    specification.validate()
    return specification


def alr_to_simplex(log_ratios: Sequence[float]) -> np.ndarray:
    """Map additive log-ratios to an open simplex with the last weight as reference."""
    values = np.append(np.asarray(log_ratios, dtype=float), 0.0)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def simplex_to_alr(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("weights must lie in the open simplex")
    values = values / values.sum()
    return np.log(values[:-1] / values[-1])


def uip_weight_log_density(
    log_ratios: Sequence[float],
    theta: float,
    m: float,
    uip: UIPSpecification,
) -> float:
    """Log density in ALR coordinates, including the softmax Jacobian."""
    weights = alr_to_simplex(log_ratios)
    candidate = uip.reweight(weights)
    # Dirichlet(w | gamma) times |d w / d ALR| = product w_k**gamma_k.
    log_dirichlet_and_jacobian = float(uip.dirichlet_concentration @ np.log(weights))
    return (
        log_dirichlet_and_jacobian
        + 0.5 * np.log(candidate.unit_information)
        - 0.5 * m * candidate.unit_information * (theta - candidate.mean) ** 2
    )


def sample_truncated_gamma_m(
    rng: np.random.Generator,
    theta: float,
    uip: UIPSpecification,
) -> float:
    """Draw M | theta from Gamma(3/2, rate) truncated to (0, m_max)."""
    shape = 1.5
    rate = 0.5 * uip.unit_information * (theta - uip.mean) ** 2
    uniform = rng.uniform(np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    if rate < 1e-12:
        return float(uip.m_max * uniform ** (1.0 / shape))
    upper_probability = float(gammainc(shape, rate * uip.m_max))
    if upper_probability <= 0 or not np.isfinite(upper_probability):
        return float(uip.m_max * uniform ** (1.0 / shape))
    quantile = min(uniform * upper_probability, np.nextafter(1.0, 0.0))
    draw = float(gammaincinv(shape, quantile) / rate)
    return float(np.clip(draw, np.finfo(float).eps, np.nextafter(uip.m_max, 0.0)))
