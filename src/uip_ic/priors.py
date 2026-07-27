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

    def precision(self, m: float) -> float:
        return float(m * self.unit_information)


def build_uip(
    summaries: Sequence[HistoricalSummary],
    m_max: float,
    weighting: str = "equal",
    preset_weights: Sequence[float] | None = None,
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
    theta = np.asarray([item.theta_hat for item in values], dtype=float)
    return UIPSpecification(
        mean=float(weights @ theta),
        unit_information=float(weights @ unit),
        weights=weights,
        m_max=float(m_max),
        summaries=values,
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
