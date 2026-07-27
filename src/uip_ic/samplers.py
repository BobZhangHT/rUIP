"""MCMC sampler for interval-censored piecewise-exponential PH models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .augmentation import (
    baseline_full_conditional,
    complete_data_sufficient_statistics,
    initialize_latent_times,
    sample_latent_times,
)
from .data_generation import IntervalCensoredData
from .interval_ph import PiecewiseBaseline
from .priors import UIPSpecification, sample_truncated_gamma_m


@dataclass(frozen=True)
class SamplerConfig:
    iterations: int = 800
    burn_in: int = 400
    thin: int = 1
    baseline_prior_shape: float = 0.5
    baseline_prior_rate: float = 0.5
    theta_nip_sd: float = 10.0
    beta_prior_sd: float = 10.0
    slice_width: float = 0.8
    slice_steps: int = 80

    def validate(self) -> None:
        if self.iterations <= self.burn_in or self.burn_in < 0 or self.thin <= 0:
            raise ValueError("iterations, burn_in, and thin imply no posterior draws")
        if min(
            self.baseline_prior_shape,
            self.baseline_prior_rate,
            self.theta_nip_sd,
            self.beta_prior_sd,
            self.slice_width,
            self.slice_steps,
        ) <= 0:
            raise ValueError("sampler tuning and prior values must be positive")


def slice_sample(
    current: float,
    log_density: Callable[[float], float],
    rng: np.random.Generator,
    width: float = 1.0,
    max_steps: int = 80,
) -> float:
    """One-dimensional stepping-out slice sampler."""
    log_current = float(log_density(current))
    if not np.isfinite(log_current):
        raise RuntimeError("slice sampler started at a non-finite log density")
    log_level = log_current + np.log(rng.uniform(np.finfo(float).eps, 1.0))
    left = current - width * rng.uniform()
    right = left + width
    left_steps = int(rng.integers(0, max_steps + 1))
    right_steps = max_steps - left_steps
    while left_steps > 0 and log_density(left) > log_level:
        left -= width
        left_steps -= 1
    while right_steps > 0 and log_density(right) > log_level:
        right += width
        right_steps -= 1
    for _ in range(1000):
        proposal = rng.uniform(left, right)
        if log_density(proposal) >= log_level:
            return float(proposal)
        if proposal < current:
            left = proposal
        else:
            right = proposal
    raise RuntimeError("slice sampler failed to accept after 1000 shrinkage steps")


def run_da_sampler(
    data: IntervalCensoredData,
    interval_starts: np.ndarray,
    method: str,
    config: SamplerConfig,
    rng: np.random.Generator,
    uip: UIPSpecification | None = None,
) -> pd.DataFrame:
    config.validate()
    allowed = {"NIP-DA", "IC-UIP-DA", "Full-borrowing"}
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}")
    if method != "NIP-DA" and uip is None:
        raise ValueError("UIP methods require a UIP specification")

    starts = np.asarray(interval_starts, dtype=float)
    latent = initialize_latent_times(data)
    theta = 0.0
    beta = 0.0
    exposure, counts, event = complete_data_sufficient_statistics(data, latent, starts)
    hazards = (counts + config.baseline_prior_shape) / (
        exposure.sum(axis=0) + config.baseline_prior_rate
    )
    m = 0.0 if method == "NIP-DA" else (uip.m_max if method == "Full-borrowing" else 0.5 * uip.m_max)
    records: list[dict[str, float]] = []

    def complete_loglik(theta_value: float, beta_value: float, integrated: np.ndarray) -> float:
        eta = theta_value * data.treatment + beta_value * data.covariate
        risk = np.exp(np.clip(eta, -30.0, 30.0))
        return float(np.sum(event * eta) - np.sum(risk * integrated))

    for iteration in range(config.iterations):
        baseline = PiecewiseBaseline(starts, hazards)
        latent = sample_latent_times(data, baseline, theta, beta, rng)

        shape, rate = baseline_full_conditional(
            data,
            latent,
            starts,
            theta,
            beta,
            config.baseline_prior_shape,
            config.baseline_prior_rate,
        )
        hazards = rng.gamma(shape, 1.0 / rate)
        exposure, _, event = complete_data_sufficient_statistics(data, latent, starts)
        integrated = exposure @ hazards

        def theta_log_density(value: float) -> float:
            likelihood = complete_loglik(value, beta, integrated)
            if method == "NIP-DA":
                return likelihood - 0.5 * (value / config.theta_nip_sd) ** 2
            precision = uip.precision(m)
            return likelihood - 0.5 * precision * (value - uip.mean) ** 2

        theta = slice_sample(theta, theta_log_density, rng, config.slice_width, config.slice_steps)

        def beta_log_density(value: float) -> float:
            likelihood = complete_loglik(theta, value, integrated)
            return likelihood - 0.5 * (value / config.beta_prior_sd) ** 2

        beta = slice_sample(beta, beta_log_density, rng, config.slice_width, config.slice_steps)

        if method == "IC-UIP-DA":
            m = sample_truncated_gamma_m(rng, theta, uip)

        if iteration >= config.burn_in and (iteration - config.burn_in) % config.thin == 0:
            row = {"iteration": float(iteration + 1), "theta": theta, "beta": beta, "m": m}
            row.update({f"lambda_{j + 1}": float(value) for j, value in enumerate(hazards)})
            records.append(row)

    draws = pd.DataFrame.from_records(records)
    if draws.empty or not np.isfinite(draws.to_numpy()).all():
        raise RuntimeError("sampler produced empty or non-finite posterior draws")
    return draws
