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
from .priors import (
    CommensurateSpecification,
    UIPSpecification,
    alr_to_simplex,
    sample_truncated_gamma_m,
    simplex_to_alr,
    uip_weight_log_density,
)


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


def update_uip_weights(
    uip: UIPSpecification,
    theta: float,
    m: float,
    log_ratios: np.ndarray,
    rng: np.random.Generator,
    width: float = 1.0,
    max_steps: int = 80,
) -> tuple[UIPSpecification, np.ndarray]:
    """Update all additive-log-ratio coordinates by conditional slice sampling."""
    updated = np.asarray(log_ratios, dtype=float).copy()
    for coordinate in range(updated.size):
        def log_density(value: float) -> float:
            proposal = updated.copy()
            proposal[coordinate] = value
            return uip_weight_log_density(proposal, theta, m, uip)

        updated[coordinate] = slice_sample(
            updated[coordinate], log_density, rng, width=width, max_steps=max_steps
        )
    return uip.reweight(alr_to_simplex(updated)), updated


def run_da_sampler(
    data: IntervalCensoredData,
    interval_starts: np.ndarray,
    method: str,
    config: SamplerConfig,
    rng: np.random.Generator,
    uip: UIPSpecification | None = None,
    commensurate: CommensurateSpecification | None = None,
) -> pd.DataFrame:
    config.validate()
    allowed = {"IC-NIP", "IC-UIP", "IC-CP"}
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}")
    if method == "IC-UIP" and uip is None:
        raise ValueError("UIP methods require a UIP specification")
    if method == "IC-CP" and commensurate is None:
        raise ValueError("IC-CP requires a commensurate-prior specification")

    starts = np.asarray(interval_starts, dtype=float)
    latent = initialize_latent_times(data)
    theta = 0.0
    beta = 0.0
    exposure, counts, event = complete_data_sufficient_statistics(data, latent, starts)
    hazards = (counts + config.baseline_prior_shape) / (
        exposure.sum(axis=0) + config.baseline_prior_rate
    )
    current_uip = uip
    weight_log_ratios = None if uip is None else simplex_to_alr(uip.weights)
    if method == "IC-UIP":
        m = 0.5 * uip.m_max
    else:
        m = 0.0
    historical_mean = None if commensurate is None else commensurate.initial_historical_mean
    commensurate_precision = (
        None if commensurate is None else commensurate.precision_shape / commensurate.precision_rate
    )
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
            if method == "IC-NIP":
                return likelihood - 0.5 * (value / config.theta_nip_sd) ** 2
            if method == "IC-CP":
                return likelihood - 0.5 * commensurate_precision * (value - historical_mean) ** 2
            precision = current_uip.precision(m)
            return likelihood - 0.5 * precision * (value - current_uip.mean) ** 2

        theta = slice_sample(theta, theta_log_density, rng, config.slice_width, config.slice_steps)

        def beta_log_density(value: float) -> float:
            likelihood = complete_loglik(theta, value, integrated)
            return likelihood - 0.5 * (value / config.beta_prior_sd) ** 2

        beta = slice_sample(beta, beta_log_density, rng, config.slice_width, config.slice_steps)

        if method == "IC-UIP":
            current_uip, weight_log_ratios = update_uip_weights(
                current_uip,
                theta,
                m,
                weight_log_ratios,
                rng,
                width=config.slice_width,
                max_steps=config.slice_steps,
            )
        if method == "IC-UIP":
            m = sample_truncated_gamma_m(rng, theta, current_uip)
        elif method == "IC-CP":
            historical_precision = np.asarray(
                [1.0 / item.se**2 for item in commensurate.summaries], dtype=float
            )
            historical_estimates = np.asarray(
                [item.theta_hat for item in commensurate.summaries], dtype=float
            )
            prior_precision = 1.0 / commensurate.historical_mean_prior_sd**2
            posterior_precision = historical_precision.sum() + commensurate_precision + prior_precision
            posterior_mean = (
                historical_precision @ historical_estimates + commensurate_precision * theta
            ) / posterior_precision
            historical_mean = rng.normal(posterior_mean, np.sqrt(1.0 / posterior_precision))
            commensurate_precision = rng.gamma(
                commensurate.precision_shape + 0.5,
                1.0
                / (
                    commensurate.precision_rate
                    + 0.5 * (theta - historical_mean) ** 2
                ),
            )

        if iteration >= config.burn_in and (iteration - config.burn_in) % config.thin == 0:
            row = {"iteration": float(iteration + 1), "theta": theta, "beta": beta}
            if method == "IC-NIP":
                row["m"] = 0.0
            elif method == "IC-UIP":
                row.update(
                    m=m,
                    uip_mean=current_uip.mean,
                    uip_unit_information=current_uip.unit_information,
                )
                row.update(
                    {f"weight_{index + 1}": float(value) for index, value in enumerate(current_uip.weights)}
                )
            else:
                row.update(
                    commensurate_precision=float(commensurate_precision),
                    historical_mean=float(historical_mean),
                )
            row.update({f"lambda_{j + 1}": float(value) for j, value in enumerate(hazards)})
            records.append(row)

    draws = pd.DataFrame.from_records(records)
    if draws.empty or not np.isfinite(draws.to_numpy()).all():
        raise RuntimeError("sampler produced empty or non-finite posterior draws")
    return draws
