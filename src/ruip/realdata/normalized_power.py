"""History-only normalized random-power prior for Gaussian summaries.

Conditional on independent a_k ~ Beta(1, 1), each powered historical
Gaussian likelihood is normalized against a N(0, 10) base prior.  Integrating
the normalized conditional priors over a gives an equal-weight Normal mixture.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import qmc

from ruip.priors.common import HistoricalSummary, validated_histories

BASE_VARIANCE = 10.0
BASE_PRECISION = 1.0 / BASE_VARIANCE


def normalized_power_components(
    histories: Iterable[HistoricalSummary],
    *,
    order: int = 16,
    qmc_power: int = 14,
    seed: int = 20260927,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return conditional Normal means, variances, and prior mixture weights.

    Tensor Gauss-Legendre integrates up to three powers. Higher dimensions use
    a fixed scrambled Sobol rule to avoid exponential tensor growth.
    """
    values = validated_histories(histories)
    if order < 2 or qmc_power < 1:
        raise ValueError("power integration order and QMC power are too small")
    information = np.array([x.n_k * x.I_Uk for x in values], dtype=float)
    estimate = np.array([x.h_k for x in values], dtype=float)
    k = len(values)
    if k <= 3:
        nodes, weights = leggauss(order)
        nodes, weights = (nodes + 1.0) / 2.0, weights / 2.0
        powers = np.stack(np.meshgrid(*([nodes] * k), indexing="ij"), axis=-1).reshape(
            -1, k
        )
        weight = np.prod(
            np.stack(np.meshgrid(*([weights] * k), indexing="ij"), axis=-1), axis=-1
        ).ravel()
        rule = "tensor_Gauss_Legendre"
    else:
        powers = qmc.Sobol(d=k, scramble=True, seed=seed).random_base2(qmc_power)
        weight = np.full(len(powers), 1.0 / len(powers))
        rule = "scrambled_Sobol"
    precision = BASE_PRECISION + powers @ information
    mean = (powers @ (information * estimate)) / precision
    variance = 1.0 / precision
    diagnostics = {
        "power_prior": "normalized_Gaussian_summary",
        "power_distribution": "independent_Beta(1,1)",
        "base_mean": 0.0,
        "base_variance": BASE_VARIANCE,
        "integration_rule": rule,
        "integration_nodes": len(weight),
        "integration_order": order if k <= 3 else None,
        "integration_qmc_power": qmc_power if k > 3 else None,
        "integration_seed": seed if k > 3 else None,
    }
    return mean, variance, weight, diagnostics


def log_mixture_density(
    x: np.ndarray, mean: np.ndarray, variance: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Evaluate a Normal-mixture density in bounded-memory chunks."""
    points = np.asarray(x, dtype=float)
    result = np.empty_like(points)
    log_constant = np.log(weight) - 0.5 * np.log(2.0 * np.pi * variance)
    for start in range(0, len(points), 64):
        part = points[start : start + 64, None]
        result[start : start + 64] = logsumexp(
            log_constant[None, :]
            - 0.5 * (part - mean[None, :]) ** 2 / variance[None, :],
            axis=1,
        )
    return result
