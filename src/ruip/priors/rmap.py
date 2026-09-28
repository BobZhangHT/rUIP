"""Frozen robust MAP comparator for Gaussian historical summaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import log, pi, sqrt

import numpy as np
from scipy.special import logsumexp, ndtr

from .common import HistoricalSummary, validated_histories

ROBUST_WEIGHT = 0.20
_QUADRATURE_ORDER = 256
_LOG_2PI = log(2.0 * pi)


@dataclass(frozen=True, slots=True)
class RMAPrior:
    """Robust MAP mixture with an information-scaled heterogeneity prior.

    The MAP component integrates a Gaussian random-effects model over tau.
    With ``I_ref = sum_k(n_k I_Uk) / sum_k n_k`` and
    ``s_ref = I_ref^-1/2``, tau has the HalfNormal(s_ref / 2) distribution.
    The 0.20 robust component is Normal(hbar_J, I_ref^-1), i.e. one reference
    information unit centered at the information-weighted historical mean.
    """

    histories: tuple[HistoricalSummary, ...]
    tau_nodes: tuple[float, ...]
    tau_weights: tuple[float, ...]
    map_log_weights: tuple[float, ...]
    robust_weight: float
    reference_sd: float
    tau_half_normal_scale: float
    vague_mean: float
    vague_precision: float

    def map_density(self, theta: float) -> float:
        """Evaluate the non-robust MAP predictive component."""

        x = float(theta)
        terms = []
        for tau, log_weight in zip(self.tau_nodes, self.map_log_weights, strict=True):
            mean, variance = _conditional_predictive(self.histories, tau)
            terms.append(log_weight + _log_normal_density(x, mean, variance))
        return float(np.exp(logsumexp(terms)))

    def density(self, theta: float) -> float:
        """Evaluate the MAP plus robust-component mixture density."""

        x = float(theta)
        vague_variance = 1.0 / self.vague_precision
        return (1.0 - self.robust_weight) * self.map_density(
            x
        ) + self.robust_weight * float(
            np.exp(_log_normal_density(x, self.vague_mean, vague_variance))
        )

    def cdf(self, theta: float) -> float:
        """Evaluate the robust-mixture CDF from Normal conditional components."""

        x = float(theta)
        map_cdf = 0.0
        for tau, log_weight in zip(self.tau_nodes, self.map_log_weights, strict=True):
            mean, variance = _conditional_predictive(self.histories, tau)
            map_cdf += float(np.exp(log_weight)) * float(
                ndtr((x - mean) / sqrt(variance))
            )
        vague_cdf = float(ndtr((x - self.vague_mean) * sqrt(self.vague_precision)))
        return (1.0 - self.robust_weight) * map_cdf + self.robust_weight * vague_cdf


def _log_normal_density(x: float, mean: float, variance: float) -> float:
    return -0.5 * (_LOG_2PI + log(variance) + (x - mean) ** 2 / variance)


def _conditional_predictive(
    histories: tuple[HistoricalSummary, ...], tau: float
) -> tuple[float, float]:
    variances = np.asarray([1.0 / (item.n_k * item.I_Uk) for item in histories])
    estimates = np.asarray([item.h_k for item in histories])
    precisions = 1.0 / (variances + tau * tau)
    precision_sum = float(np.sum(precisions))
    mean = float(np.sum(precisions * estimates) / precision_sum)
    return mean, tau * tau + 1.0 / precision_sum


def rmap_conditional_predictive(
    histories: Iterable[HistoricalSummary], *, tau: float
) -> tuple[float, float]:
    """Return MAP predictive mean and variance conditional on fixed tau >= 0."""

    values = validated_histories(histories)
    heterogeneity = float(tau)
    if not np.isfinite(heterogeneity) or heterogeneity < 0:
        raise ValueError("tau must be finite and non-negative")
    return _conditional_predictive(values, heterogeneity)


def _log_marginal_likelihood(
    histories: tuple[HistoricalSummary, ...], tau: float
) -> float:
    """log p(h | tau), integrating the flat random-effects location mu."""

    variances = np.asarray([1.0 / (item.n_k * item.I_Uk) for item in histories])
    estimates = np.asarray([item.h_k for item in histories])
    marginal_variances = variances + tau * tau
    precisions = 1.0 / marginal_variances
    precision_sum = float(np.sum(precisions))
    mean = float(np.sum(precisions * estimates) / precision_sum)
    residual = float(np.sum(precisions * (estimates - mean) ** 2))
    return -0.5 * (
        (len(histories) - 1) * _LOG_2PI
        + float(np.sum(np.log(marginal_variances)))
        + log(precision_sum)
        + residual
    )


def _tau_quadrature(scale: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Gauss--Legendre rule under the HalfNormal(scale) CDF transform."""

    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
    probabilities = 0.5 * (raw_nodes + 1.0)
    weights = 0.5 * raw_weights
    tau = sqrt(2.0) * np.erfinv(probabilities) if hasattr(np, "erfinv") else None
    if tau is None:  # NumPy does not expose erfinv on all supported versions.
        from scipy.special import erfinv

        tau = sqrt(2.0) * erfinv(probabilities)
    return tuple(float(scale * value) for value in tau), tuple(
        float(value) for value in weights
    )


def rmap_prior(histories: Iterable[HistoricalSummary]) -> RMAPrior:
    """Construct rMAP with RBesT-style robust weight and unit-info scaling."""

    values = validated_histories(histories)
    J = np.asarray([item.n_k * item.I_Uk for item in values])
    estimates = np.asarray([item.h_k for item in values])
    I_ref = float(np.sum(J) / sum(item.n_k for item in values))
    reference_sd = 1.0 / sqrt(I_ref)
    tau_half_normal_scale = reference_sd / 2.0
    tau_nodes, tau_weights = _tau_quadrature(tau_half_normal_scale)
    log_weights = np.asarray(
        [
            log(weight) + _log_marginal_likelihood(values, tau)
            for tau, weight in zip(tau_nodes, tau_weights, strict=True)
        ]
    )
    log_weights -= logsumexp(log_weights)
    hbar_j = float(np.sum(J * estimates) / np.sum(J))
    return RMAPrior(
        histories=values,
        tau_nodes=tau_nodes,
        tau_weights=tau_weights,
        map_log_weights=tuple(float(value) for value in log_weights),
        robust_weight=ROBUST_WEIGHT,
        reference_sd=reference_sd,
        tau_half_normal_scale=tau_half_normal_scale,
        vague_mean=hbar_j,
        vague_precision=I_ref,
    )
