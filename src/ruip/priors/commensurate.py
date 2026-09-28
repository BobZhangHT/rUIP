"""Canonical pooled-history commensurate prior for Gaussian summaries.

The public CP comparator first pools historical summaries by Fisher information,
then places one Hobbs-style commensurability link between the pooled historical
parameter and the current parameter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import log, pi

import numpy as np
from scipy.special import logsumexp, ndtr

from .common import HistoricalSummary, validated_histories

UNIFORM_WEIGHT = 0.99
SPIKE_WEIGHT = 0.01
KAPPA_STAR_LOWER = 0.005
KAPPA_STAR_UPPER = 2.0
KAPPA_STAR_SPIKE = 200.0
_QUADRATURE_ORDER = 80
_LOG2PI = log(2 * pi)


@dataclass(frozen=True, slots=True)
class CommensuratePrior:
    """Mixture induced by one link to the pooled historical parameter."""

    histories: tuple[HistoricalSummary, ...]
    historical_mean: float
    historical_variance: float
    kappas: tuple[float, ...]
    kappa_stars: tuple[float, ...]
    weights: tuple[float, ...]
    reference_sd: float

    @property
    def stored_node_count(self) -> int:
        return len(self.kappas)

    @property
    def component_variances(self) -> np.ndarray:
        return self.historical_variance + 1.0 / np.asarray(self.kappas)

    def log_density(self, theta: float) -> float:
        variances = self.component_variances
        logn = -0.5 * (
            _LOG2PI
            + np.log(variances)
            + (float(theta) - self.historical_mean) ** 2 / variances
        )
        return float(logsumexp(np.log(self.weights) + logn))

    def density(self, theta: float) -> float:
        return float(np.exp(self.log_density(theta)))

    def cdf(self, theta: float) -> float:
        z = (float(theta) - self.historical_mean) / np.sqrt(self.component_variances)
        return float(np.dot(np.asarray(self.weights), ndtr(z)))


def _nodes(
    reference_sd: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    x, w = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
    kappa_stars = 0.5 * (KAPPA_STAR_UPPER - KAPPA_STAR_LOWER) * x + 0.5 * (
        KAPPA_STAR_UPPER + KAPPA_STAR_LOWER
    )
    weights = UNIFORM_WEIGHT * 0.5 * w
    kappa_stars = tuple(float(v) for v in kappa_stars) + (KAPPA_STAR_SPIKE,)
    kappas = tuple(value / reference_sd**2 for value in kappa_stars)
    return kappas, kappa_stars, tuple(float(v) for v in weights) + (SPIKE_WEIGHT,)


def commensurate_prior(histories: Iterable[HistoricalSummary]) -> CommensuratePrior:
    """Construct the canonical pooled CP used in the paper and frozen engine."""
    values = validated_histories(histories)
    source_information = np.asarray([item.n_k * item.I_Uk for item in values])
    total_information = float(source_information.sum())
    historical_mean = float(
        np.dot(source_information, [item.h_k for item in values]) / total_information
    )
    historical_variance = 1.0 / total_information
    reference_variance = sum(item.n_k for item in values) / total_information
    reference_sd = float(np.sqrt(reference_variance))
    kappas, kappa_stars, weights = _nodes(reference_sd)
    return CommensuratePrior(
        histories=values,
        historical_mean=historical_mean,
        historical_variance=historical_variance,
        kappas=kappas,
        kappa_stars=kappa_stars,
        weights=weights,
        reference_sd=reference_sd,
    )
