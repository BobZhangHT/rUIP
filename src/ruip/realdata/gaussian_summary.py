"""History-only prior and posterior calculations for Gaussian rate summaries."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr

from ruip.analysis.scalable_comparators import uip_components
from ruip.priors import HistoricalSummary, commensurate_prior, rmap_prior
from ruip.priors.rmap import rmap_conditional_predictive
from ruip.priors.ruip import robust_uip_prior
from ruip.realdata.normalized_power import normalized_power_components

METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP")


@dataclass(frozen=True, slots=True)
class GaussianSummary:
    """One log-hazard MLE with its exponential-model Gaussian variance."""

    source_id: str
    events: int
    exposure: float

    @property
    def estimate(self) -> float:
        return math.log(self.events / self.exposure)

    @property
    def variance(self) -> float:
        return 1.0 / self.events

    def historical(self) -> HistoricalSummary:
        return HistoricalSummary(self.source_id, self.estimate, self.events, 1.0)


@dataclass(frozen=True, slots=True)
class GaussianPosteriorResult:
    method: str
    status: str
    estimate: float
    posterior_sd: float
    interval_lower: float
    interval_upper: float
    favorable_probability: float
    prior_predictive_log_density: float | None
    diagnostics_json: str

    def row(self) -> dict[str, object]:
        return asdict(self)


def _normal_components(mean, variance, weight=1.0):
    means = np.atleast_1d(np.asarray(mean, dtype=float))
    variances = np.atleast_1d(np.asarray(variance, dtype=float))
    weights = np.broadcast_to(np.asarray(weight, dtype=float), means.shape).copy()
    if means.shape != variances.shape or np.any(variances <= 0) or np.any(weights < 0):
        raise ValueError("invalid Normal-mixture components")
    weights /= weights.sum()
    return means, variances, weights


def _update_mixture(mean, variance, weight, current: GaussianSummary):
    y, v = current.estimate, current.variance
    log_weight = np.log(weight) - 0.5 * (
        math.log(2.0 * math.pi)
        + np.log(variance + v)
        + (y - mean) ** 2 / (variance + v)
    )
    log_evidence = float(logsumexp(log_weight))
    posterior_weight = np.exp(log_weight - log_evidence)
    posterior_variance = 1.0 / (1.0 / variance + 1.0 / v)
    posterior_mean = posterior_variance * (mean / variance + y / v)
    return posterior_mean, posterior_variance, posterior_weight, log_evidence


def _mixture_summary(method, mean, variance, weight, current, diagnostics):
    mean, variance, weight, log_evidence = _update_mixture(
        mean, variance, weight, current
    )
    estimate = float(np.dot(weight, mean))
    second = float(np.dot(weight, variance + mean * mean))
    posterior_sd = math.sqrt(max(0.0, second - estimate * estimate))

    def cdf(x):
        return float(np.dot(weight, ndtr((x - mean) / np.sqrt(variance))))

    scale = max(
        1.0, float(np.max(np.abs(mean))) + 12.0 * float(np.max(np.sqrt(variance)))
    )
    lower = float(brentq(lambda x: cdf(x) - 0.025, -scale, scale, xtol=2e-10))
    upper = float(brentq(lambda x: cdf(x) - 0.975, -scale, scale, xtol=2e-10))
    return GaussianPosteriorResult(
        method,
        "success",
        estimate,
        posterior_sd,
        lower,
        upper,
        float(np.clip(1.0 - cdf(0.0), 0.0, 1.0)),
        log_evidence,
        json.dumps(diagnostics, sort_keys=True),
    )


def freeze_priors(
    histories: Iterable[HistoricalSummary],
    *,
    delta_clin: float,
    planned_events: int,
    uip_scrambles: int = 4,
    uip_power: int = 13,
    uip_seed: int = 202609188301,
    pp_power: int = 14,
):
    """Build all informative priors before any current outcome is supplied."""
    values = tuple(histories)
    matrix = np.asarray([(x.n_k, x.h_k, x.I_Uk) for x in values], dtype=float)
    pp_mean, pp_variance, pp_weight, pp_diagnostics = normalized_power_components(
        values, qmc_power=pp_power
    )
    rmap = rmap_prior(values)
    rmap_means, rmap_variances, rmap_weights = [], [], []
    for tau, log_weight in zip(rmap.tau_nodes, rmap.map_log_weights, strict=True):
        mean, variance = rmap_conditional_predictive(values, tau=tau)
        rmap_means.append(mean)
        rmap_variances.append(variance)
        rmap_weights.append((1.0 - rmap.robust_weight) * math.exp(log_weight))
    rmap_means.append(rmap.vague_mean)
    rmap_variances.append(1.0 / rmap.vague_precision)
    rmap_weights.append(rmap.robust_weight)
    cp = commensurate_prior(values)
    uip_means, uip_variances = [], []
    for scramble in range(uip_scrambles):
        mean, variance = uip_components(
            matrix, planned_events, uip_power, uip_seed + 100003 * scramble
        )
        uip_means.append(mean)
        uip_variances.append(variance)
    ruip = robust_uip_prior(values, delta_clin=delta_clin)
    return {
        "PP": (*_normal_components(pp_mean, pp_variance, pp_weight), pp_diagnostics),
        "rMAP": (
            *_normal_components(rmap_means, rmap_variances, rmap_weights),
            {"tau_nodes": len(rmap.tau_nodes), "robust_weight": rmap.robust_weight},
        ),
        "CP": (
            *_normal_components(
                np.full(cp.stored_node_count, cp.historical_mean),
                cp.component_variances,
                cp.weights,
            ),
            {"kappa_nodes": cp.stored_node_count, "classical_pooled": True},
        ),
        "UIP": (
            *_normal_components(
                np.concatenate(uip_means), np.concatenate(uip_variances), 1.0
            ),
            {
                "scrambles": uip_scrambles,
                "power": uip_power,
                "draws_per_scramble": 2**uip_power,
                "planned_events": planned_events,
            },
        ),
        "rUIP": (
            *_normal_components(ruip.mean, ruip.variance),
            {
                "delta_clin": delta_clin,
                "local_variance": ruip.local_variance,
                "global_discount": ruip.global_discount,
                "effective_information_units": ruip.effective_information_units,
            },
        ),
    }, ruip


def analyze_gaussian_summary(
    histories: Iterable[HistoricalSummary],
    current: GaussianSummary,
    *,
    delta_clin: float,
    planned_events: int,
    uip_scrambles: int = 4,
    uip_power: int = 13,
    uip_seed: int = 202609188301,
    pp_power: int = 14,
) -> tuple[list[GaussianPosteriorResult], object]:
    """Analyze a current log-hazard summary using frozen Gaussian priors."""
    values = tuple(histories)
    priors, ruip = freeze_priors(
        values,
        delta_clin=delta_clin,
        planned_events=planned_events,
        uip_scrambles=uip_scrambles,
        uip_power=uip_power,
        uip_seed=uip_seed,
        pp_power=pp_power,
    )
    nip = GaussianPosteriorResult(
        "NIP",
        "success",
        current.estimate,
        math.sqrt(current.variance),
        current.estimate - 1.95996398454 * math.sqrt(current.variance),
        current.estimate + 1.95996398454 * math.sqrt(current.variance),
        float(ndtr(current.estimate / math.sqrt(current.variance))),
        None,
        json.dumps({"prior": "flat"}, sort_keys=True),
    )
    rows = [nip]
    for method in METHODS[1:]:
        mean, variance, weight, diagnostics = priors[method]
        rows.append(
            _mixture_summary(method, mean, variance, weight, current, diagnostics)
        )
    return rows, ruip


def validate_gaussian_rows(rows: Iterable[GaussianPosteriorResult]) -> None:
    values = tuple(rows)
    if not values:
        raise ValueError("no Gaussian-summary rows")
    for row in values:
        if row.status != "success" or not all(
            math.isfinite(x)
            for x in (
                row.estimate,
                row.posterior_sd,
                row.interval_lower,
                row.interval_upper,
                row.favorable_probability,
            )
        ):
            raise ValueError(f"invalid Gaussian-summary row: {row.method}")
        if (
            row.posterior_sd < 0
            or row.interval_lower > row.interval_upper
            or not 0 <= row.favorable_probability <= 1
            or (
                row.prior_predictive_log_density is not None
                and not math.isfinite(row.prior_predictive_log_density)
            )
        ):
            raise ValueError(f"invalid Gaussian-summary row: {row.method}")
