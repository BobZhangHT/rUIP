"""Gaussian continuous-outcome analyses for the frozen M2 priors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import exp, isfinite, log, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp, ndtr, roots_jacobi

from ruip.priors import (
    CommensuratePrior,
    DesignStageRUIPPrior,
    HistoricalSummary,
    RMAPrior,
    commensurate_prior,
    no_information_prior,
    power_prior,
    rmap_prior,
    robust_uip_prior,
    standard_uip_hyperparameters,
)

_LOG_2PI = log(2.0 * pi)
_UIP_WEIGHT_ORDER = 24
_UIP_M_NODES = 32


@dataclass(frozen=True, slots=True)
class CurrentGroupSummary:
    """Immutable Normal-summary data for one current-trial arm."""

    n: int
    mean: float
    known_variance: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or int(self.n) != self.n or self.n < 1:
            raise ValueError("n must be a positive integer")
        if not isfinite(float(self.mean)):
            raise ValueError("mean must be finite")
        if not isfinite(float(self.known_variance)) or self.known_variance <= 0:
            raise ValueError("known_variance must be finite and positive")

    @property
    def variance_of_mean(self) -> float:
        return float(self.known_variance) / self.n


@dataclass(frozen=True, slots=True)
class PosteriorResult:
    """Common posterior reporting contract for all continuous methods."""

    method: str
    mu_c_mean: float
    mu_c_var: float
    delta_mean: float
    delta_var: float
    probability_delta_positive: float
    delta_q025: float
    delta_q975: float
    numerical_flag: str = "ok"
    diagnostics: dict[str, float | list[float]] = field(default_factory=dict)

    @property
    def p_delta_positive(self) -> float:
        return self.probability_delta_positive

    @property
    def q025(self) -> float:
        return self.delta_q025

    @property
    def q975(self) -> float:
        return self.delta_q975


@dataclass(frozen=True, slots=True)
class _NormalComponent:
    mean: float
    variance: float
    weight: float
    diagnostics: dict[str, float]


def _normal_logpdf(x: float, mean: float, variance: float) -> float:
    return -0.5 * (_LOG_2PI + log(variance) + (x - mean) ** 2 / variance)


def _update_component(
    component: _NormalComponent, current: CurrentGroupSummary
) -> tuple[float, float, float]:
    v_data = current.variance_of_mean
    precision = 1.0 / component.variance + 1.0 / v_data
    variance = 1.0 / precision
    mean = variance * (component.mean / component.variance + current.mean / v_data)
    return (
        mean,
        variance,
        _normal_logpdf(current.mean, component.mean, component.variance + v_data),
    )


def _mixture_result(
    method: str,
    components: Iterable[_NormalComponent],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
) -> PosteriorResult:
    values = tuple(components)
    if not values:
        raise ValueError("at least one Normal component is required")
    log_weights = np.asarray([log(x.weight) for x in values])
    updated = [_update_component(item, control) for item in values]
    log_weights += np.asarray([item[2] for item in updated])
    log_weights -= logsumexp(log_weights)
    weights = np.exp(log_weights)
    means = np.asarray([item[0] for item in updated])
    variances = np.asarray([item[1] for item in updated])
    mu_mean = float(np.dot(weights, means))
    mu_var = float(np.dot(weights, variances + (means - mu_mean) ** 2))
    t_mean, t_var = treatment.mean, treatment.variance_of_mean
    delta_means = t_mean - means
    delta_vars = t_var + variances
    delta_mean = float(np.dot(weights, delta_means))
    delta_var = float(np.dot(weights, delta_vars + (delta_means - delta_mean) ** 2))
    probability = float(np.dot(weights, ndtr(delta_means / np.sqrt(delta_vars))))

    def cdf(x: float) -> float:
        return float(np.dot(weights, ndtr((x - delta_means) / np.sqrt(delta_vars))))

    spread = sqrt(max(delta_var, 1e-15))
    lower, upper = delta_mean - 12 * spread, delta_mean + 12 * spread
    while cdf(lower) > 0.025:
        lower -= 12 * spread
    while cdf(upper) < 0.975:
        upper += 12 * spread
    q025 = float(brentq(lambda x: cdf(x) - 0.025, lower, upper))
    q975 = float(brentq(lambda x: cdf(x) - 0.975, lower, upper))
    diagnostics: dict[str, float | list[float]] = {
        "component_count": float(len(values))
    }
    for key in set().union(*(item.diagnostics for item in values)):
        diagnostics[key] = float(
            np.dot(weights, [item.diagnostics.get(key, 0.0) for item in values])
        )
    return PosteriorResult(
        method,
        mu_mean,
        mu_var,
        delta_mean,
        delta_var,
        probability,
        q025,
        q975,
        diagnostics=diagnostics,
    )


def nip_posterior(
    control: CurrentGroupSummary, treatment: CurrentGroupSummary
) -> PosteriorResult:
    """Flat-prior closed-form analysis (the NIP comparator)."""
    _validate_current_variance(control, treatment)
    no_information_prior()
    d_mean = treatment.mean - control.mean
    d_var = treatment.variance_of_mean + control.variance_of_mean
    sd = sqrt(d_var)
    return PosteriorResult(
        "nip",
        control.mean,
        control.variance_of_mean,
        d_mean,
        d_var,
        float(ndtr(d_mean / sd)),
        d_mean - 1.959963984540054 * sd,
        d_mean + 1.959963984540054 * sd,
        diagnostics={"component_count": 1.0},
    )


def rmap_posterior(
    histories: Iterable[HistoricalSummary],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
) -> PosteriorResult:
    _validate_current_variance(control, treatment)
    prior: RMAPrior = rmap_prior(histories)
    from ruip.priors.rmap import rmap_conditional_predictive

    components = []
    for tau, log_weight in zip(prior.tau_nodes, prior.map_log_weights, strict=True):
        mean, variance = rmap_conditional_predictive(prior.histories, tau=tau)
        components.append(
            _NormalComponent(
                mean,
                variance,
                (1.0 - prior.robust_weight) * exp(log_weight),
                {"tau": tau},
            )
        )
    components.append(
        _NormalComponent(
            prior.vague_mean,
            1.0 / prior.vague_precision,
            prior.robust_weight,
            {"robust_component": 1.0},
        )
    )
    return _mixture_result("rmap", components, control, treatment)


def power_prior_posterior(
    histories: Iterable[HistoricalSummary],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
) -> PosteriorResult:
    """Conjugate posterior under the fixed ``a0=0.5`` power prior."""
    _validate_current_variance(control, treatment)
    prior = power_prior(histories)
    return _mixture_result(
        "power_prior",
        (_NormalComponent(prior.mean, prior.variance, 1.0, {"a0": 0.5}),),
        control,
        treatment,
    )


def ruip_posterior(
    histories: Iterable[HistoricalSummary],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
    *,
    delta_clin: float,
) -> PosteriorResult:
    """Conjugate analysis under the scalar design-stage rUIP."""

    _validate_current_variance(control, treatment)
    prior: DesignStageRUIPPrior = robust_uip_prior(
        histories, delta_clin=delta_clin
    )
    diagnostics = {
        "B0": prior.precision,
        "V_L": prior.local_variance,
        "rho_G": prior.global_discount,
        "M0": prior.effective_information_units,
    }
    diagnostics.update(
        {f"q_{index + 1}": value for index, value in enumerate(prior.weights)}
    )
    return _mixture_result(
        "ruip",
        (_NormalComponent(prior.mean, prior.variance, 1.0, diagnostics),),
        control,
        treatment,
    )


def _standard_uip_components(
    histories: tuple[HistoricalSummary, ...],
    planned_n_control: float,
    order: int = _UIP_WEIGHT_ORDER,
) -> list[_NormalComponent]:
    hyper = standard_uip_hyperparameters(histories, planned_n_control=planned_n_control)
    if len(histories) == 1:
        weights = [(np.array([1.0]), 1.0)]
    elif len(histories) == 2:
        x, mass = roots_jacobi(order, hyper.gammas[1] - 1, hyper.gammas[0] - 1)
        mass = mass / mass.sum()
        weights = [
            (np.array([(v + 1) / 2, (1 - v) / 2]), float(p))
            for v, p in zip(x, mass, strict=True)
        ]
    elif len(histories) == 3:
        x1, p1 = roots_jacobi(
            order, hyper.gammas[1] + hyper.gammas[2] - 1, hyper.gammas[0] - 1
        )
        x2, p2 = roots_jacobi(order, hyper.gammas[2] - 1, hyper.gammas[1] - 1)
        weights = [
            (np.array([a, (1 - a) * b, (1 - a) * (1 - b)]), float(pa * pb))
            for a, pa in zip((x1 + 1) / 2, p1 / p1.sum(), strict=True)
            for b, pb in zip((x2 + 1) / 2, p2 / p2.sum(), strict=True)
        ]
    else:
        raise ValueError("standard UIP supports one to three historical studies")
    raw_m, raw_weights = np.polynomial.legendre.leggauss(_UIP_M_NODES)
    u = 0.5 * (raw_m + 1)
    m_values, m_weights = hyper.M_max * u * u, raw_weights * u
    parts = []
    for w, weight_mass in weights:
        mean = float(np.dot(w, [item.h_k for item in histories]))
        unit_information = float(np.dot(w, [item.I_Uk for item in histories]))
        for m, mass in zip(m_values, m_weights, strict=True):
            diagnostics = {"M": float(m)}
            diagnostics.update(
                {f"w_{index + 1}": float(value) for index, value in enumerate(w)}
            )
            parts.append(
                _NormalComponent(
                    mean,
                    1.0 / (m * unit_information),
                    float(mass * weight_mass),
                    diagnostics,
                )
            )
    return parts


def standard_uip_posterior(
    histories: Iterable[HistoricalSummary],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
    *,
    planned_n_control: float,
) -> PosteriorResult:
    _validate_current_variance(control, treatment)
    return _mixture_result(
        "standard_uip",
        _standard_uip_components(tuple(histories), planned_n_control),
        control,
        treatment,
    )


def commensurate_posterior(
    histories: Iterable[HistoricalSummary],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
) -> PosteriorResult:
    """One-dimensional adaptive integration of stable log prior plus likelihood."""
    _validate_current_variance(control, treatment)
    prior: CommensuratePrior = commensurate_prior(histories)
    v = control.variance_of_mean

    def log_kernel(x: float) -> float:
        return prior.log_density(x) + _normal_logpdf(control.mean, x, v)

    centers = [control.mean, *(item.h_k for item in prior.histories)]
    scale = max(1.0, max(centers) - min(centers), sqrt(v))
    optimum = minimize_scalar(
        lambda x: -log_kernel(x),
        bounds=(min(centers) - 12 * scale, max(centers) + 12 * scale),
        method="bounded",
    )
    if not optimum.success:
        raise RuntimeError("commensurate posterior mode optimization failed")
    mode = float(optimum.x)
    log_peak = log_kernel(mode)

    def kernel(x: float) -> float:
        return exp(log_kernel(x) - log_peak)

    normalizer, error = quad(
        kernel, -np.inf, np.inf, epsabs=2e-10, epsrel=2e-10, limit=300
    )
    if error > 1e-7 or not isfinite(normalizer) or normalizer <= 0:
        raise RuntimeError("commensurate posterior integration failed")

    integration_errors: list[float] = [float(error)]

    def expectation(fn):
        value, integration_error = quad(
            lambda x: fn(x) * kernel(x),
            -np.inf,
            np.inf,
            epsabs=2e-9,
            epsrel=2e-9,
            limit=300,
        )
        integration_errors.append(float(integration_error))
        if integration_error > 1e-7:
            raise RuntimeError("commensurate posterior moment integration failed")
        return value / normalizer

    mu_mean = float(expectation(lambda x: x))
    mu_var = float(expectation(lambda x: (x - mu_mean) ** 2))
    d_mean, d_var = treatment.mean - mu_mean, treatment.variance_of_mean + mu_var
    probability = float(
        expectation(
            lambda x: ndtr((treatment.mean - x) / sqrt(treatment.variance_of_mean))
        )
    )

    def delta_cdf(d: float) -> float:
        return float(
            expectation(
                lambda x: ndtr(
                    (d - (treatment.mean - x)) / sqrt(treatment.variance_of_mean)
                )
            )
        )

    spread = sqrt(d_var)
    q025 = brentq(
        lambda x: delta_cdf(x) - 0.025, d_mean - 14 * spread, d_mean + 14 * spread
    )
    q975 = brentq(
        lambda x: delta_cdf(x) - 0.975, d_mean - 14 * spread, d_mean + 14 * spread
    )
    return PosteriorResult(
        "commensurate",
        mu_mean,
        mu_var,
        d_mean,
        d_var,
        probability,
        float(q025),
        float(q975),
        diagnostics={"integration_error": max(integration_errors)},
    )


def analyze_continuous(
    method: str,
    histories: Iterable[HistoricalSummary],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
    *,
    planned_n_control: float | None = None,
    delta_clin: float | None = None,
) -> PosteriorResult:
    """Dispatch one prespecified continuous outcome method."""
    functions = {
        "nip": lambda: nip_posterior(control, treatment),
        "rmap": lambda: rmap_posterior(histories, control, treatment),
        "power_prior": lambda: power_prior_posterior(histories, control, treatment),
        "ruip": lambda: ruip_posterior(
            histories,
            control,
            treatment,
            delta_clin=(
                delta_clin
                if delta_clin is not None
                else (_ for _ in ()).throw(ValueError("ruip requires delta_clin"))
            ),
        ),
        "commensurate": lambda: commensurate_posterior(histories, control, treatment),
        "standard_uip": lambda: standard_uip_posterior(
            histories,
            control,
            treatment,
            planned_n_control=(
                planned_n_control
                if planned_n_control is not None
                else (_ for _ in ()).throw(
                    ValueError("standard_uip requires planned_n_control")
                )
            ),
        ),
    }
    try:
        return functions[method.lower()]()
    except KeyError as error:
        raise ValueError(f"unknown continuous method: {method}") from error


def _validate_current_variance(
    control: CurrentGroupSummary, treatment: CurrentGroupSummary
) -> None:
    if control.known_variance != treatment.known_variance:
        raise ValueError("control and treatment must have the same known_variance")
