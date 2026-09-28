"""Exact-Binomial analyses on the logit scale for the M5 binary engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from math import exp, isfinite, lgamma, log, pi, sqrt
from numbers import Integral

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import (
    betainc,
    digamma,
    expit,
    logsumexp,
    polygamma,
    roots_hermitenorm,
    roots_legendre,
)

from ruip.priors import (
    HistoricalSummary,
    commensurate_prior,
    power_prior,
    rmap_prior,
    robust_uip_prior,
)
from ruip.priors.rmap import rmap_conditional_predictive

_LOG_2PI = log(2.0 * pi)
_AGHQ_ORDER = 16
_AGHQ_CHECK_ORDER = 12


class ImproperFlatLogitPosterior(RuntimeError):
    """A flat-logit Binomial posterior cannot be normalized."""


@dataclass(frozen=True, slots=True)
class BinomialSummary:
    """Event count and sample size for one current arm."""

    n: int
    events: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.n, bool)
            or isinstance(self.events, bool)
            or not isinstance(self.n, Integral)
            or not isinstance(self.events, Integral)
            or self.n < 1
            or not 0 <= self.events <= self.n
        ):
            raise ValueError("events must be integers with 0 <= events <= n and n >= 1")

    @property
    def proportion(self) -> float:
        return self.events / self.n


@dataclass(frozen=True, slots=True)
class BinaryPosteriorResult:
    """Explicit posterior reporting contract for all binary methods."""

    method: str
    eta_c_mean: float
    eta_c_var: float
    log_or_mean: float
    log_or_var: float
    probability_log_or_positive: float
    log_or_q025: float
    log_or_q975: float
    p_c_mean: float
    p_t_mean: float
    risk_difference_mean: float
    risk_difference_q025: float
    risk_difference_q975: float
    numerical_flag: str = "ok"
    diagnostics: dict[str, float | list[float] | str] = field(default_factory=dict)

    @property
    def p_log_or_positive(self) -> float:
        return self.probability_log_or_positive

    @property
    def rd_mean(self) -> float:
        return self.risk_difference_mean

    @property
    def rd_q025(self) -> float:
        return self.risk_difference_q025

    @property
    def rd_q975(self) -> float:
        return self.risk_difference_q975


def historical_logit_summary(source_id: str, events: int, n: int) -> HistoricalSummary:
    """Create the frozen 0.5-corrected historical logit summary."""
    data = BinomialSummary(n=n, events=events)
    p = (data.events + 0.5) / (data.n + 1.0)
    return HistoricalSummary(source_id, log(p / (1.0 - p)), data.n, p * (1.0 - p))


def log_binomial_likelihood(theta: float | np.ndarray, data: BinomialSummary):
    """Full exact Binomial log likelihood, evaluated from logits."""
    constant = (
        lgamma(data.n + 1) - lgamma(data.events + 1) - lgamma(data.n - data.events + 1)
    )
    return (
        constant + data.events * np.asarray(theta) - data.n * np.logaddexp(0.0, theta)
    )


class NormalizedPosterior1D:
    """Mode-centered normalized 1D posterior using adaptive quadrature."""

    def __init__(
        self,
        log_kernel: Callable[[float], float],
        *,
        centers: Iterable[float] = (0.0,),
        epsabs: float = 2e-10,
        epsrel: float = 2e-10,
    ) -> None:
        values = tuple(float(x) for x in centers)
        if not values or not all(isfinite(x) for x in values):
            raise ValueError("centers must be non-empty and finite")
        width = max(2.0, max(values) - min(values))
        optimum = minimize_scalar(
            lambda x: -float(log_kernel(float(x))),
            bounds=(min(values) - 12 * width, max(values) + 12 * width),
            method="bounded",
            options={"xatol": 1e-11},
        )
        if not optimum.success or not isfinite(float(optimum.fun)):
            raise RuntimeError("posterior mode optimization failed")
        self.mode = float(optimum.x)
        self.log_peak = float(log_kernel(self.mode))
        self._epsabs, self._epsrel = epsabs, epsrel

        def scaled(x: float) -> float:
            value = float(log_kernel(x)) - self.log_peak
            return exp(value) if value > -745.0 else 0.0

        self._scaled = scaled
        left, left_error = quad(
            scaled, -np.inf, self.mode, epsabs=epsabs, epsrel=epsrel, limit=350
        )
        right, right_error = quad(
            scaled, self.mode, np.inf, epsabs=epsabs, epsrel=epsrel, limit=350
        )
        self._normalizer = float(left + right)
        self.integration_error = float(left_error + right_error)
        if (
            not isfinite(self._normalizer)
            or self._normalizer <= 0.0
            or self.integration_error > max(2e-7, 2e-7 * self._normalizer)
        ):
            raise RuntimeError("posterior normalization quadrature failed")
        self.log_normalizer = self.log_peak + log(self._normalizer)
        left_bound, right_bound = self.mode - 2.0, self.mode + 2.0
        while scaled(left_bound) > 1e-13:
            left_bound -= max(2.0, self.mode - left_bound)
        while scaled(right_bound) > 1e-13:
            right_bound += max(2.0, right_bound - self.mode)
        raw_nodes, raw_weights = roots_legendre(128)
        half_width = 0.5 * (right_bound - left_bound)
        midpoint = 0.5 * (right_bound + left_bound)
        self._nodes = midpoint + half_width * raw_nodes
        node_mass = (
            half_width
            * raw_weights
            * np.asarray([scaled(float(x)) for x in self._nodes])
        )
        self._masses = node_mass / node_mass.sum()

    def density(self, x: float) -> float:
        return self._scaled(float(x)) / self._normalizer

    def expectation(self, function: Callable) -> float:
        values = np.asarray([function(float(x)) for x in self._nodes], dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("posterior expectation integrand is not finite")
        return float(np.dot(self._masses, values))

    @property
    def mean(self) -> float:
        return self.expectation(lambda x: x)

    @property
    def variance(self) -> float:
        mean = self.mean
        return self.expectation(lambda x: (x - mean) ** 2)

    def cdf(self, x: float) -> float:
        value, error = quad(
            self._scaled, -np.inf, float(x), epsabs=2e-9, epsrel=2e-9, limit=350
        )
        if error > max(2e-7, 2e-7 * self._normalizer):
            raise RuntimeError("posterior CDF quadrature failed")
        return float(np.clip(value / self._normalizer, 0.0, 1.0))

    def quantile(self, probability: float) -> float:
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly between zero and one")
        span = 2.0
        while self.cdf(self.mode - span) > probability:
            span *= 2.0
        while self.cdf(self.mode + span) < probability:
            span *= 2.0
        return float(
            brentq(
                lambda x: self.cdf(x) - probability,
                self.mode - span,
                self.mode + span,
            )
        )


@dataclass(slots=True)
class _NodePosterior:
    nodes: np.ndarray
    masses: np.ndarray
    log_evidence: float
    integration_error: float

    def expectation(self, function: Callable) -> float:
        return float(np.dot(self.masses, np.asarray(function(self.nodes), dtype=float)))

    @property
    def mean(self) -> float:
        return float(np.dot(self.masses, self.nodes))

    @property
    def variance(self) -> float:
        mean = self.mean
        return float(np.dot(self.masses, (self.nodes - mean) ** 2))


def _empty_result(method: str, status: str, diagnostics=None) -> BinaryPosteriorResult:
    nan = float("nan")
    return BinaryPosteriorResult(
        method, *(nan for _ in range(12)), status, diagnostics or {}
    )


def _flat_control_posterior(control: BinomialSummary) -> _NodePosterior:
    if control.events in (0, control.n):
        raise ImproperFlatLogitPosterior("flat-logit posterior is proper iff 0 < y < n")
    a, b = float(control.events), float(control.n - control.events)
    nodes, weights = roots_hermitenorm(_AGHQ_ORDER)
    mode = log(a / b)
    scale = 1.0 / sqrt(control.n * expit(mode) * (1.0 - expit(mode)))
    theta = mode + scale * nodes
    log_mass = (
        np.log(weights)
        + np.asarray(log_binomial_likelihood(theta, control))
        + 0.5 * nodes * nodes
        + log(scale)
    )
    log_z = float(logsumexp(log_mass))
    return _NodePosterior(theta, np.exp(log_mass - log_z), log_z, 0.0)


def _aghq_mixture(
    control: BinomialSummary,
    means: np.ndarray,
    variances: np.ndarray,
    prior_weights: np.ndarray,
    diagnostics: list[dict[str, float]],
) -> tuple[_NodePosterior, dict[str, float | list[float] | str]]:
    """Update every Normal component with exact likelihood using vectorized AGHQ."""
    if control.events in (0, control.n):
        raise ValueError("endpoint controls use adaptive aggregate quadrature")
    y, n = float(control.events), float(control.n)
    theta = means.copy()
    for _ in range(40):
        p = expit(theta)
        score = y - n * p - (theta - means) / variances
        curvature = n * p * (1.0 - p) + 1.0 / variances
        step = score / curvature
        theta += step
        if float(np.max(np.abs(step))) < 2e-12:
            break
    scales = 1.0 / np.sqrt(n * expit(theta) * (1.0 - expit(theta)) + 1.0 / variances)

    def evaluate(order: int, keep_nodes: bool):
        roots, weights = roots_hermitenorm(order)
        x = theta[:, None] + scales[:, None] * roots[None, :]
        log_prior = -0.5 * (
            _LOG_2PI
            + np.log(variances)[:, None]
            + (x - means[:, None]) ** 2 / variances[:, None]
        )
        log_terms = (
            np.log(weights)[None, :]
            + np.asarray(log_binomial_likelihood(x, control))
            + log_prior
            + 0.5 * roots[None, :] ** 2
            + np.log(scales)[:, None]
        )
        component_log_evidence = logsumexp(log_terms, axis=1)
        total = float(logsumexp(np.log(prior_weights) + component_log_evidence))
        if not keep_nodes:
            return total, None, None, component_log_evidence
        return (
            total,
            x.ravel(),
            (np.log(prior_weights)[:, None] + log_terms).ravel(),
            component_log_evidence,
        )

    log_z, nodes, log_masses, component_log_z = evaluate(_AGHQ_ORDER, True)
    check_log_z, _, _, _ = evaluate(_AGHQ_CHECK_ORDER, False)
    error = abs(exp(min(0.0, check_log_z - log_z)) - exp(min(0.0, log_z - check_log_z)))
    masses = np.exp(log_masses - log_z)
    masses /= masses.sum()
    posterior_component_weights = np.exp(
        np.log(prior_weights) + component_log_z - log_z
    )
    result_diagnostics: dict[str, float | list[float] | str] = {
        "component_count": float(len(means)),
        "quadrature_order": float(_AGHQ_ORDER),
        "quadrature_check_order": float(_AGHQ_CHECK_ORDER),
        "quadrature_evidence_relative_difference": float(error),
        "integration_method": "adaptive_gauss_hermite_exact_likelihood",
    }
    for key in set().union(*(item.keys() for item in diagnostics)):
        result_diagnostics[key] = float(
            np.dot(
                posterior_component_weights,
                [item.get(key, 0.0) for item in diagnostics],
            )
        )
    return _NodePosterior(nodes, masses, log_z, float(error)), result_diagnostics


def _component_arrays(
    method: str,
    histories: tuple[HistoricalSummary, ...],
    planned_n_control: float | None,
    delta_clin: float | None = None,
):
    means, variances, weights, diagnostics = [], [], [], []
    if method == "rmap":
        prior = rmap_prior(histories)
        for tau, log_weight in zip(prior.tau_nodes, prior.map_log_weights, strict=True):
            mean, variance = rmap_conditional_predictive(histories, tau=tau)
            means.append(mean)
            variances.append(variance)
            weights.append((1.0 - prior.robust_weight) * exp(log_weight))
            diagnostics.append({"tau": tau})
        means.append(prior.vague_mean)
        variances.append(1.0 / prior.vague_precision)
        weights.append(prior.robust_weight)
        diagnostics.append({"robust_component": 1.0})
    elif method == "power_prior":
        prior = power_prior(histories)
        means.append(prior.mean)
        variances.append(prior.variance)
        weights.append(1.0)
        diagnostics.append({"a0": 0.5})
    elif method == "ruip":
        if delta_clin is None:
            raise ValueError("ruip requires delta_clin")
        prior = robust_uip_prior(histories, delta_clin=delta_clin)
        means.append(prior.mean)
        variances.append(prior.variance)
        weights.append(1.0)
        detail = {
            "B0": prior.precision,
            "V_L": prior.local_variance,
            "rho_G": prior.global_discount,
            "M0": prior.effective_information_units,
        }
        detail.update({f"q_{i + 1}": value for i, value in enumerate(prior.weights)})
        diagnostics.append(detail)
    elif method == "standard_uip":
        if planned_n_control is None:
            raise ValueError("standard_uip requires planned_n_control")
        from ruip.outcomes.continuous import _standard_uip_components

        parts = _standard_uip_components(histories, planned_n_control)
        for part in parts:
            means.append(part.mean)
            variances.append(part.variance)
            weights.append(part.weight)
            diagnostics.append(part.diagnostics)
    else:
        raise ValueError(f"method {method!r} has no Normal mixture")
    mass = np.asarray(weights, dtype=float)
    mass /= mass.sum()
    return np.asarray(means), np.asarray(variances), mass, diagnostics


def _adaptive_control(
    method: str,
    control: BinomialSummary,
    histories: tuple[HistoricalSummary, ...],
    planned_n_control: float | None,
    delta_clin: float | None = None,
):
    current_center = log((control.events + 0.5) / (control.n - control.events + 0.5))
    if method == "nip":
        posterior = NormalizedPosterior1D(
            lambda x: float(log_binomial_likelihood(x, control)),
            centers=(current_center,),
        )
        return posterior, {
            "integration_method": "adaptive_quadrature_exact_likelihood",
            "integration_error": posterior.integration_error,
        }
    if method == "commensurate":
        prior = commensurate_prior(histories)
        posterior = NormalizedPosterior1D(
            lambda x: prior.log_density(x) + float(log_binomial_likelihood(x, control)),
            centers=(current_center, *(h.h_k for h in histories)),
        )
        return posterior, {
            "integration_method": "adaptive_quadrature_exact_likelihood",
            "integration_error": posterior.integration_error,
            "stored_node_count": float(prior.stored_node_count),
        }
    means, variances, weights, detail = _component_arrays(
        method, histories, planned_n_control, delta_clin
    )
    log_norm = -0.5 * (_LOG_2PI + np.log(variances))

    def log_prior(x: float) -> float:
        return float(
            logsumexp(np.log(weights) + log_norm - 0.5 * (x - means) ** 2 / variances)
        )

    posterior = NormalizedPosterior1D(
        lambda x: log_prior(x) + float(log_binomial_likelihood(x, control)),
        centers=(current_center, *(h.h_k for h in histories)),
    )
    # Endpoint datasets are rare. Integrate component evidences directly so
    # posterior mixture diagnostics remain likelihood-reweighted.
    component_log_z = []
    for mean, variance in zip(means, variances, strict=True):
        normal_log = -0.5 * (_LOG_2PI + log(variance))
        integral = NormalizedPosterior1D(
            lambda x, m=mean, v=variance, c=normal_log: (
                c - 0.5 * (x - m) ** 2 / v + float(log_binomial_likelihood(x, control))
            ),
            centers=(mean, current_center),
        )
        component_log_z.append(integral.log_normalizer)
    post_weights = np.exp(
        np.log(weights)
        + np.asarray(component_log_z)
        - logsumexp(np.log(weights) + component_log_z)
    )
    diag: dict[str, float | list[float] | str] = {
        "component_count": float(len(means)),
        "integration_method": "adaptive_quadrature_exact_likelihood_endpoint",
        "integration_error": posterior.integration_error,
    }
    for key in set().union(*(item.keys() for item in detail)):
        diag[key] = float(np.dot(post_weights, [item.get(key, 0.0) for item in detail]))
    return posterior, diag


def _summarize(
    method: str,
    posterior,
    treatment: BinomialSummary,
    diagnostics: dict[str, float | list[float] | str],
) -> BinaryPosteriorResult:
    eta_c_mean = posterior.mean
    eta_c_var = posterior.variance
    p_c_mean = posterior.expectation(expit)
    if treatment.events in (0, treatment.n):
        nan = float("nan")
        return BinaryPosteriorResult(
            method,
            eta_c_mean,
            eta_c_var,
            *(nan for _ in range(10)),
            "treatment_endpoint_effect_unavailable",
            diagnostics,
        )
    a, b = float(treatment.events), float(treatment.n - treatment.events)
    eta_t_mean = float(digamma(a) - digamma(b))
    eta_t_var = float(polygamma(1, a) + polygamma(1, b))
    log_or_mean = eta_t_mean - eta_c_mean
    log_or_var = eta_t_var + eta_c_var

    def log_or_cdf(value: float) -> float:
        return posterior.expectation(lambda x: betainc(a, b, expit(x + value)))

    probability = float(np.clip(1.0 - log_or_cdf(0.0), 0.0, 1.0))
    log_or_q025 = float(brentq(lambda x: log_or_cdf(x) - 0.025, -40.0, 40.0))
    log_or_q975 = float(brentq(lambda x: log_or_cdf(x) - 0.975, -40.0, 40.0))
    p_t_mean = a / (a + b)

    def rd_cdf(value: float) -> float:
        return posterior.expectation(
            lambda x: betainc(a, b, np.clip(expit(x) + value, 0.0, 1.0))
        )

    rd_q025 = float(brentq(lambda x: rd_cdf(x) - 0.025, -1.0, 1.0))
    rd_q975 = float(brentq(lambda x: rd_cdf(x) - 0.975, -1.0, 1.0))
    numerical_flag = "ok"
    error = float(diagnostics.get("quadrature_evidence_relative_difference", 0.0))
    if error > 5e-5:
        numerical_flag = "quadrature_accuracy_warning"
    return BinaryPosteriorResult(
        method,
        eta_c_mean,
        eta_c_var,
        log_or_mean,
        log_or_var,
        probability,
        log_or_q025,
        log_or_q975,
        p_c_mean,
        p_t_mean,
        p_t_mean - p_c_mean,
        rd_q025,
        rd_q975,
        numerical_flag,
        diagnostics,
    )


def analyze_binary(
    method: str,
    histories: Iterable[HistoricalSummary],
    control: BinomialSummary,
    treatment: BinomialSummary,
    *,
    planned_n_control: float | None = None,
    delta_clin: float | None = None,
) -> BinaryPosteriorResult:
    """Analyze one shared binary dataset using a prespecified method."""
    name = method.lower()
    if name not in {
        "nip",
        "rmap",
        "power_prior",
        "ruip",
        "commensurate",
        "standard_uip",
    }:
        raise ValueError(f"unknown binary method: {method}")
    history = tuple(histories)
    if name == "nip" and control.events in (0, control.n):
        return _empty_result(
            name, "improper_flat_logit_posterior", {"posterior_proper": 0.0}
        )
    if name in {"nip", "commensurate"} or control.events in (0, control.n):
        posterior, diagnostics = _adaptive_control(
            name, control, history, planned_n_control, delta_clin
        )
    else:
        means, variances, weights, detail = _component_arrays(
            name, history, planned_n_control, delta_clin
        )
        posterior, diagnostics = _aghq_mixture(
            control, means, variances, weights, detail
        )
    diagnostics["posterior_proper"] = 1.0
    return _summarize(name, posterior, treatment, diagnostics)


__all__ = [
    "BinaryPosteriorResult",
    "BinomialSummary",
    "ImproperFlatLogitPosterior",
    "NormalizedPosterior1D",
    "analyze_binary",
    "historical_logit_summary",
    "log_binomial_likelihood",
]
