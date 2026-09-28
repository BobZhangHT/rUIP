"""Classical pooled CP and randomized-QMC UIP clinical calculations.

The multi-source CP first pools the homogeneous historical population and
therefore needs only 81 components for every K. UIP uses scrambled Sobol draws,
the exact Dirichlet stick-breaking transform, and analytic marginalization of
its uniform total-information parameter. Current-data
likelihoods are the Normal likelihood for the continuous application and the
unmodified Binomial likelihood for the binary application.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.special import (
    betainc,
    betaincinv,
    digamma,
    expit,
    gamma,
    gammainc,
    logsumexp,
    ndtr,
    roots_legendre,
)
from scipy.stats import qmc

LOG2PI = math.log(2.0 * math.pi)


def _normal_logpdf(x, mean, variance):
    return -0.5 * (LOG2PI + np.log(variance) + (x - mean) ** 2 / variance)


def _quantile(cdf: Callable[[float], float], probability: float, lo=-60.0, hi=60.0):
    return float(brentq(lambda x: cdf(x) - probability, lo, hi, xtol=2e-10))


def _delta_from_control_nodes(theta, weight, treatment_mean, treatment_variance):
    mean = float(treatment_mean - weight @ theta)
    second = float(treatment_variance + weight @ ((treatment_mean - theta) ** 2))
    sd = math.sqrt(max(0.0, second - mean * mean))
    probability = float(
        weight @ ndtr((treatment_mean - theta) / math.sqrt(treatment_variance))
    )

    def cdf(delta):
        return float(
            weight
            @ ndtr((delta + theta - treatment_mean) / math.sqrt(treatment_variance))
        )

    return {
        "estimate": mean,
        "posterior_sd": sd,
        "interval_lower": _quantile(cdf, 0.025),
        "interval_upper": _quantile(cdf, 0.975),
        "posterior_probability_positive": probability,
    }


def _binary_from_control_nodes(eta, weight, treatment):
    nt, yt = treatment
    a, b = float(yt), float(nt - yt)
    if a <= 0 or b <= 0:
        raise ValueError("flat-logit treatment posterior requires nonendpoint counts")
    treatment_logit_mean = float(digamma(a) - digamma(b))
    estimate = float(treatment_logit_mean - weight @ eta)
    from scipy.special import polygamma

    treatment_logit_var = float(polygamma(1, a) + polygamma(1, b))
    variance = treatment_logit_var + float(weight @ (eta * eta) - (weight @ eta) ** 2)
    probability = float(weight @ (1.0 - betainc(a, b, expit(eta))))

    def cdf(delta):
        return float(weight @ betainc(a, b, expit(delta + eta)))

    pc = expit(eta)
    risk_mean = float(a / (a + b) - weight @ pc)

    def risk_cdf(delta):
        return float(weight @ betainc(a, b, np.clip(pc + delta, 0, 1)))

    return {
        "estimate": estimate,
        "posterior_sd": math.sqrt(max(variance, 0.0)),
        "interval_lower": _quantile(cdf, 0.025),
        "interval_upper": _quantile(cdf, 0.975),
        "posterior_probability_positive": probability,
        "risk_difference_mean": risk_mean,
        "risk_difference_q025": _quantile(risk_cdf, 0.025, -1, 1),
        "risk_difference_q975": _quantile(risk_cdf, 0.975, -1, 1),
    }


def cp_pooled_mixture(histories, nodes=80):
    """Return classical pooled-history CP; frozen-C equality is asserted only at K=1."""
    histories = np.asarray(histories, float)
    n, h, info = histories.T
    x, w = roots_legendre(nodes)
    kstar = 0.9975 * x + 1.0025
    mass = 0.99 * 0.5 * w
    kstar = np.r_[kstar, 200.0]
    mass = np.r_[mass, 0.01]
    reference_variance = n.sum() / np.sum(n * info)
    kappa = kstar / reference_variance
    total_information = np.sum(n * info)
    historical_mean = np.sum(n * info * h) / total_information
    historical_variance = 1.0 / total_information
    variances = historical_variance + 1.0 / kappa
    means = np.full(kstar.size, historical_mean)
    logmass = np.log(mass)
    return means, variances, logmass


def cp_log_kernel(theta, histories, nodes=80):
    theta = np.asarray(theta, float)
    means, variances, logmass = cp_pooled_mixture(histories, nodes)
    return logsumexp(
        logmass[:, None]
        + _normal_logpdf(theta[None, :], means[:, None], variances[:, None]),
        axis=0,
    )


def _legendre_posterior(log_kernel, log_likelihood, lo, hi, order):
    x, w = roots_legendre(order)
    theta = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    logmass = np.log(w * 0.5 * (hi - lo)) + log_kernel(theta) + log_likelihood(theta)
    weight = np.exp(logmass - logsumexp(logmass))
    return theta, weight


def cp_continuous(
    histories, control, treatment, common_variance, order=1536, kappa_nodes=80
):
    nc, yc = control
    nt, yt = treatment
    vc, vt = common_variance / nc, common_variance / nt
    hs = np.asarray(histories, float)[:, 1]
    scale = max(
        math.sqrt(vc), math.sqrt(common_variance / min(np.asarray(histories)[:, 0]))
    )
    lo, hi = (
        min(float(hs.min()), yc) - 15 * scale,
        max(float(hs.max()), yc) + 15 * scale,
    )
    theta, weight = _legendre_posterior(
        lambda z: cp_log_kernel(z, histories, kappa_nodes),
        lambda z: -0.5 * (z - yc) ** 2 / vc,
        lo,
        hi,
        order,
    )
    result = _delta_from_control_nodes(theta, weight, yt, vt)
    result["diagnostics"] = {
        "theta_order": order,
        "theta_bounds": [lo, hi],
        "classical_pooled": True,
    }
    return result


def cp_binary(
    histories,
    control,
    treatment,
    order=1536,
    bound=18.0,
    check_bound=24.0,
    kappa_nodes=80,
):
    nc, yc = control

    def calculate(limit):
        theta, weight = _legendre_posterior(
            lambda z: cp_log_kernel(z, histories, kappa_nodes),
            lambda z: yc * z - nc * np.logaddexp(0.0, z),
            -limit,
            limit,
            order,
        )
        return _binary_from_control_nodes(theta, weight, treatment)

    primary, check = calculate(bound), calculate(check_bound)
    fields = (
        "estimate",
        "interval_lower",
        "interval_upper",
        "posterior_probability_positive",
        "risk_difference_mean",
        "risk_difference_q025",
        "risk_difference_q975",
    )
    primary["diagnostics"] = {
        "theta_order": order,
        "bounds": [bound, check_bound],
        "bound_check_max_abs": max(abs(primary[k] - check[k]) for k in fields),
        "classical_pooled": True,
    }
    return primary


def dirichlet_stick_breaking(uniforms, alpha):
    uniforms, alpha = np.asarray(uniforms, float), np.asarray(alpha, float)
    if alpha.size == 1:
        return np.ones((uniforms.shape[0], 1))
    remaining = np.ones(uniforms.shape[0])
    result = np.empty((uniforms.shape[0], alpha.size))
    for j in range(alpha.size - 1):
        v = betaincinv(alpha[j], alpha[j + 1 :].sum(), uniforms[:, j])
        result[:, j] = remaining * v
        remaining *= 1.0 - v
    result[:, -1] = remaining
    return result


def uip_components(histories, planned_n_control, power, scramble_seed):
    histories = np.asarray(histories, float)
    n, h, info = histories.T
    sampler = qmc.Sobol(
        d=max(1, len(histories)), scramble=True, seed=int(scramble_seed)
    )
    u = sampler.random_base2(power)
    if len(histories) == 1:
        weight = np.ones((u.shape[0], 1))
        m_u = u[:, 0]
    else:
        weight = dirichlet_stick_breaking(
            u[:, :-1], np.minimum(1.0, n / planned_n_control)
        )
        m_u = u[:, -1]
    M = min(float(planned_n_control), float(n.sum())) * m_u
    mean = weight @ h
    precision = M * (weight @ info)
    variance = 1.0 / np.maximum(precision, np.finfo(float).tiny)
    return mean, variance


def _continuous_uip_once(
    histories, control, treatment, common_variance, planned_n_control, power, seed
):
    mu, var = uip_components(histories, planned_n_control, power, seed)
    nc, yc = control
    nt, yt = treatment
    vc, vt = common_variance / nc, common_variance / nt
    loge = _normal_logpdf(yc, mu, var + vc)
    weight = np.exp(loge - logsumexp(loge))
    pv = 1.0 / (1.0 / var + 1.0 / vc)
    pm = pv * (mu / var + yc / vc)
    dm, dv = yt - pm, vt + pv
    estimate = float(weight @ dm)
    variance_delta = float(weight @ (dv + dm * dm) - estimate * estimate)
    prob = float(weight @ ndtr(dm / np.sqrt(dv)))

    def cdf(x):
        return float(weight @ ndtr((x - dm) / np.sqrt(dv)))

    return {
        "estimate": estimate,
        "posterior_sd": math.sqrt(max(variance_delta, 0)),
        "interval_lower": _quantile(cdf, 0.025),
        "interval_upper": _quantile(cdf, 0.975),
        "posterior_probability_positive": prob,
    }


def _uip_weight_draws(histories, planned_n_control, power, seed):
    histories = np.asarray(histories, float)
    n, h, info = histories.T
    k = len(histories)
    if k == 1:
        source_weight = np.ones((1, 1))
    else:
        sampler = qmc.Sobol(d=k - 1, scramble=True, seed=int(seed))
        source_weight = dirichlet_stick_breaking(
            sampler.random_base2(power), np.minimum(1.0, n / planned_n_control)
        )
    return (
        source_weight @ h,
        source_weight @ info,
        min(float(planned_n_control), float(n.sum())),
    )


def _uniform_m_marginal_density(theta, means, informations, mmax, chunk=2048):
    """Evaluate E_w int phi(theta;mu_w,1/(M I_w)) dM/Mmax stably."""
    theta = np.asarray(theta, float)
    total = np.zeros(theta.size)
    g32 = float(gamma(1.5))
    for start in range(0, means.size, chunk):
        stop = min(start + chunk, means.size)
        info = informations[start:stop, None]
        c = 0.5 * info * (theta[None, :] - means[start:stop, None]) ** 2
        x = c * mmax
        integral = np.empty_like(c)
        small = x < 1e-5
        # Integral_0^M m^(1/2) exp(-c m) dm, with a stable expansion at c=0.
        integral[small] = (
            (2.0 / 3.0)
            * mmax**1.5
            * (1.0 - 0.6 * x[small] + (3.0 / 14.0) * x[small] ** 2)
        )
        integral[~small] = g32 * gammainc(1.5, x[~small]) / c[~small] ** 1.5
        density = np.sqrt(info / (2.0 * math.pi)) * integral / mmax
        total += density.sum(axis=0)
    return total / means.size


def _binary_uip_once(
    histories, control, treatment, planned_n_control, power, seed, bound, theta_order
):
    means, informations, mmax = _uip_weight_draws(
        histories, planned_n_control, power, seed
    )
    x, qw = roots_legendre(theta_order)
    eta = bound * x
    prior = _uniform_m_marginal_density(eta, means, informations, mmax)
    nc, yc = control
    logw = np.log(qw * bound) + yc * eta - nc * np.logaddexp(0.0, eta) + np.log(prior)
    weight = np.exp(logw - logsumexp(logw))
    return _binary_from_control_nodes(eta, weight, treatment)


def _rqmc(run_once, scrambles, initial_power, convergence_power, seed, fields):
    levels = {}
    raw = {}
    for power in (initial_power, convergence_power):
        values = [run_once(power, seed + 100003 * r) for r in range(scrambles)]
        raw[power] = values
        levels[str(power)] = {
            field: {
                "estimate": float(np.mean([v[field] for v in values])),
                "qmc_se": float(
                    np.std([v[field] for v in values], ddof=1) / math.sqrt(scrambles)
                )
                if scrambles > 1
                else None,
            }
            for field in fields
        }
    result = {
        field: levels[str(convergence_power)][field]["estimate"] for field in fields
    }
    result["diagnostics"] = {
        "scrambles": scrambles,
        "initial_power": initial_power,
        "convergence_power": convergence_power,
        "levels": levels,
        "max_level_change": max(
            abs(
                levels[str(initial_power)][f]["estimate"]
                - levels[str(convergence_power)][f]["estimate"]
            )
            for f in fields
        ),
    }
    return result


def uip_continuous(
    histories,
    control,
    treatment,
    common_variance,
    planned_n_control,
    scrambles=16,
    initial_power=15,
    convergence_power=16,
    seed=202609160501,
):
    fields = (
        "estimate",
        "posterior_sd",
        "interval_lower",
        "interval_upper",
        "posterior_probability_positive",
    )
    return _rqmc(
        lambda p, s: _continuous_uip_once(
            histories, control, treatment, common_variance, planned_n_control, p, s
        ),
        scrambles,
        initial_power,
        convergence_power,
        seed,
        fields,
    )


def uip_binary(
    histories,
    control,
    treatment,
    planned_n_control,
    scrambles=16,
    initial_power=15,
    convergence_power=16,
    seed=202609160501,
    theta_order=768,
    bound=18.0,
    check_bound=24.0,
):
    fields = (
        "estimate",
        "posterior_sd",
        "interval_lower",
        "interval_upper",
        "posterior_probability_positive",
        "risk_difference_mean",
        "risk_difference_q025",
        "risk_difference_q975",
    )

    def run(p, s, b=bound):
        return _binary_uip_once(
            histories, control, treatment, planned_n_control, p, s, b, theta_order
        )

    result = _rqmc(run, scrambles, initial_power, convergence_power, seed, fields)
    check_values = [
        run(convergence_power, seed + 100003 * r, check_bound) for r in range(scrambles)
    ]
    result["diagnostics"]["bounds"] = [bound, check_bound]
    result["diagnostics"]["bound_check_max_abs"] = max(
        abs(result[f] - np.mean([v[f] for v in check_values])) for f in fields
    )
    result["diagnostics"]["theta_order"] = theta_order
    result["diagnostics"]["binary_integration"] = (
        "shared_logit_Gauss-Legendre_grid_with_analytic_Uniform-M_marginal_kernel"
    )
    return result
