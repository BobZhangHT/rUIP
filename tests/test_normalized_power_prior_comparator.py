"""Numerical and structural checks for the random-power PP comparator."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tri_power_comparator", ROOT / "scripts" / "run_outcome_simulations.py"
)
assert SPEC is not None and SPEC.loader is not None
tri = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tri
SPEC.loader.exec_module(tri)


def data(outcome: str, scenario: int = 4):
    return tri.make_dataset(
        outcome, tri.SCENARIOS[scenario], np.random.default_rng(4381)
    )


def test_power_prior_is_normalized_and_invariant_to_current_data():
    original = data("binary")
    changed = replace(original, current_events=original.current_events + 17)
    prior = tri.normalized_power_prior(original, order=12)
    assert np.exp(prior.log_weights).sum() == pytest.approx(1.0)
    assert tri.prior_fingerprint(prior) == tri.prior_fingerprint(
        tri.normalized_power_prior(changed, order=12)
    )
    # Each conditional prior integrates to one; quadrature weights implement
    # the independent uniform law on the three powers.
    for index in (0, 49, len(prior.means) - 1):
        mass = quad(
            lambda x: norm.pdf(x, prior.means[index], np.sqrt(prior.variances[index])),
            -np.inf,
            np.inf,
        )[0]
        assert mass == pytest.approx(1.0, abs=1e-10)


def test_power_prior_analytic_single_history_reference():
    original = data("continuous")
    # Make the second and third historical likelihoods uninformative. The
    # three-power cubature must reduce to one-dimensional beta integration.
    modified = replace(
        original,
        history_information=np.array([120.0, 0.0, 0.0]),
    )
    prior = tri.normalized_power_prior(modified, order=32)
    theta = 0.12
    observed = float(np.exp(tri.mixture_log_density(np.array([theta]), prior))[0])
    reference = quad(
        lambda a: norm.pdf(
            theta,
            (a * 120.0 * modified.history_estimate[0]) / (0.1 + a * 120.0),
            np.sqrt(1.0 / (0.1 + a * 120.0)),
        ),
        0.0,
        1.0,
        epsabs=1e-11,
    )[0]
    assert observed == pytest.approx(reference, rel=2e-4)


@pytest.mark.parametrize("outcome", tri.ALL_OUTCOMES)
def test_power_posterior_cubature_converges_on_conflict(outcome):
    dataset = data(outcome, scenario=5)
    low = tri.normalized_power_prior(dataset, order=12)
    high = tri.normalized_power_prior(dataset, order=20)
    if outcome == "continuous":

        def summarize(prior):
            post_variance = 1.0 / (1.0 / prior.variances + dataset.current_information)
            post_mean = post_variance * (
                prior.means / prior.variances
                + dataset.current_information * dataset.current_estimate
            )
            log_weight = prior.log_weights - 0.5 * (
                np.log(
                    2.0 * np.pi * (prior.variances + 1.0 / dataset.current_information)
                )
                + (dataset.current_estimate - prior.means) ** 2
                / (prior.variances + 1.0 / dataset.current_information)
            )
            weight = np.exp(log_weight - tri.logsumexp(log_weight))
            mean = float(np.dot(weight, post_mean))
            variance = float(np.dot(weight, post_variance + (post_mean - mean) ** 2))
            return mean, variance

        a, b = summarize(low), summarize(high)
    else:
        a, b = (
            tri.numerical_posterior(dataset, low),
            tri.numerical_posterior(dataset, high),
        )
    assert abs(a[0] - b[0]) < 3e-4
    assert abs(a[1] - b[1]) < 1e-4


def test_precision_leverage_conflict_cubature_converges():
    original = data("continuous")
    dataset = replace(
        original,
        history_information=np.array([300.0, 50.0, 80.0]),
        history_estimate=np.array([0.75, 0.0, 0.0]),
        current_estimate=0.0,
    )

    def mean_at(order):
        prior = tri.normalized_power_prior(dataset, order)
        variance = 1.0 / (1.0 / prior.variances + dataset.current_information)
        mean = variance * (prior.means / prior.variances)
        log_weight = prior.log_weights - 0.5 * (
            np.log(2.0 * np.pi * (prior.variances + 0.01))
            + prior.means**2 / (prior.variances + 0.01)
        )
        weight = np.exp(log_weight - tri.logsumexp(log_weight))
        return float(np.dot(weight, mean))

    assert abs(mean_at(12) - mean_at(20)) < 3e-4
