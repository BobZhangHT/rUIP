from __future__ import annotations

import numpy as np
import pytest

from ruip.outcomes.survival import (
    compress_risk_sets,
    log_partial_likelihood,
    partial_likelihood_summary,
    posterior_summary,
    score_information,
)


def test_compressed_risk_sets_match_hand_calculation():
    data = compress_risk_sets(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1, 1, 0, 1]),
        np.array([1, 0, 1, 0]),
    )
    assert data.risk_control.tolist() == [2.0, 2.0, 1.0]
    assert data.risk_treatment.tolist() == [2.0, 1.0, 0.0]
    assert data.treated_events.tolist() == [1.0, 0.0, 0.0]
    theta = 0.3
    expected = theta - np.log(2 + 2 * np.exp(theta)) - np.log(
        2 + np.exp(theta)
    )
    expected -= np.log(1.0)
    assert log_partial_likelihood(theta, data) == pytest.approx(expected)


def test_score_is_zero_at_partial_likelihood_mle_and_info_is_curvature():
    rng = np.random.default_rng(4)
    treatment = np.repeat([0, 1], 100)
    rng.shuffle(treatment)
    event_time = rng.exponential(1.0 / np.exp(0.4 * treatment))
    censor_time = rng.exponential(2.0, treatment.size)
    data = compress_risk_sets(
        np.minimum(event_time, censor_time), event_time <= censor_time, treatment
    )
    estimate, information = partial_likelihood_summary(data)
    score, observed_information = score_information(estimate, data)
    assert score == pytest.approx(0.0, abs=1e-9)
    assert information == pytest.approx(observed_information)
    step = 1e-4
    curvature = -(
        log_partial_likelihood(estimate + step, data)
        - 2 * log_partial_likelihood(estimate, data)
        + log_partial_likelihood(estimate - step, data)
    ) / step**2
    assert information == pytest.approx(curvature, rel=1e-5)


def test_no_event_dataset_is_rejected():
    with pytest.raises(ValueError, match="at least one event"):
        compress_risk_sets(np.ones(4), np.zeros(4), np.array([0, 0, 1, 1]))


def test_partial_likelihood_is_invariant_to_time_rescaling():
    time = np.array([0.7, 1.4, 2.1, 3.5, 4.2, 5.0])
    event = np.array([1, 0, 1, 1, 0, 1])
    treatment = np.array([0, 1, 1, 0, 1, 0])
    original = compress_risk_sets(time, event, treatment)
    rescaled = compress_risk_sets(17.0 * time, event, treatment)
    grid = np.array([-0.8, 0.0, 1.1])
    np.testing.assert_allclose(
        log_partial_likelihood(grid, original),
        log_partial_likelihood(grid, rescaled),
    )


def test_power_prior_grid_covers_historical_conflict():
    rng = np.random.default_rng(91)

    def trial(theta: float):
        treatment = np.repeat([0, 1], 300)
        rng.shuffle(treatment)
        event_time = rng.exponential(1.0 / np.exp(theta * treatment))
        censor_time = rng.exponential(2.0, treatment.size)
        return compress_risk_sets(
            np.minimum(event_time, censor_time), event_time <= censor_time, treatment
        )

    current = trial(0.0)
    history = trial(2.0)
    nip = posterior_summary(current)
    pp = posterior_summary(current, historical=(history,), historical_weight=0.5)
    assert pp[0] > nip[0] + 0.4
    assert pp[2] < pp[0] < pp[3]
    assert pp[4] < 1e-4
