from __future__ import annotations

import inspect
from math import exp, sqrt

import pytest

from ruip.priors import (
    DesignStageRUIPPrior,
    HistoricalSummary,
    design_stage_ruip_prior,
    robust_uip_prior,
)


def _histories() -> tuple[HistoricalSummary, ...]:
    return (
        HistoricalSummary("a", -0.5, 20, 0.7),
        HistoricalSummary("b", 1.2, 8, 1.9),
        HistoricalSummary("c", 0.1, 21, 0.4),
    )


def test_public_constructor_is_authoritative_scalar_design_stage_rule() -> None:
    prior = robust_uip_prior(_histories(), delta_clin=0.4)
    assert isinstance(prior, DesignStageRUIPPrior)
    assert prior == design_stage_ruip_prior(_histories(), delta_clin=0.4)
    assert not hasattr(prior, "states")
    assert not hasattr(prior, "lambda_nodes")


def test_capped_medoid_components_match_hand_calculation() -> None:
    histories = _histories()
    prior = robust_uip_prior(histories, delta_clin=0.5)
    information = [item.n_k * item.I_Uk for item in histories]
    assert prior.anchor_index == 2
    expected_scores = tuple(
        1.0
        if k == prior.anchor_index
        else exp(
            -0.5
            * (histories[k].h_k - histories[prior.anchor_index].h_k) ** 2
            / (1 / information[k] + 1 / information[prior.anchor_index])
        )
        for k in range(3)
    )
    retained = tuple(
        min(information[k], information[prior.anchor_index]) * expected_scores[k]
        for k in range(3)
    )
    weights = tuple(value / sum(retained) for value in retained)
    assert prior.compatibility == pytest.approx(expected_scores)
    assert prior.effective_information == pytest.approx(retained)
    assert prior.weights == pytest.approx(weights)
    assert prior.mean == pytest.approx(
        sum(w * h.h_k for w, h in zip(weights, histories, strict=True))
    )


def test_prior_factorization_and_margin_monotonicity() -> None:
    small = robust_uip_prior(_histories(), delta_clin=0.2)
    large = robust_uip_prior(_histories(), delta_clin=0.8)
    assert small.mean == pytest.approx(large.mean)
    assert small.weights == pytest.approx(large.weights)
    assert small.precision == pytest.approx(
        1 / (small.local_variance + small.delta_clin**2)
    )
    assert small.precision == pytest.approx(
        small.global_discount * small.local_precision
    )
    assert small.precision > large.precision
    assert small.global_discount > large.global_discount


def test_local_conflict_reduces_effective_information() -> None:
    compatible = (
        HistoricalSummary("a", 0.0, 100, 1.0),
        HistoricalSummary("b", 0.0, 100, 1.0),
        HistoricalSummary("c", 0.0, 100, 1.0),
    )
    heterogeneous = compatible[:2] + (HistoricalSummary("c", 1.0, 100, 1.0),)
    stable = robust_uip_prior(compatible, delta_clin=0.4)
    unstable = robust_uip_prior(heterogeneous, delta_clin=0.4)
    assert unstable.local_variance > stable.local_variance
    assert unstable.variance == pytest.approx(
        unstable.local_variance + unstable.delta_clin**2
    )
    assert unstable.precision < stable.precision
    assert unstable.effective_information[2] < stable.effective_information[2]


def test_margin_is_not_augmented_by_leave_one_out_distance() -> None:
    prior = robust_uip_prior(_histories(), delta_clin=0.4)
    assert prior.precision == pytest.approx(
        1.0 / (prior.local_variance + prior.delta_clin**2)
    )


def test_single_history_has_no_internal_heterogeneity_estimate() -> None:
    history = (_histories()[0],)
    prior = robust_uip_prior(history, delta_clin=0.3)
    expected_sampling_variance = 1 / (history[0].n_k * history[0].I_Uk)
    assert prior.working_variances == pytest.approx((expected_sampling_variance,))
    assert prior.anchor_index == 0
    assert prior.compatibility == (1.0,)
    assert prior.weights == (1.0,)
    assert prior.mean == history[0].h_k
    assert prior.variance == pytest.approx(expected_sampling_variance + 0.3**2)


def test_normal_density_is_normalized_by_known_formula() -> None:
    prior = robust_uip_prior(_histories(), delta_clin=0.4)
    expected_at_mean = 1 / sqrt(2 * 3.141592653589793 * prior.variance)
    assert prior.density(prior.mean) == pytest.approx(expected_at_mean)
    assert exp(prior.log_density(prior.mean)) == pytest.approx(expected_at_mean)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("inf"), float("nan")])
def test_margin_must_be_prespecified_positive_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        robust_uip_prior(_histories(), delta_clin=bad)


def test_signature_has_margin_but_no_current_outcome_or_design_terms() -> None:
    names = set(inspect.signature(robust_uip_prior).parameters)
    assert names == {"histories", "delta_clin"}
    forbidden = ("current", "n_control", "n_current", "target", "power")
    assert not any(pattern in name.lower() for name in names for pattern in forbidden)


def test_source_is_single_normal_and_has_no_mixture_gate_language() -> None:
    source = inspect.getsource(inspect.getmodule(robust_uip_prior)).lower()
    for forbidden in ("half_cauchy", "retention_mask", "compatibility_gate", "2**k"):
        assert forbidden not in source

