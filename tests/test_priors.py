import numpy as np

from uip_ic.priors import HistoricalSummary, build_uip, sample_truncated_gamma_m, simplex_to_alr
from uip_ic.samplers import update_uip_weights


def test_m_draws_obey_bounds_and_shrink_under_conflict() -> None:
    uip = build_uip(
        [HistoricalSummary("H1", 0.0, 0.2, 100), HistoricalSummary("H2", 0.0, 0.2, 100)],
        m_max=80.0,
    )
    compatible_rng = np.random.default_rng(18)
    conflict_rng = np.random.default_rng(19)
    compatible = np.array([sample_truncated_gamma_m(compatible_rng, 0.0, uip) for _ in range(4000)])
    conflict = np.array([sample_truncated_gamma_m(conflict_rng, 2.0, uip) for _ in range(4000)])
    assert np.all((compatible > 0) & (compatible < uip.m_max))
    assert np.all((conflict > 0) & (conflict < uip.m_max))
    assert conflict.mean() < compatible.mean()


def test_dynamic_weights_favor_compatible_history() -> None:
    uip = build_uip(
        [HistoricalSummary("H1", 0.0, 0.2, 100), HistoricalSummary("H2", 1.0, 0.2, 100)],
        m_max=80.0,
        dirichlet_concentration=[1.0, 1.0],
    )
    rng = np.random.default_rng(81)
    log_ratios = simplex_to_alr(uip.weights)
    draws = []
    current = uip
    for iteration in range(1200):
        current, log_ratios = update_uip_weights(
            current, theta=0.0, m=50.0, log_ratios=log_ratios, rng=rng
        )
        if iteration >= 200:
            draws.append(current.weights.copy())
    weights = np.asarray(draws)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(weights > 0)
    assert weights[:, 0].mean() > 0.65
