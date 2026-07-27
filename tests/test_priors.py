import numpy as np

from uip_ic.priors import HistoricalSummary, build_uip, sample_truncated_gamma_m


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
