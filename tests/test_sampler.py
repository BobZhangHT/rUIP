import numpy as np

from uip_ic.data_generation import generate_interval_censored_data
from uip_ic.interval_ph import PiecewiseBaseline
from uip_ic.priors import HistoricalSummary, build_commensurate_prior
from uip_ic.samplers import SamplerConfig, run_da_sampler


def test_short_nip_chain_runs_and_is_reproducible() -> None:
    baseline = PiecewiseBaseline(np.array([0.0, 1.0, 2.0]), np.array([0.6, 0.4, 0.25]))
    data = generate_interval_censored_data(
        np.random.default_rng(44), 35, baseline, -0.35, 0.2, [0.5, 1.0, 1.5, 2.0, 2.5]
    )
    config = SamplerConfig(iterations=60, burn_in=30, slice_steps=30)
    first = run_da_sampler(data, baseline.interval_starts, "IC-NIP", config, np.random.default_rng(8))
    second = run_da_sampler(data, baseline.interval_starts, "IC-NIP", config, np.random.default_rng(8))
    assert len(first) == 30
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy(), rtol=0, atol=0)


def test_short_commensurate_chain_has_positive_precision() -> None:
    baseline = PiecewiseBaseline(np.array([0.0, 1.0]), np.array([0.5, 0.3]))
    data = generate_interval_censored_data(
        np.random.default_rng(51), 35, baseline, -0.35, 0.2, [0.5, 1.0, 1.5, 2.0]
    )
    cp = build_commensurate_prior(
        [HistoricalSummary("H1", -0.3, 0.25, 100), HistoricalSummary("H2", -0.4, 0.25, 100)]
    )
    draws = run_da_sampler(
        data,
        baseline.interval_starts,
        "IC-CP",
        SamplerConfig(iterations=60, burn_in=30, slice_steps=30),
        np.random.default_rng(9),
        commensurate=cp,
    )
    assert np.all(draws["commensurate_precision"] > 0)
