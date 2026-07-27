import numpy as np

from uip_ic.data_generation import generate_interval_censored_data
from uip_ic.interval_ph import PiecewiseBaseline
from uip_ic.samplers import SamplerConfig, run_da_sampler


def test_short_nip_chain_runs_and_is_reproducible() -> None:
    baseline = PiecewiseBaseline(np.array([0.0, 1.0, 2.0]), np.array([0.6, 0.4, 0.25]))
    data = generate_interval_censored_data(
        np.random.default_rng(44), 35, baseline, -0.35, 0.2, [0.5, 1.0, 1.5, 2.0, 2.5]
    )
    config = SamplerConfig(iterations=60, burn_in=30, slice_steps=30)
    first = run_da_sampler(data, baseline.interval_starts, "NIP-DA", config, np.random.default_rng(8))
    second = run_da_sampler(data, baseline.interval_starts, "NIP-DA", config, np.random.default_rng(8))
    assert len(first) == 30
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy(), rtol=0, atol=0)
