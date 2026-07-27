import numpy as np

from uip_ic.augmentation import (
    baseline_full_conditional,
    initialize_latent_times,
    sample_latent_times,
)
from uip_ic.data_generation import generate_interval_censored_data
from uip_ic.interval_ph import PiecewiseBaseline


def _toy_data():
    baseline = PiecewiseBaseline(
        np.array([0.0, 0.75, 1.5, 2.25]), np.array([0.6, 0.4, 0.3, 0.2])
    )
    data = generate_interval_censored_data(
        np.random.default_rng(25), 60, baseline, -0.3, 0.2, [0.5, 1.0, 1.5, 2.0, 2.5]
    )
    return baseline, data


def test_latent_draws_remain_in_observed_intervals() -> None:
    baseline, data = _toy_data()
    rng = np.random.default_rng(7)
    for _ in range(50):
        latent = sample_latent_times(data, baseline, -0.3, 0.2, rng)
        finite = data.finite_event
        assert np.all(latent[finite] > data.left[finite])
        assert np.all(latent[finite] <= data.right[finite])


def test_gamma_baseline_parameters_are_positive() -> None:
    baseline, data = _toy_data()
    latent = initialize_latent_times(data)
    shape, rate = baseline_full_conditional(
        data, latent, baseline.interval_starts, -0.3, 0.2, 0.5, 0.5
    )
    assert np.all(shape > 0)
    assert np.all(rate > 0)
    assert np.isfinite(rate).all()
