import numpy as np

from uip_ic.data_generation import construct_intervals, generate_interval_censored_data
from uip_ic.interval_ph import PiecewiseBaseline


def test_interval_construction_contains_true_failure() -> None:
    event = np.array([0.1, 0.5, 0.7, 1.5, 2.7])
    left, right = construct_intervals(event, np.array([0.5, 1.0, 1.5, 2.0]))
    assert np.all(left < event)
    assert np.all(event <= right)
    assert np.isinf(right[-1])


def test_fixed_seed_reproduces_generated_data() -> None:
    baseline = PiecewiseBaseline(np.array([0.0, 1.0]), np.array([0.5, 0.3]))
    first = generate_interval_censored_data(
        np.random.default_rng(99), 30, baseline, -0.3, 0.2, [0.5, 1.0, 1.5]
    )
    second = generate_interval_censored_data(
        np.random.default_rng(99), 30, baseline, -0.3, 0.2, [0.5, 1.0, 1.5]
    )
    for name in ["left", "right", "treatment", "covariate", "true_event_time"]:
        assert np.array_equal(getattr(first, name), getattr(second, name))
