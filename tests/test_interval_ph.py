import numpy as np

from uip_ic.data_generation import generate_exact_survival_data
from uip_ic.interval_ph import PiecewiseBaseline, fit_cox_summary


def test_cumulative_hazard_inverse_round_trip() -> None:
    baseline = PiecewiseBaseline(
        np.array([0.0, 0.5, 1.5, 2.5]), np.array([0.4, 0.8, 0.2, 0.5])
    )
    time = np.array([0.0, 0.1, 0.5, 0.9, 1.5, 2.4, 2.5, 4.0])
    recovered = baseline.inverse_cumulative_hazard(baseline.cumulative_hazard(time))
    np.testing.assert_allclose(recovered, time, rtol=1e-12, atol=1e-12)


def test_cox_summary_is_finite_on_generated_history() -> None:
    baseline = PiecewiseBaseline(np.array([0.0, 1.0, 2.0]), np.array([0.7, 0.5, 0.3]))
    data = generate_exact_survival_data(
        np.random.default_rng(123), 250, baseline, -0.4, 0.2, 3.0, True
    )
    fit = fit_cox_summary(data.observed_time, data.event, data.treatment, data.covariate)
    assert np.isfinite(fit.theta_hat)
    assert np.isfinite(fit.se) and fit.se > 0
