from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tri_outcome", ROOT / "scripts" / "run_outcome_simulations.py"
)
assert SPEC is not None and SPEC.loader is not None
tri = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tri
SPEC.loader.exec_module(tri)


def dataset(outcome: str, scenario_index: int = 0):
    return tri.make_dataset(
        outcome, tri.SCENARIOS[scenario_index], np.random.default_rng(20260918)
    )


def test_design_has_six_prespecified_scenarios_and_frozen_subject_counts():
    assert [item.scenario_id for item in tri.SCENARIOS] == [
        "GW-LW-0",
        "GW-LW-D",
        "GS-LW-P",
        "GS-LW-N",
        "GW-LS-I",
        "GS-LS-I",
    ]
    binary_counts = {tuple(dataset("binary", i).history_subjects) for i in range(6)}
    survival_counts = {tuple(dataset("survival", i).history_subjects) for i in range(6)}
    assert binary_counts == {(500, 333, 208)}
    assert survival_counts == {(686, 458, 286)}
    assert dataset("binary").current_subjects == 417
    assert dataset("survival").current_subjects == 572


def test_design_config_matches_authoritative_runner_constants():
    config = json.loads((ROOT / "scripts" / "outcome_design.json").read_text())
    assert config["seed"] == tri.SEED
    assert config["survival"]["baseline_hazard_dgp_only"] == tri.BASELINE_HAZARD
    assert config["survival"]["analysis"] == "Cox partial likelihood"
    assert "scripts/run_outcome_simulations.py" in config["runner_authority"]


def test_binary_power_prior_uses_normalized_history_and_exact_current_likelihood():
    data = dataset("binary")
    prior = tri.normalized_power_prior(data)
    mean, variance, lower, upper, error = tri.exact_nip_or_pp(data, "PP")
    assert (mean, variance, lower, upper, error) == pytest.approx(
        tri.numerical_posterior(data, prior)
    )
    assert lower < mean < upper
    assert error < 1e-4


def test_survival_power_prior_uses_current_partial_likelihood():
    data = dataset("survival")
    observed = tri.exact_nip_or_pp(data, "PP")
    expected = tri.numerical_posterior(data, tri.normalized_power_prior(data))
    assert observed == pytest.approx(expected)


@pytest.mark.parametrize("method", ["rUIP", "rUIP no-local", "rUIP no-global"])
def test_ruip_prior_is_invariant_to_current_data(method):
    original = dataset("survival", 4)
    changed = replace(
        original,
        current_estimate=2.0,
        current_information=17.0,
        current_events=8,
        current_exposure=93.0,
        current_censored=120,
    )
    assert tri.prior_fingerprint(
        tri.ruip_prior(original, method)
    ) == tri.prior_fingerprint(tri.ruip_prior(changed, method))


def test_ruip_uses_capped_medoid_information_plus_margin_variance():
    data = dataset("continuous", 4)
    prior = tri.ruip_prior(data, "rUIP")
    h = data.history_estimate
    info = data.history_information
    objectives = np.abs(h[:, None] - h[None, :]).sum(axis=1)
    anchor = min(range(3), key=lambda k: (objectives[k], -info[k], k))
    compatibility = np.exp(
        -0.5 * (h - h[anchor]) ** 2 / (1.0 / info + 1.0 / info[anchor])
    )
    compatibility[anchor] = 1.0
    retained = np.minimum(info, info[anchor]) * compatibility
    retained[anchor] = info[anchor]
    local_variance = 1.0 / retained.sum()
    assert prior.variances[0] == pytest.approx(local_variance + tri.DELTA_CLIN**2)


def test_closed_form_continuous_nip_and_normalized_pp():
    data = dataset("continuous")
    nip = tri.exact_nip_or_pp(data, "NIP")
    assert nip[0] == data.current_estimate
    assert nip[1] == pytest.approx(0.01)
    pp = tri.exact_nip_or_pp(data, "PP")
    assert 1.0 / 350.1 < pp[1] < 1.0 / 100.0
    assert pp == pytest.approx(tri.analyze(data, "PP", None, None)[:5])


def test_quadrant_rmse_pools_squared_errors_before_square_root():
    raw = pd.DataFrame(
        {
            "status": ["success", "success"],
            "outcome": ["continuous"] * 2,
            "global_conflict": ["weak"] * 2,
            "local_conflict": ["weak"] * 2,
            "method": ["rUIP"] * 2,
            "squared_error": [0.0, 4.0],
            "error": [0.0, 2.0],
            "covered": [1, 0],
            "interval_width": [1.0, 1.0],
            "replicate_id": [0, 1],
        }
    )
    result = tri.quadrant_summary(raw).iloc[0]
    assert result.rmse == pytest.approx(np.sqrt(2.0))


def test_frozen_prior_invariance_uses_planned_subject_count():
    if not tri.FROZEN_LIBRARY.exists():
        pytest.skip("frozen comparator library unavailable")
    abi, dll = tri.load_frozen_engine()
    original = dataset("binary", 2)
    changed = replace(
        original, current_estimate=-3.0, current_information=12.0, current_events=5
    )
    for method in ("rMAP", "CP", "UIP"):
        a = tri.prior_fingerprint(tri.frozen_prior(original, method, abi, dll))
        b = tri.prior_fingerprint(tri.frozen_prior(changed, method, abi, dll))
        assert a == b


def test_git_head_is_safe_without_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(tri, "ROOT", tmp_path)
    assert tri.git_head() == "archived/unavailable"


def test_numerical_error_is_normalization_mass_discrepancy():
    data = dataset("survival")
    prior = tri.ruip_prior(data, "rUIP")
    *_, discrepancy = tri.numerical_posterior(data, prior)
    assert 0.0 <= discrepancy < 1e-4


def test_survival_analysis_does_not_use_exposure_or_baseline_hazard():
    original = dataset("survival")
    changed = replace(
        original,
        current_exposure=original.current_exposure * 17.0,
        history_exposure=original.history_exposure * 23.0,
    )
    assert tri.exact_nip_or_pp(original, "NIP") == pytest.approx(
        tri.exact_nip_or_pp(changed, "NIP")
    )
    prior = tri.ruip_prior(original, "rUIP")
    assert tri.numerical_posterior(original, prior) == pytest.approx(
        tri.numerical_posterior(changed, prior)
    )
