from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from ruip.realdata import METHODS, analyze_memantine, analyze_secukinumab, validate_rows

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "realdata79", ROOT / "scripts" / "run_clinical_applications.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_public_arm_level_inputs_map_to_expected_analysis_summaries() -> None:
    mem_h, mem_c, mem_t, pooled = runner.load_memantine()
    sec_h, sec_y, sec_c, sec_t = runner.load_secukinumab()
    assert len(mem_h) == 5
    assert (mem_c.n, mem_c.mean, mem_t.n, mem_t.mean) == (125, 0.86, 136, 0.97)
    assert pooled == pytest.approx(124.8624, abs=5e-5)
    assert len(sec_h) == len(sec_y) == 8
    assert (sec_c.events, sec_c.n, sec_t.events, sec_t.n) == (1, 6, 14, 23)


def test_all_six_methods_run_for_both_applications() -> None:
    mem_h, mem_c, mem_t, _ = runner.load_memantine()
    sec_h, sec_y, sec_c, sec_t = runner.load_secukinumab()
    mem = analyze_memantine(
        mem_h,
        mem_c,
        mem_t,
        delta_clin=8.0,
        margin_label="test",
        uip_scrambles=2,
        uip_initial_power=5,
        uip_convergence_power=6,
        pp_power=10,
    )
    sec = analyze_secukinumab(
        sec_h,
        sec_y,
        sec_c,
        sec_t,
        delta_clin=math.log(1.5),
        margin_label="test",
        uip_scrambles=2,
        uip_initial_power=5,
        uip_convergence_power=6,
        theta_order=256,
        pp_power=10,
    )
    validate_rows(mem + sec)
    assert tuple(row.method for row in mem) == METHODS
    assert tuple(row.method for row in sec) == METHODS
    assert all(row.status == "success" for row in mem + sec)


def test_margin_specification_is_external_or_transparently_assumed() -> None:
    spec = runner._margin_specification()
    mem = spec["memantine_npi"]
    sec = spec["secukinumab_asas20"]
    assert mem["primary_delta_clin"] == 8.0
    assert mem["source"]["doi"] == "10.1002/gps.2607"
    assert "not the shorter NPI-Q" in mem["source"]["scale_warning"]
    assert sec["source"] is None
    assert sec["sensitivity_odds_ratio_grid"] == [1.25, 1.5, 2.0]
    assert "must not be described as a validated clinical margin" in sec["rationale"]


def test_binary_power_prior_uses_gaussian_history_and_exact_current_binomial():
    sec_h, sec_y, sec_c, sec_t = runner.load_secukinumab()
    estimates = []
    for power in (10, 12):
        rows = analyze_secukinumab(
            sec_h,
            sec_y,
            sec_c,
            sec_t,
            delta_clin=math.log(1.5),
            margin_label="test",
            uip_scrambles=2,
            uip_initial_power=5,
            uip_convergence_power=6,
            theta_order=256,
            pp_power=power,
        )
        pp = next(row for row in rows if row.method == "PP")
        diagnostics = json.loads(pp.diagnostics_json)
        assert (
            diagnostics["historical_likelihood"] == "Gaussian_corrected_logit_summary"
        )
        assert diagnostics["current_likelihood"] == "exact_binomial"
        assert diagnostics["power_distribution"] == "independent_Beta(1,1)"
        estimates.append(pp.estimate)
    assert abs(estimates[0] - estimates[1]) < 1e-3
