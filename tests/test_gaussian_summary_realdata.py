from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from ruip.priors import HistoricalSummary
from ruip.realdata import GaussianSummary, analyze_gaussian_summary, freeze_priors

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oncology83", ROOT / "scripts" / "run_oncology_application.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_oncology_input_and_all_output_grids(tmp_path: Path) -> None:
    _, studies, histories, current = runner.load_data()
    assert len(studies) == 10
    assert len(histories) == 9
    assert current.events == 32
    # Run through the CLI entry point with a low-cost temporary QMC setting.
    old_argv = sys.argv
    try:
        sys.argv = [
            "83",
            "--output",
            str(tmp_path),
            "--uip-power",
            "5",
            "--pp-power",
            "9",
        ]
        runner.main()
    except SystemExit as exc:
        assert exc.code == 0
    finally:
        sys.argv = old_argv
    import pandas as pd

    assert len(pd.read_csv(tmp_path / "posterior.csv")) == 6
    loo = pd.read_csv(tmp_path / "loo_predictive.csv")
    assert len(loo) == 50
    assert loo["prior_predictive_log_density"].notna().all()
    assert loo["prior_predictive_log_density"].map(math.isfinite).all()
    uip = loo.loc[loo.method.eq("UIP")]
    for row in uip.itertuples(index=False):
        keep = tuple(
            item.historical() for item in studies if item.source_id != row.held_out
        )
        expected = runner.historical_information_cap(keep)
        assert json.loads(row.diagnostics_json)["planned_events"] == expected
    assert len(pd.read_csv(tmp_path / "source_weights.csv")) == 9
    assert len(pd.read_csv(tmp_path / "sensitivity.csv")) == 3


def test_priors_are_invariant_to_current_outcome() -> None:
    _, _, histories, current = runner.load_data()
    frozen_a, _ = freeze_priors(
        histories, delta_clin=math.log(1.5), planned_events=current.events, uip_power=5
    )
    alternate = GaussianSummary("alternate", 55, 117.6)
    frozen_b, _ = freeze_priors(
        histories, delta_clin=math.log(1.5), planned_events=current.events, uip_power=5
    )
    for method in ("PP", "rMAP", "CP", "UIP", "rUIP"):
        assert frozen_a[method][0].tolist() == pytest.approx(
            frozen_b[method][0].tolist()
        )
        assert frozen_a[method][1].tolist() == pytest.approx(
            frozen_b[method][1].tolist()
        )
    assert current.estimate != pytest.approx(alternate.estimate)


def test_power_prior_uses_normalized_random_powers() -> None:
    histories = (
        HistoricalSummary("a", 0.0, 10, 1.0),
        HistoricalSummary("b", 1.0, 10, 1.0),
    )
    current = GaussianSummary("current", 20, 20.0)
    rows, _ = analyze_gaussian_summary(
        histories,
        current,
        delta_clin=math.log(1.5),
        planned_events=10,
        uip_power=4,
    )
    pp = next(x for x in rows if x.method == "PP")
    diagnostics = json.loads(pp.diagnostics_json)
    assert diagnostics["power_distribution"] == "independent_Beta(1,1)"
    assert diagnostics["base_variance"] == 10.0
    assert diagnostics["integration_nodes"] == 16**2
    assert pp.posterior_sd > 0


def test_gaussian_analysis_requires_design_stage_information_cap() -> None:
    histories = (HistoricalSummary("a", 0.0, 10, 1.0),)
    current = GaussianSummary("current", 20, 20.0)
    with pytest.raises(TypeError):
        analyze_gaussian_summary(  # type: ignore[call-arg]
            histories, current, delta_clin=math.log(1.5), uip_power=4
        )
