"""Check that component variants isolate their named change."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_component_ablation.py"
spec = importlib.util.spec_from_file_location("component_ablation_tested", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_full_prior_matches_authoritative_rule():
    tri = module.tri
    scenario = tri.Scenario("check", "check", "check", 0.15, (0.35, 0.0, 0.0))
    dataset = module.make_dataset(
        tri,
        "continuous",
        scenario,
        module.INFORMATION,
        np.random.default_rng(7401),
    )
    new, _ = module.component_prior(dataset, "rUIP")
    old = tri.ruip_prior(dataset, "rUIP")
    np.testing.assert_allclose(new.means, old.means)
    np.testing.assert_allclose(new.variances, old.variances)
    np.testing.assert_allclose(new.local_weights, old.local_weights)


def test_each_variant_preserves_other_parts():
    tri = module.tri
    scenario = tri.Scenario("check", "check", "check", 0.15, (0.35, 0.0, 0.0))
    dataset = module.make_dataset(
        tri,
        "continuous",
        scenario,
        module.INFORMATION,
        np.random.default_rng(7401),
    )
    full, full_anchor = module.component_prior(dataset, "rUIP")
    no_global, no_global_anchor = module.component_prior(dataset, "no-global")
    assert full_anchor == no_global_anchor
    np.testing.assert_allclose(full.means, no_global.means)
    np.testing.assert_allclose(full.local_weights, no_global.local_weights)
    np.testing.assert_allclose(
        full.variances - no_global.variances,
        tri.DELTA_CLIN**2,
    )
    no_score, score_anchor = module.component_prior(dataset, "no-score")
    no_cap, cap_anchor = module.component_prior(dataset, "no-cap")
    assert full_anchor == score_anchor == cap_anchor
    assert no_score.variances[0] <= full.variances[0]
    assert no_cap.variances[0] <= full.variances[0]
