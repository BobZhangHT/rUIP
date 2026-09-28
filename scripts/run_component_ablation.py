"""Paired, one-component-at-a-time rUIP ablations for three outcomes.

This mechanism experiment reuses the common outcome generator and posterior
integration. Each variant changes one rUIP component at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ruip.simulation.precision_leverage import make_dataset

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_PATH = ROOT / "scripts" / "run_outcome_simulations.py"
OUTPUT_ROOT = ROOT / "results" / "component_ablation_20260928"
OUTCOMES = ("continuous", "binary", "survival")
INFORMATION = np.array([300.0, 50.0, 80.0])
METHODS = ("rUIP", "weighted-anchor", "no-score", "no-cap", "no-global")
SEED = 202609281011
REPLICATIONS = 1000

# Fixed before inspecting any results from this component experiment.
SCENARIOS = (
    ("compatible", 0.0, 0.0),
    ("global-positive", 0.30, 0.0),
    ("global-negative", -0.30, 0.0),
    ("local-positive", 0.0, 0.35),
    ("local-negative", 0.0, -0.35),
    ("joint-positive", 0.15, 0.35),
    ("joint-negative", -0.15, -0.35),
)


def load_authority():
    spec = importlib.util.spec_from_file_location(
        "ablation_tri_authority", AUTHORITY_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load authoritative outcome implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tri = load_authority()


def component_prior(dataset, method: str):
    """Change exactly one rUIP component, retaining the Gaussian prior form."""
    if method not in METHODS:
        raise ValueError(method)
    h = np.asarray(dataset.history_estimate, float)
    info = np.asarray(dataset.history_information, float)
    if method == "weighted-anchor":
        objectives = (np.abs(h[:, None] - h[None, :]) * info[None, :]).sum(axis=1)
    else:
        objectives = np.abs(h[:, None] - h[None, :]).sum(axis=1)
    anchor = min(range(h.size), key=lambda k: (objectives[k], -info[k], k))
    variance = 1.0 / info + 1.0 / info[anchor]
    scores = np.exp(-0.5 * (h - h[anchor]) ** 2 / variance)
    scores[anchor] = 1.0
    if method == "no-score":
        scores[:] = 1.0
    retained = info.copy() if method == "no-cap" else np.minimum(info, info[anchor])
    retained *= scores
    retained[anchor] = info[anchor]
    total = float(retained.sum())
    weights = retained / total
    center = float(np.dot(weights, h))
    prior_variance = 1.0 / total
    if method != "no-global":
        prior_variance += tri.DELTA_CLIN**2
    prior = tri.Prior(
        np.array([center]),
        np.array([prior_variance]),
        np.array([0.0]),
        weights,
    )
    return prior, anchor


def analyze_prior(dataset, prior):
    """Use the same Gaussian conjugacy or numerical integration as the authority."""
    if dataset.outcome != "continuous":
        return tri.numerical_posterior(dataset, prior)
    prior_precision = 1.0 / float(prior.variances[0])
    posterior_variance = 1.0 / (dataset.current_information + prior_precision)
    posterior_mean = posterior_variance * (
        dataset.current_information * dataset.current_estimate
        + prior_precision * float(prior.means[0])
    )
    half_width = 1.959963984540054 * math.sqrt(posterior_variance)
    return (
        posterior_mean,
        posterior_variance,
        posterior_mean - half_width,
        posterior_mean + half_width,
        0.0,
    )


def run(
    outcome: str, replications: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate paired histories and current data from the frozen generator."""
    rows: list[dict] = []
    checks: list[dict] = []
    outcome_index = OUTCOMES.index(outcome)
    for scenario_index, (name, global_shift, local_shift) in enumerate(SCENARIOS):
        scenario = tri.Scenario(
            name,
            "component-ablation",
            name,
            global_shift,
            (local_shift, 0.0, 0.0),
        )
        for replication in range(replications):
            sequence = np.random.SeedSequence(
                seed, spawn_key=(outcome_index, scenario_index, replication)
            )
            dataset = make_dataset(
                tri, outcome, scenario, INFORMATION, np.random.default_rng(sequence)
            )
            for method in METHODS:
                prior, anchor = component_prior(dataset, method)
                if replication < 3:
                    changed = replace(
                        dataset,
                        current_estimate=dataset.current_estimate + 0.41,
                        current_information=dataset.current_information * 0.79,
                        current_events=max(1, dataset.current_events - 2),
                    )
                    revised, _ = component_prior(changed, method)
                    checks.append(
                        {
                            "outcome": outcome,
                            "scenario": name,
                            "method": method,
                            "passed": tri.prior_fingerprint(prior)
                            == tri.prior_fingerprint(revised),
                        }
                    )
                mean, _, lower, upper, discrepancy = analyze_prior(dataset, prior)
                rows.append(
                    {
                        "outcome": outcome,
                        "scenario": name,
                        "replication": replication,
                        "method": method,
                        "estimate": mean,
                        "squared_error": (mean - dataset.theta) ** 2,
                        "covered": int(lower <= dataset.theta <= upper),
                        "width": upper - lower,
                        "prior_center": float(prior.means[0]),
                        "prior_precision": 1.0 / float(prior.variances[0]),
                        "discordant_source_weight": float(prior.local_weights[0]),
                        "anchor": anchor + 1,
                        "normalization_mass_discrepancy": discrepancy,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(checks)


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = ["outcome", "scenario", "method"]
    summary = (
        raw.groupby(groups, sort=False)
        .agg(
            replications=("replication", "count"),
            mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            width=("width", "mean"),
            prior_precision=("prior_precision", "mean"),
            discordant_source_weight=("discordant_source_weight", "mean"),
            max_mass_discrepancy=("normalization_mass_discrepancy", "max"),
        )
        .reset_index()
    )
    wide = raw.pivot(
        index=["outcome", "scenario", "replication"],
        columns="method",
        values=["squared_error", "covered", "width"],
    )
    paired: list[dict] = []
    for (outcome, scenario), block in wide.groupby(level=[0, 1], sort=False):
        baseline = block["squared_error"]["rUIP"].to_numpy()
        base_coverage = block["covered"]["rUIP"].to_numpy()
        base_width = block["width"]["rUIP"].to_numpy()
        for method in METHODS[1:]:
            candidate = block["squared_error"][method].to_numpy()
            mse_difference = baseline - candidate
            paired.append(
                {
                    "outcome": outcome,
                    "scenario": scenario,
                    "ablation": method,
                    "mse_ratio_ablation_over_ruip": float(
                        candidate.mean() / baseline.mean()
                    ),
                    "paired_mse_difference_ruip_minus_ablation": float(
                        mse_difference.mean()
                    ),
                    "paired_mse_difference_mcse": float(
                        mse_difference.std(ddof=1) / math.sqrt(len(mse_difference))
                    ),
                    "coverage_difference_ablation_minus_ruip": float(
                        block["covered"][method].mean() - base_coverage.mean()
                    ),
                    "width_difference_ablation_minus_ruip": float(
                        block["width"][method].mean() - base_width.mean()
                    ),
                }
            )
    return summary, pd.DataFrame(paired)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Evaluate full rUIP and one-component variants on identical generated
    datasets so each difference has a clear mechanism interpretation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", choices=OUTCOMES, required=True)
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("replications must be at least 2")
    output = args.output_root / args.outcome
    output.mkdir(parents=True, exist_ok=True)
    raw, checks = run(args.outcome, args.replications, args.seed)
    summary, paired = summarize(raw)
    raw.to_csv(output / "replicate_results.csv.gz", index=False, compression="gzip")
    summary.to_csv(output / "scenario_method_summary.csv", index=False)
    paired.to_csv(output / "paired_component_summary.csv", index=False)
    checks.to_csv(output / "prior_invariance.csv", index=False)
    manifest = {
        "status": "PASS"
        if bool(checks.passed.all())
        and bool(
            np.isfinite(
                raw.select_dtypes(include="number").drop(columns=["anchor"]).to_numpy()
            ).all()
        )
        and raw.normalization_mass_discrepancy.max() < 1e-4
        else "FAIL",
        "role": "supplementary post-review component ablation; exploratory",
        "outcome": args.outcome,
        "seed": args.seed,
        "replications": args.replications,
        "historical_target_information": INFORMATION.tolist(),
        "scenarios": [list(item) for item in SCENARIOS],
        "methods": list(METHODS),
        "rows": len(raw),
        "prior_invariance_passed": bool(checks.passed.all()),
        "max_mass_discrepancy": float(raw.normalization_mass_discrepancy.max()),
        "authority_sha256": sha256(AUTHORITY_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "files": {
            name: sha256(output / name)
            for name in (
                "replicate_results.csv.gz",
                "scenario_method_summary.csv",
                "paired_component_summary.csv",
                "prior_invariance.csv",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": manifest["status"], "outcome": args.outcome, "rows": len(raw)}
        )
    )


if __name__ == "__main__":
    main()
