"""Paired sensitivity analysis for the final capped-medoid rUIP.

This runner changes only the prespecified global transport scale.  It reuses
the data-generating processes, exact current likelihoods, scenarios, seeds,
and posterior integration from ``run_outcome_simulations.py``.  Every value
of ``delta_clin`` is evaluated on the same generated dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "run_outcome_simulations.py"
DEFAULT_OUTPUT = ROOT / "results" / "delta_clin_sensitivity"
DELTA_VALUES = (0.10, 0.25, 0.50)


def load_base():
    spec = importlib.util.spec_from_file_location(
        "tri_outcome_base_76_delta", BASE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "archived/unavailable"


def ruip_prior(dataset, delta_clin: float):
    """Rebuild only the history-based prior at the requested transport scale."""
    """Final capped-medoid rUIP with an explicit transport scale."""
    h = dataset.history_estimate
    information = dataset.history_information
    objectives = np.abs(h[:, None] - h[None, :]).sum(axis=1)
    anchor = min(range(h.size), key=lambda k: (objectives[k], -information[k], k))
    variance = 1.0 / information + 1.0 / information[anchor]
    compatibility = np.exp(-0.5 * (h - h[anchor]) ** 2 / variance)
    compatibility[anchor] = 1.0
    retained = np.minimum(information, information[anchor]) * compatibility
    retained[anchor] = information[anchor]
    retained_information = float(retained.sum())
    weights = retained / retained_information
    center = float(np.dot(weights, h))
    prior_variance = 1.0 / retained_information + delta_clin**2
    return BASE.Prior(
        np.array([center]), np.array([prior_variance]), np.array([0.0]), weights
    )


def analyze_prior(dataset, prior):
    if dataset.outcome != "continuous":
        return BASE.numerical_posterior(dataset, prior)
    current_variance = 1.0 / dataset.current_information
    post_variance = 1.0 / (1.0 / prior.variances + dataset.current_information)
    post_mean = post_variance * (
        prior.means / prior.variances
        + dataset.current_information * dataset.current_estimate
    )
    log_weight = prior.log_weights - 0.5 * (
        np.log(2.0 * np.pi * (prior.variances + current_variance))
        + (dataset.current_estimate - prior.means) ** 2
        / (prior.variances + current_variance)
    )
    weight = np.exp(log_weight - logsumexp(log_weight))
    mean = float(np.dot(weight, post_mean))
    variance = float(np.dot(weight, post_variance + (post_mean - mean) ** 2))
    lower, upper = BASE.normal_mixture_quantiles(
        BASE.Prior(post_mean, post_variance, np.log(weight))
    )
    return mean, variance, lower, upper, 0.0


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["outcome", "scenario_id", "global_conflict", "local_conflict", "delta_clin"]
    for key, frame in raw.groupby(keys, sort=False):
        error = frame.error.to_numpy(float)
        squared = error**2
        rmse = math.sqrt(float(squared.mean()))
        covered = frame.covered.to_numpy(float)
        coverage = float(covered.mean())
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "replications": len(frame),
                "bias": float(error.mean()),
                "bias_mcse": float(error.std(ddof=1) / math.sqrt(len(frame))),
                "rmse": rmse,
                "rmse_mcse": float(
                    squared.std(ddof=1) / math.sqrt(len(frame)) / (2 * rmse)
                ),
                "coverage": coverage,
                "coverage_mcse": math.sqrt(coverage * (1 - coverage) / len(frame)),
                "mean_interval_width": float(frame.interval_width.mean()),
                "width_mcse": float(
                    frame.interval_width.std(ddof=1) / math.sqrt(len(frame))
                ),
                "maximum_normalization_mass_discrepancy": float(
                    frame.normalization_mass_discrepancy.max()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Regenerate each outcome once per paired replication and change only the
    clinically specified transport standard deviation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BASE.SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--outcomes",
        nargs="+",
        choices=BASE.ALL_OUTCOMES,
        default=list(BASE.ALL_OUTCOMES),
    )
    args = parser.parse_args()
    if args.replications <= 1:
        parser.error("--replications must exceed one")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows = []
    for outcome in args.outcomes:
        outcome_index = BASE.ALL_OUTCOMES.index(outcome)
        for scenario_index, scenario in enumerate(BASE.SCENARIOS):
            for replicate in range(args.replications):
                sequence = np.random.SeedSequence(
                    args.seed, spawn_key=(outcome_index, scenario_index, replicate)
                )
                dataset = BASE.make_dataset(
                    outcome, scenario, np.random.default_rng(sequence)
                )
                for delta_clin in DELTA_VALUES:
                    prior = ruip_prior(dataset, delta_clin)
                    mean, _, lower, upper, discrepancy = analyze_prior(dataset, prior)
                    error = mean - dataset.theta
                    rows.append(
                        {
                            "outcome": outcome,
                            "scenario_id": scenario.scenario_id,
                            "global_conflict": scenario.global_conflict,
                            "local_conflict": scenario.local_conflict,
                            "replicate_id": replicate,
                            "delta_clin": delta_clin,
                            "posterior_mean": mean,
                            "error": error,
                            "squared_error": error**2,
                            "covered": int(lower <= dataset.theta <= upper),
                            "interval_width": upper - lower,
                            "normalization_mass_discrepancy": discrepancy,
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw_path = output / "replicate_results.csv.gz"
    summary_path = output / "scenario_summary.csv"
    raw.to_csv(raw_path, index=False, compression="gzip")
    summary.to_csv(summary_path, index=False)
    manifest = {
        "protocol": "final-capped-medoid-delta-clin-sensitivity-v1",
        "method": "final capped-medoid rUIP",
        "runner": str(Path(__file__).relative_to(ROOT)),
        "base_runner": str(BASE_PATH.relative_to(ROOT)),
        "git_head": git_head(),
        "seed": args.seed,
        "replications": args.replications,
        "outcomes": list(args.outcomes),
        "delta_clin_values": list(DELTA_VALUES),
        "scenarios": [item.scenario_id for item in BASE.SCENARIOS],
        "paired_design": "all margins evaluated on every generated dataset",
        "selection_policy": (
            "0.25 remains the prespecified primary scale; sensitivity results are "
            "reported in full and are not used to relabel the primary analysis"
        ),
        "elapsed_seconds": time.perf_counter() - start,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "files": {
            raw_path.name: {
                "sha256": sha256(raw_path),
                "bytes": raw_path.stat().st_size,
            },
            summary_path.name: {
                "sha256": sha256(summary_path),
                "bytes": summary_path.stat().st_size,
            },
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
