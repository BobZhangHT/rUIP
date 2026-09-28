"""Confirm the prespecified high-precision conflict test for capped-medoid rUIP.

The conflicting historical source has information 300, whereas the two sources
that agree with one another have information 50 and 80. All methods use the same
paired datasets. The grid contains compatibility, common-shift checks,
and all sign combinations of a moderate common shift with an isolated local
departure.  Current outcomes never enter prior construction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ruip.simulation.precision_leverage import (
    analyze,
    make_dataset,
    prior_invariant_to_current,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_PATH = ROOT / "scripts" / "run_outcome_simulations.py"
METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP")


def load_authority():
    spec = importlib.util.spec_from_file_location(
        "precision_leverage_authority", AUTHORITY_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the validated three-outcome implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tri = load_authority()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenarios() -> list[dict]:
    rows = [
        {"scenario_id": "compatible", "global_shift": 0.0, "local_shift": 0.0},
        {"scenario_id": "global-negative", "global_shift": -0.15, "local_shift": 0.0},
        {"scenario_id": "global-positive", "global_shift": 0.15, "local_shift": 0.0},
    ]
    for global_sign in (-1, 1):
        for local_sign in (-1, 1):
            rows.append(
                {
                    "scenario_id": f"joint-g{global_sign:+d}-l{local_sign:+d}",
                    "global_shift": 0.15 * global_sign,
                    "local_shift": 0.75 * local_sign,
                }
            )
    return rows


def run(
    outcome: str, replications: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    abi, dll = tri.load_frozen_engine()
    information = np.array([300.0, 50.0, 80.0])
    rows: list[dict] = []
    checks: list[dict] = []
    for scenario_index, item in enumerate(scenarios()):
        local = (item["local_shift"], 0.0, 0.0)
        scenario = tri.Scenario(
            item["scenario_id"],
            "precision-leverage",
            "targeted",
            item["global_shift"],
            local,
        )
        for replication in range(replications):
            sequence = np.random.SeedSequence(
                seed, spawn_key=(scenario_index, replication)
            )
            dataset = make_dataset(
                tri, outcome, scenario, information, np.random.default_rng(sequence)
            )
            if replication < 3:
                for method in METHODS:
                    checks.append(
                        {
                            "outcome": outcome,
                            "scenario_id": item["scenario_id"],
                            "method": method,
                            "passed": prior_invariant_to_current(
                                tri, dataset, method, abi, dll
                            ),
                        }
                    )
            for method in METHODS:
                row = {
                    **item,
                    "outcome": outcome,
                    "replication": replication,
                    "method": method,
                }
                try:
                    mean, variance, lower, upper, discrepancy, prior = analyze(
                        tri, dataset, method, abi, dll
                    )
                    values = np.array([mean, variance, lower, upper, discrepancy])
                    if not np.all(np.isfinite(values)) or variance < 0 or lower > upper:
                        raise FloatingPointError("invalid posterior summary")
                    error = mean - dataset.theta
                    row.update(
                        status="success",
                        estimate=mean,
                        posterior_variance=variance,
                        interval_lower=lower,
                        interval_upper=upper,
                        error=error,
                        squared_error=error**2,
                        covered=int(lower <= dataset.theta <= upper),
                        interval_width=upper - lower,
                        normalization_mass_discrepancy=discrepancy,
                        prior_precision=(
                            float(1.0 / prior.variances[0])
                            if prior is not None and prior.means.size == 1
                            else math.nan
                        ),
                    )
                except Exception as exc:
                    row.update(
                        status="failure",
                        failure_type=type(exc).__name__,
                        failure_message=str(exc),
                        estimate=math.nan,
                        posterior_variance=math.nan,
                        interval_lower=math.nan,
                        interval_upper=math.nan,
                        error=math.nan,
                        squared_error=math.nan,
                        covered=math.nan,
                        interval_width=math.nan,
                        normalization_mass_discrepancy=math.nan,
                        prior_precision=math.nan,
                    )
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = raw[raw.status.eq("success")]
    summary = (
        success.groupby(["outcome", "scenario_id", "method"], sort=True)
        .agg(
            replications=("replication", "count"),
            bias=("error", "mean"),
            mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            width=("interval_width", "mean"),
            prior_precision=("prior_precision", "mean"),
        )
        .reset_index()
    )
    summary["rmse"] = np.sqrt(summary.mse)
    keys = ["outcome", "scenario_id", "replication"]
    wide = success.pivot(index=keys, columns="method", values="squared_error")
    paired_rows = []
    for (outcome, scenario_id), frame in wide.groupby(level=[0, 1], sort=True):
        for comparator in METHODS:
            if comparator == "rUIP":
                continue
            delta = (frame["rUIP"] - frame[comparator]).dropna().to_numpy(float)
            paired_rows.append(
                {
                    "outcome": outcome,
                    "scenario_id": scenario_id,
                    "comparator": comparator,
                    "paired_replications": delta.size,
                    "mean_mse_difference_ruip_minus_comparator": float(delta.mean()),
                    "mcse_mse_difference": float(
                        delta.std(ddof=1) / math.sqrt(delta.size)
                    ),
                    "probability_ruip_lower_squared_error": float(np.mean(delta < 0)),
                }
            )
    return summary, pd.DataFrame(paired_rows)


def main() -> None:
    """Vary the disagreement of one high-information historical source while
    keeping the compatible sources and paired random streams fixed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", required=True, choices=tri.ALL_OUTCOMES)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026091994)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("replications must exceed one")
    started = time.perf_counter()
    raw, checks = run(args.outcome, args.replications, args.seed)
    summary, paired = summarize(raw)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "replicate_results.csv.gz", index=False, compression="gzip")
    summary.to_csv(output / "scenario_method_summary.csv", index=False)
    paired.to_csv(output / "paired_mse_summary.csv", index=False)
    checks.to_csv(output / "prior_invariance.csv", index=False)
    failures = int(raw.status.ne("success").sum())
    maximum_discrepancy = float(raw.normalization_mass_discrepancy.max())
    passed = failures == 0 and checks.passed.all() and maximum_discrepancy < 1e-4
    manifest = {
        "status": "PASS" if passed else "FAIL",
        "protocol": "precision-leverage-confirmation-v1",
        "outcome": args.outcome,
        "replications": args.replications,
        "seed": args.seed,
        "methods": list(METHODS),
        "scenarios": scenarios(),
        "historical_target_information": [300.0, 50.0, 80.0],
        "power_prior": {
            "powers": "three independent Beta(1,1)",
            "historical_likelihood": "normalized Gaussian summaries",
            "base_normal_mean": tri.POWER_BASE_MEAN,
            "base_normal_sd": tri.POWER_BASE_SD,
            "tensor_gauss_legendre_order": tri.POWER_QUADRATURE_ORDER,
        },
        "datasets": len(scenarios()) * args.replications,
        "analyses": len(raw),
        "failure_count": failures,
        "prior_invariance_failures": int((~checks.passed).sum()),
        "maximum_normalization_mass_discrepancy": maximum_discrepancy,
        "runner_sha256": sha256(Path(__file__)),
        "authority_sha256": sha256(AUTHORITY_PATH),
        "python": platform.python_version(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
