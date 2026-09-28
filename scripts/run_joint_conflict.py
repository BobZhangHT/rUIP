# ruff: noqa: E501
"""Joint-conflict confirmation runner for the current comparator set.

The runner imports the authoritative outcome generators and analysis routines
from ``run_outcome_simulations.py``.  It intentionally supplies only a new,
prespecified scenario grid and reporting layer; comparator implementations are
never reimplemented here.  Survival data are generated from an exponential
DGP but analyzed exclusively with the imported Cox partial-likelihood code.
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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "joint_conflict_design.json"
AUTHORITY_PATH = ROOT / "scripts" / "run_outcome_simulations.py"
DEFAULT_OUTPUT = ROOT / "results" / "ruip_joint_conflict_confirmation"


def load_authority():
    spec = importlib.util.spec_from_file_location(
        "tri_outcome_authority", AUTHORITY_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load authoritative tri-outcome runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tri = load_authority()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_vector(
    magnitude: float, sign: int, source: int
) -> tuple[float, float, float]:
    values = np.full(3, -sign * magnitude / 3.0)
    values[source] = 2.0 * sign * magnitude / 3.0
    return tuple(float(x) for x in values)


def target_scenarios(config: dict, outcome: str) -> list[dict]:
    scale = config["target_region"][outcome]
    rows: list[dict] = []
    for a, global_magnitude in enumerate(scale["global"]):
        for b, local_magnitude in enumerate(scale["local"]):
            for global_sign in config["target_design"]["global_signs"]:
                for local_sign in config["target_design"]["local_signs"]:
                    rows.append(
                        {
                            "scenario_id": (
                                f"target-g{a + 1}-l{b + 1}-"
                                f"gs{global_sign:+d}-ls{local_sign:+d}"
                            ),
                            "region": "target",
                            "cell_type": "joint_conflict",
                            "global_magnitude": float(global_magnitude),
                            "local_magnitude": float(local_magnitude),
                            "global_sign": global_sign,
                            "local_sign": local_sign,
                            "discordant_source": "rotated_by_replicate",
                            "delta": float(global_sign * global_magnitude),
                            "local_deviation": (0.0, 0.0, 0.0),
                        }
                    )
    return rows


def guardrail_scenarios(config: dict) -> list[dict]:
    rows: list[dict] = []
    for name, values in config["guardrails"].items():
        local = float(values["local"])
        source = 0 if local else None
        rows.append(
            {
                "scenario_id": f"guardrail-{name}",
                "region": "guardrail",
                "cell_type": name,
                "global_magnitude": abs(float(values["global"])),
                "local_magnitude": local,
                "global_sign": 1 if values["global"] else 0,
                "local_sign": 1 if local else 0,
                "discordant_source": None if source is None else source + 1,
                "delta": float(values["global"]),
                "local_deviation": (0.0, 0.0, 0.0)
                if source is None
                else _local_vector(local, 1, source),
            }
        )
    return rows


def scenario_grid(config: dict, outcome: str) -> list[dict]:
    rows = target_scenarios(config, outcome) + guardrail_scenarios(config)
    expected = config["target_design"]["cells_per_outcome"] + len(config["guardrails"])
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} scenarios, found {len(rows)}")
    return rows


def scenario_object(spec: dict):
    return tri.Scenario(
        spec["scenario_id"],
        "joint" if spec["region"] == "target" else "guardrail",
        spec["cell_type"],
        spec["delta"],
        spec["local_deviation"],
    )


def realize_source(spec: dict, replicate_id: int) -> dict:
    """Set the single discordant source for a prespecified replicate mixture."""
    if spec["region"] != "target":
        return spec
    realized = dict(spec)
    source = replicate_id % 3
    local = np.zeros(3)
    local[source] = spec["local_sign"] * spec["local_magnitude"]
    realized["discordant_source"] = source + 1
    realized["local_deviation"] = tuple(float(value) for value in local)
    return realized


def prior_invariance(dataset, method: str, abi, dll) -> bool:
    changed = replace(
        dataset,
        current_estimate=dataset.current_estimate + 0.37,
        current_information=max(1.0, dataset.current_information * 0.83),
        current_events=max(1, dataset.current_events - 3),
        current_exposure=dataset.current_exposure + 2.5,
        current_censored=min(dataset.current_subjects, dataset.current_censored + 3),
    )
    if method == "NIP":
        return True
    if method == "PP":
        return tri.prior_fingerprint(
            tri.normalized_power_prior(dataset)
        ) == tri.prior_fingerprint(tri.normalized_power_prior(changed))
    original = (
        tri.ruip_prior(dataset, method)
        if method.startswith("rUIP")
        else tri.frozen_prior(dataset, method, abi, dll)
    )
    regenerated = (
        tri.ruip_prior(changed, method)
        if method.startswith("rUIP")
        else tri.frozen_prior(changed, method, abi, dll)
    )
    return tri.prior_fingerprint(original) == tri.prior_fingerprint(regenerated)


def outcome_index(config: dict, outcome: str) -> int:
    """Return the protocol-global outcome index, independent of CLI subsetting."""
    return tuple(config["outcomes"]).index(outcome)


def run(config: dict, outcomes: tuple[str, ...], replications: int, seed: int):
    """Keep each generated dataset paired across all comparator methods."""
    abi, dll = tri.load_frozen_engine()
    rows: list[dict] = []
    checks: list[dict] = []
    for outcome in outcomes:
        oi = outcome_index(config, outcome)
        for si, spec in enumerate(scenario_grid(config, outcome)):
            for replicate in range(replications):
                sequence = np.random.SeedSequence(seed, spawn_key=(oi, si, replicate))
                data_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
                realized = realize_source(spec, replicate)
                dataset = tri.make_dataset(
                    outcome, scenario_object(realized), np.random.default_rng(sequence)
                )
                if replicate < min(3, replications):
                    for method in config["methods"]:
                        checks.append(
                            {
                                "outcome": outcome,
                                "scenario_id": spec["scenario_id"],
                                "discordant_source": realized["discordant_source"],
                                "method": method,
                                "passed": prior_invariance(dataset, method, abi, dll),
                            }
                        )
                for method in config["methods"]:
                    row = dict(realized)
                    row.update(
                        outcome=outcome,
                        replicate_id=replicate,
                        data_seed=data_seed,
                        method=method,
                    )
                    try:
                        mean, variance, lower, upper, qerr, prior = tri.analyze(
                            dataset, method, abi, dll
                        )
                        if (
                            not np.all(
                                np.isfinite([mean, variance, lower, upper, qerr])
                            )
                            or variance < 0
                            or lower > upper
                        ):
                            raise FloatingPointError("invalid posterior summary")
                        error = mean - dataset.theta
                        row.update(
                            status="success",
                            estimate=mean,
                            posterior_variance=variance,
                            interval_lower=lower,
                            interval_upper=upper,
                            error=error,
                            squared_error=error * error,
                            covered=int(lower <= dataset.theta <= upper),
                            interval_width=upper - lower,
                            normalization_mass_discrepancy=qerr,
                            prior_precision=float(1.0 / prior.variances[0])
                            if prior is not None and prior.means.size == 1
                            else math.nan,
                        )
                        if prior is not None and prior.local_weights is not None:
                            for k, value in enumerate(prior.local_weights, start=1):
                                row[f"q{k}"] = float(value)
                    except Exception as exc:  # retain all failures for audit
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
                    row.update(
                        actual_current_information=dataset.current_information,
                        current_subjects=dataset.current_subjects,
                        current_events=dataset.current_events,
                        censoring_fraction=(
                            dataset.current_censored / dataset.current_subjects
                            if outcome == "survival"
                            else math.nan
                        ),
                        mean_historical_information=float(
                            dataset.history_information.mean()
                        ),
                    )
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def _summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    success = frame.loc[frame.status == "success"].copy()
    grouped = success.groupby(keys, sort=True, dropna=False)
    result = grouped.agg(
        replications=("replicate_id", "count"),
        mse=("squared_error", "mean"),
        bias=("error", "mean"),
        coverage=("covered", "mean"),
        mean_interval_width=("interval_width", "mean"),
        mean_prior_precision=("prior_precision", "mean"),
        maximum_normalization_mass_discrepancy=(
            "normalization_mass_discrepancy",
            "max",
        ),
    ).reset_index()
    result["rmse"] = np.sqrt(result["mse"])
    return result


def paired_mse(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    success = raw.loc[raw.status == "success"]
    index = ["outcome", "region", "scenario_id", "replicate_id"]
    wide = success.pivot(index=index, columns="method", values="squared_error")
    rows = []
    for (outcome, region), frame in wide.groupby(level=[0, 1], sort=True):
        for comparator in config["methods"]:
            if comparator == "rUIP":
                continue
            difference = (frame["rUIP"] - frame[comparator]).dropna().to_numpy(float)
            rows.append(
                {
                    "outcome": outcome,
                    "region": region,
                    "comparator": comparator,
                    "paired_replications": difference.size,
                    "mean_mse_difference_ruip_minus_comparator": float(
                        difference.mean()
                    ),
                    "mcse_mse_difference": float(
                        difference.std(ddof=1) / math.sqrt(difference.size)
                    )
                    if difference.size > 1
                    else math.nan,
                    "probability_ruip_lower_squared_error": float(
                        np.mean(difference < -1e-14)
                    ),
                }
            )
    return pd.DataFrame(rows)


def pooled_region_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    target = raw[(raw.status == "success") & (raw.region == "target")].copy()
    # Equal scenario weighting: average each 1,000-replication scenario before pooling.
    cell = _summary(target, ["outcome", "scenario_id", "method"])
    pooled = (
        cell.groupby(["outcome", "method"], sort=True)
        .agg(
            target_scenarios=("scenario_id", "count"),
            mse=("mse", "mean"),
            bias=("bias", "mean"),
            coverage=("coverage", "mean"),
            mean_interval_width=("mean_interval_width", "mean"),
            mean_prior_precision=("mean_prior_precision", "mean"),
        )
        .reset_index()
    )
    pooled["rmse"] = np.sqrt(pooled["mse"])
    nip = pooled.loc[pooled.method == "NIP", ["outcome", "mse"]].rename(
        columns={"mse": "nip_mse"}
    )
    return pooled.merge(nip, on="outcome", how="left").assign(
        standardized_mse=lambda x: x.mse / x.nip_mse
    )


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust one-sided p-values within one prespecified family."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    size = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (size - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def _wilson_lower(successes: int, total: int, alpha: float) -> float:
    """One-sided Wilson lower bound with tail probability ``alpha``."""
    if total <= 0:
        return math.nan
    p = successes / total
    z = float(norm.ppf(1.0 - alpha))
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (center - radius) / denominator


def _target_arrays(raw: pd.DataFrame, methods: list[str]):
    target = raw[(raw.status == "success") & (raw.region == "target")].copy()
    arrays: dict[tuple[str, str], np.ndarray] = {}
    coverage: dict[tuple[str, str], np.ndarray] = {}
    for (outcome, scenario_id), frame in target.groupby(
        ["outcome", "scenario_id"], sort=True
    ):
        squared = frame.pivot(
            index="replicate_id", columns="method", values="squared_error"
        )
        covered = frame.pivot(index="replicate_id", columns="method", values="covered")
        if list(squared.columns.intersection(methods)) != methods:
            squared = squared.reindex(columns=methods)
            covered = covered.reindex(columns=methods)
        if squared.isna().any().any() or covered.isna().any().any():
            raise ValueError(
                f"incomplete paired target stratum: {outcome}/{scenario_id}"
            )
        arrays[(str(outcome), str(scenario_id))] = squared.to_numpy(float)
        coverage[(str(outcome), str(scenario_id))] = covered.to_numpy(float)
    return arrays, coverage


def confirmation_analysis(raw: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the current comparison gates once all outcomes are present."""
    analysis = config["analysis"]
    methods = list(config["methods"])
    outcomes = list(config["outcomes"])
    method_index = {method: index for index, method in enumerate(methods)}
    arrays, coverage_arrays = _target_arrays(raw, methods)
    expected_strata = len(outcomes) * config["target_design"]["cells_per_outcome"]
    if len(arrays) != expected_strata:
        raise ValueError(
            f"expected {expected_strata} target strata, found {len(arrays)}"
        )

    # Point estimates give every scenario and every outcome equal weight.
    outcome_mse: dict[str, np.ndarray] = {}
    outcome_coverage: dict[str, np.ndarray] = {}
    for outcome in outcomes:
        outcome_keys = [key for key in arrays if key[0] == outcome]
        outcome_mse[outcome] = np.mean(
            [arrays[key].mean(axis=0) for key in outcome_keys], axis=0
        )
        outcome_coverage[outcome] = np.mean(
            [coverage_arrays[key].mean(axis=0) for key in outcome_keys], axis=0
        )
    nip_index = method_index["NIP"]
    ruip_index = method_index["rUIP"]
    standardized = np.vstack(
        [outcome_mse[outcome] / outcome_mse[outcome][nip_index] for outcome in outcomes]
    )
    combined_standardized = standardized.mean(axis=0)
    point_difference = combined_standardized[ruip_index] - combined_standardized
    point_ratio = combined_standardized[ruip_index] / combined_standardized

    bootstrap_replications = int(analysis["bootstrap_replications"])
    rng = np.random.default_rng(int(analysis["bootstrap_seed"]))
    bootstrap_mse = np.zeros((bootstrap_replications, len(outcomes), len(methods)))
    for oi, outcome in enumerate(outcomes):
        outcome_keys = [key for key in arrays if key[0] == outcome]
        for key in outcome_keys:
            values = arrays[key]
            n = values.shape[0]
            # Multinomial counts retain pairing across all methods in a stratum.
            counts = rng.multinomial(
                n, np.full(n, 1.0 / n), size=bootstrap_replications
            )
            bootstrap_mse[:, oi, :] += counts @ values / n
        bootstrap_mse[:, oi, :] /= len(outcome_keys)
    bootstrap_standardized = bootstrap_mse / bootstrap_mse[:, :, [nip_index]]
    bootstrap_combined = bootstrap_standardized.mean(axis=1)
    bootstrap_difference = bootstrap_combined[:, [ruip_index]] - bootstrap_combined
    bootstrap_ratio = bootstrap_combined[:, [ruip_index]] / bootstrap_combined

    superiority = list(analysis["superiority_family"])
    ablations = list(analysis["ablation_family"])
    raw_p: dict[str, float] = {}
    for comparator in superiority + ablations:
        ci = method_index[comparator]
        standard_error = float(bootstrap_difference[:, ci].std(ddof=1))
        raw_p[comparator] = (
            float(norm.cdf(point_difference[ci] / standard_error))
            if standard_error > 0
            else float(point_difference[ci] >= 0)
        )
    superiority_adjusted = _holm_adjust(
        {method: raw_p[method] for method in superiority}
    )
    ablation_adjusted = _holm_adjust({method: raw_p[method] for method in ablations})

    alpha = float(analysis["familywise_alpha"])
    superiority_upper_q = 1.0 - alpha / len(superiority)
    ablation_upper_q = 1.0 - alpha / len(ablations)
    comparison_rows: list[dict] = []
    for comparator in superiority + ablations:
        ci = method_index[comparator]
        family = "superiority" if comparator in superiority else "ablation"
        upper_q = superiority_upper_q if family == "superiority" else ablation_upper_q
        comparison_rows.append(
            {
                "comparator": comparator,
                "family": family,
                "combined_standardized_mse_difference": float(point_difference[ci]),
                "combined_mse_ratio": float(point_ratio[ci]),
                "bootstrap_standard_error_difference": float(
                    bootstrap_difference[:, ci].std(ddof=1)
                ),
                "one_sided_p": raw_p[comparator],
                "holm_adjusted_p": (
                    superiority_adjusted[comparator]
                    if family == "superiority"
                    else ablation_adjusted[comparator]
                ),
                "familywise_one_sided_ratio_upper": float(
                    np.quantile(bootstrap_ratio[:, ci], upper_q)
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    coverage_rows = []
    coverage_alpha = alpha / len(outcomes)
    for oi, outcome in enumerate(outcomes):
        values = np.concatenate(
            [
                coverage_arrays[key][:, ruip_index]
                for key in coverage_arrays
                if key[0] == outcome
            ]
        )
        coverage_rows.append(
            {
                "outcome": outcome,
                "coverage": float(outcome_coverage[outcome][ruip_index]),
                "bonferroni_one_sided_wilson_lower": _wilson_lower(
                    int(values.sum()), int(values.size), coverage_alpha
                ),
                "replications": int(values.size),
            }
        )

    classic = list(analysis["classic_methods"])
    calibrated_classic = []
    for method in classic:
        mi = method_index[method]
        method_ok = True
        for outcome in outcomes:
            values = np.concatenate(
                [
                    coverage_arrays[key][:, mi]
                    for key in coverage_arrays
                    if key[0] == outcome
                ]
            )
            point = float(outcome_coverage[outcome][mi])
            lower = _wilson_lower(int(values.sum()), int(values.size), coverage_alpha)
            lo, hi = analysis["coverage_point_interval"]
            method_ok &= (
                lo <= point <= hi
                and lower > analysis["coverage_bonferroni_lower_minimum"]
            )
        if method_ok:
            calibrated_classic.append(method)
    strongest = (
        min(
            calibrated_classic,
            key=lambda method: combined_standardized[method_index[method]],
        )
        if calibrated_classic
        else None
    )

    lo, hi = map(float, analysis["coverage_point_interval"])
    coverage_pass = all(
        lo <= row["coverage"] <= hi
        and row["bonferroni_one_sided_wilson_lower"]
        > float(analysis["coverage_bonferroni_lower_minimum"])
        for row in coverage_rows
    )
    superiority_pass = all(
        superiority_adjusted[method] < alpha
        and point_difference[method_index[method]] < 0
        for method in analysis["classic_methods"]
    )
    strongest_ratio = (
        float(point_ratio[method_index[strongest]])
        if strongest is not None
        else math.nan
    )
    strongest_upper = (
        float(
            comparisons.loc[
                comparisons.comparator == strongest,
                "familywise_one_sided_ratio_upper",
            ].iloc[0]
        )
        if strongest is not None
        else math.nan
    )
    practical_reduction = 1.0 - strongest_ratio if strongest is not None else math.nan
    outcome_ratios = (
        {
            outcome: float(
                outcome_mse[outcome][ruip_index]
                / outcome_mse[outcome][method_index[strongest]]
            )
            for outcome in outcomes
        }
        if strongest is not None
        else {}
    )
    ablation_rows = comparisons[comparisons.family == "ablation"]
    gates = {
        "coverage": coverage_pass,
        "statistical_superiority_to_all_classic": bool(
            superiority_pass and strongest is not None and strongest_upper < 1.0
        ),
        "practical_mse_reduction_at_least_5_percent": bool(
            strongest is not None
            and practical_reduction >= analysis["practical_mse_reduction_minimum"]
        ),
        "headline_mse_reduction_at_least_7_5_percent": bool(
            strongest is not None
            and practical_reduction >= analysis["headline_mse_reduction_target"]
        ),
        "cross_outcome_consistency": bool(
            strongest is not None
            and all(ratio < 1.0 for ratio in outcome_ratios.values())
            and all(
                ratio <= analysis["maximum_outcome_mse_ratio"]
                for ratio in outcome_ratios.values()
            )
        ),
        "ablations": bool(
            (ablation_rows.combined_standardized_mse_difference < 0).all()
            and (ablation_rows.holm_adjusted_p < alpha).all()
            and coverage_pass
        ),
    }
    report = {
        "analysis_status": "COMPLETE",
        "analysis_design_status": "post-review comparator-set reanalysis",
        "bootstrap_seed": int(analysis["bootstrap_seed"]),
        "bootstrap_replications": bootstrap_replications,
        "familywise_alpha": alpha,
        "coverage": coverage_rows,
        "calibrated_classic_methods": calibrated_classic,
        "strongest_calibrated_classic_method": strongest,
        "strongest_classic_combined_mse_ratio": strongest_ratio,
        "strongest_classic_familywise_ratio_upper": strongest_upper,
        "practical_mse_reduction": practical_reduction,
        "outcome_mse_ratios_vs_strongest_classic": outcome_ratios,
        "gates": gates,
        "all_current_gates_pass": all(
            gates[name]
            for name in (
                "coverage",
                "statistical_superiority_to_all_classic",
                "practical_mse_reduction_at_least_5_percent",
                "cross_outcome_consistency",
                "ablations",
            )
        ),
        "interpretation_rule": (
            "Failure of any primary gate downgrades the result to mechanism or "
            "exploratory evidence; the headline 7.5% target is reported separately."
        ),
    }
    return comparisons, report


def write_outputs(
    output: Path,
    raw: pd.DataFrame,
    checks: pd.DataFrame,
    config: dict,
    outcomes: tuple[str, ...],
    replications: int,
    seed: int,
    sentinel: bool,
    elapsed: float,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    scenario_rows = []
    for outcome in outcomes:
        scenario_rows.extend(
            {"outcome": outcome, **row} for row in scenario_grid(config, outcome)
        )
    paths = {
        "replicate_results.csv.gz": output / "replicate_results.csv.gz",
        "scenario_method_summary.csv": output / "scenario_method_summary.csv",
        "pooled_region_metrics.csv": output / "pooled_region_metrics.csv",
        "paired_mse_summary.csv": output / "paired_mse_summary.csv",
        "prior_invariance.csv": output / "prior_invariance.csv",
        "scenarios.csv": output / "scenarios.csv",
    }
    raw.to_csv(paths["replicate_results.csv.gz"], index=False, compression="gzip")
    _summary(raw, ["outcome", "region", "scenario_id", "cell_type", "method"]).to_csv(
        paths["scenario_method_summary.csv"], index=False
    )
    pooled_region_metrics(raw).to_csv(paths["pooled_region_metrics.csv"], index=False)
    paired_mse(raw, config).to_csv(paths["paired_mse_summary.csv"], index=False)
    checks.to_csv(paths["prior_invariance.csv"], index=False)
    pd.DataFrame(scenario_rows).to_csv(paths["scenarios.csv"], index=False)
    confirmation_report = None
    if (
        not sentinel
        and replications == int(config["replications"])
        and tuple(outcomes) == tuple(config["outcomes"])
    ):
        comparisons, confirmation_report = confirmation_analysis(raw, config)
        paths["confirmation_comparisons.csv"] = output / "confirmation_comparisons.csv"
        paths["confirmation_gate_report.json"] = (
            output / "confirmation_gate_report.json"
        )
        comparisons.to_csv(paths["confirmation_comparisons.csv"], index=False)
        paths["confirmation_gate_report.json"].write_text(
            json.dumps(confirmation_report, indent=2) + "\n", encoding="utf-8"
        )
    failures = int((raw.status != "success").sum())
    max_qerr = float(raw.normalization_mass_discrepancy.max())
    passed = (
        failures == 0
        and bool(checks.passed.all())
        and max_qerr < config["analysis"]["normalization_mass_discrepancy_tolerance"]
    )
    manifest = {
        "status": "PASS" if passed else "FAIL",
        "protocol": config["protocol"],
        "sentinel": sentinel,
        "seed": seed,
        "seed_role": "sentinel" if sentinel else "formal",
        "sentinel_seed": config["sentinel_seed"],
        "formal_seed": config["formal_seed"],
        "seed_streams_disjoint_by_root": config["sentinel_seed"]
        != config["formal_seed"],
        "replications": replications,
        "outcomes": list(outcomes),
        "scenarios_per_outcome": config["target_design"]["cells_per_outcome"]
        + len(config["guardrails"]),
        "target_scenarios_per_outcome": config["target_design"]["cells_per_outcome"],
        "guardrail_scenarios_per_outcome": len(config["guardrails"]),
        "methods": config["methods"],
        "datasets": len(outcomes)
        * (config["target_design"]["cells_per_outcome"] + len(config["guardrails"]))
        * replications,
        "analyses": len(raw),
        "failure_count": failures,
        "prior_invariance_checks": len(checks),
        "prior_invariance_failures": int((~checks.passed).sum()),
        "maximum_normalization_mass_discrepancy": max_qerr,
        "normalization_mass_discrepancy_tolerance": config["analysis"][
            "normalization_mass_discrepancy_tolerance"
        ],
        "survival_analysis": "Cox partial likelihood; no baseline hazard is read by the analysis",
        "power_prior": {
            "powers": "three independent Beta(1,1)",
            "historical_likelihood": "normalized Gaussian summaries",
            "base_normal_mean": tri.POWER_BASE_MEAN,
            "base_normal_sd": tri.POWER_BASE_SD,
            "tensor_gauss_legendre_order": tri.POWER_QUADRATURE_ORDER,
        },
        "confirmation_analysis_written": confirmation_report is not None,
        "config_sha256": sha256(CONFIG_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "authority_runner": str(AUTHORITY_PATH.relative_to(ROOT)),
        "authority_runner_sha256": sha256(AUTHORITY_PATH),
        "frozen_comparator_library": str(tri.FROZEN_LIBRARY.relative_to(ROOT)),
        "frozen_comparator_sha256": sha256(tri.FROZEN_LIBRARY),
        "python": platform.python_version(),
        "runtime_seconds": elapsed,
        "files": {},
    }
    for name, path in paths.items():
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def combine_outcome_outputs(input_dirs: list[Path], output: Path, config: dict) -> dict:
    """Combine independently run formal outcomes and apply the frozen gates."""
    manifests = []
    raw_frames = []
    check_frames = []
    elapsed = 0.0
    for directory in input_dirs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        if manifest["sentinel"]:
            raise ValueError(f"cannot combine sentinel output: {directory}")
        if manifest["seed"] != config["formal_seed"]:
            raise ValueError(f"formal seed mismatch in {directory}")
        if manifest["replications"] != config["replications"]:
            raise ValueError(f"formal replication mismatch in {directory}")
        if manifest["config_sha256"] != sha256(CONFIG_PATH):
            raise ValueError(f"config hash mismatch in {directory}")
        if manifest["runner_sha256"] != sha256(Path(__file__)):
            raise ValueError(f"runner hash mismatch in {directory}")
        raw_frames.append(pd.read_csv(directory / "replicate_results.csv.gz"))
        check_frames.append(pd.read_csv(directory / "prior_invariance.csv"))
        elapsed += float(manifest["runtime_seconds"])
    outcomes = tuple(
        outcome
        for outcome in config["outcomes"]
        if any(outcome in manifest["outcomes"] for manifest in manifests)
    )
    if outcomes != tuple(config["outcomes"]):
        raise ValueError(f"combined outcomes {outcomes} do not equal frozen outcomes")
    if sum(len(manifest["outcomes"]) for manifest in manifests) != len(outcomes):
        raise ValueError("duplicate outcome supplied to combine step")
    return write_outputs(
        output,
        pd.concat(raw_frames, ignore_index=True),
        pd.concat(check_frames, ignore_index=True),
        config,
        outcomes,
        int(config["replications"]),
        int(config["formal_seed"]),
        False,
        elapsed,
    )


def main() -> None:
    """Load the shared outcome engine, apply the joint-conflict scenario grid,
    and preserve method pairing before combining outcome-specific runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--outcomes", nargs="+", choices=tri.ALL_OUTCOMES, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sentinel",
        action="store_true",
        help="run the frozen low-replication implementation gate",
    )
    parser.add_argument(
        "--combine-outcome-dirs",
        nargs="+",
        type=Path,
        help="combine disjoint formal outcome runs and execute the frozen analysis gates",
    )
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.combine_outcome_dirs:
        if (
            args.sentinel
            or args.replications is not None
            or args.outcomes
            or args.seed is not None
        ):
            parser.error("combine mode cannot be mixed with run-mode arguments")
        manifest = combine_outcome_outputs(
            [path.resolve() for path in args.combine_outcome_dirs],
            args.output_dir.resolve(),
            config,
        )
        print(json.dumps(manifest, indent=2))
        raise SystemExit(0 if manifest["status"] == "PASS" else 1)
    replications = (
        args.replications
        if args.replications is not None
        else (
            config["sentinel_replications"] if args.sentinel else config["replications"]
        )
    )
    if replications <= 1:
        parser.error("--replications must exceed one")
    outcomes = tuple(args.outcomes or config["outcomes"])
    default_seed = config["sentinel_seed"] if args.sentinel else config["formal_seed"]
    seed = default_seed if args.seed is None else args.seed
    started = time.perf_counter()
    raw, checks = run(config, outcomes, replications, seed)
    manifest = write_outputs(
        args.output_dir.resolve(),
        raw,
        checks,
        config,
        outcomes,
        replications,
        seed,
        args.sentinel,
        time.perf_counter() - started,
    )
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0 if manifest["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
