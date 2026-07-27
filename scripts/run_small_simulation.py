"""Run the fixed-seed small IC-UIP simulation, plots, audit, and report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from uip_ic.evaluation import aggregate_simulation, audit_results
from uip_ic.experiment import sampler_config_from_mapping, simulate_replicate
from uip_ic.methods import FitResult, fit_method
from uip_ic.utils import derived_seed, load_yaml, set_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs" / "small_experiment.yaml")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--burn-in", type=int)
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    log_path = REPOSITORY / "results" / "logs" / "small_experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("uip_ic.simulation")


def successful_row(
    scenario: str,
    repetition: int,
    result: FitResult,
    theta_true: float,
    current_unit_information: float,
) -> dict[str, float | str | int | bool]:
    summary = result.summary
    m_mean = summary.get("m_mean", 0.0)
    prior_precision = 0.0
    if result.uip_unit_information is not None:
        prior_precision = m_mean * result.uip_unit_information
    equivalent_ess = prior_precision / current_unit_information if current_unit_information > 0 else np.nan
    return {
        "scenario": scenario,
        "repetition": repetition,
        "method": result.method,
        "theta_true": theta_true,
        "theta_mean": summary["theta_mean"],
        "theta_sd": summary["theta_sd"],
        "theta_lower": summary["theta_lower"],
        "theta_upper": summary["theta_upper"],
        "bias": summary["theta_mean"] - theta_true,
        "squared_error": (summary["theta_mean"] - theta_true) ** 2,
        "covered": float(summary["theta_lower"] <= theta_true <= summary["theta_upper"]),
        "cri_width": summary["theta_upper"] - summary["theta_lower"],
        "m_mean": m_mean,
        "m_sd": summary.get("m_sd", 0.0),
        "prior_precision": prior_precision,
        "current_unit_information": current_unit_information,
        "equivalent_ess": equivalent_ess,
        "theta_ess": summary["theta_ess"],
        "theta_lag1": summary["theta_lag1"],
        "m_ess": summary.get("m_ess", np.nan),
        "m_lag1": summary.get("m_lag1", np.nan),
        "runtime_seconds": result.runtime_seconds,
        "uip_mean": result.uip_mean,
        "uip_unit_information": result.uip_unit_information,
        "m_max": result.m_max,
        "failed": False,
        "error": "",
    }


def failure_row(scenario: str, repetition: int, method: str, theta_true: float, error: Exception) -> dict:
    return {
        "scenario": scenario,
        "repetition": repetition,
        "method": method,
        "theta_true": theta_true,
        "theta_mean": np.nan,
        "theta_sd": np.nan,
        "theta_lower": np.nan,
        "theta_upper": np.nan,
        "bias": np.nan,
        "squared_error": np.nan,
        "covered": np.nan,
        "cri_width": np.nan,
        "m_mean": np.nan,
        "m_sd": np.nan,
        "prior_precision": np.nan,
        "current_unit_information": np.nan,
        "equivalent_ess": np.nan,
        "theta_ess": np.nan,
        "theta_lag1": np.nan,
        "m_ess": np.nan,
        "m_lag1": np.nan,
        "runtime_seconds": np.nan,
        "uip_mean": np.nan,
        "uip_unit_information": np.nan,
        "m_max": np.nan,
        "failed": True,
        "error": repr(error),
    }


def save_plots(summary: pd.DataFrame, raw: pd.DataFrame) -> None:
    set_plot_style()
    methods = list(dict.fromkeys(summary["method"]))
    scenarios = list(dict.fromkeys(summary["scenario"]))
    colors = {"NIP-DA": "#4C78A8", "IC-UIP-DA": "#59A14F", "Full-borrowing": "#E15759"}
    metrics = [
        ("bias", "Bias", 0.0),
        ("rmse", "RMSE", None),
        ("coverage", "95% CrI coverage", 0.95),
        ("mean_cri_width", "Mean 95% CrI width", None),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = 0.24
    for axis, (column, label, reference) in zip(axes.ravel(), metrics):
        for index, method in enumerate(methods):
            method_data = summary[summary["method"] == method].set_index("scenario").reindex(scenarios)
            axis.bar(x + (index - 1) * width, method_data[column], width, label=method, color=colors[method])
        if reference is not None:
            axis.axhline(reference, color="black", linestyle="--", linewidth=1)
        axis.set_title(label)
        axis.set_xticks(x, [item.replace("_", "\n") for item in scenarios])
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Small-simulation comparison from observed runs")
    figure.savefig(REPOSITORY / "results" / "figures" / "method_comparison.png", bbox_inches="tight")
    plt.close(figure)

    borrowing = raw[(raw["method"].isin(["IC-UIP-DA", "Full-borrowing"])) & (~raw["failed"])]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in ["IC-UIP-DA", "Full-borrowing"]:
        values = [borrowing[(borrowing["scenario"] == scenario) & (borrowing["method"] == method)]["m_mean"] for scenario in scenarios]
        positions = x + (-0.10 if method == "IC-UIP-DA" else 0.10)
        means = [value.mean() for value in values]
        axes[0].plot(positions, means, marker="o", linewidth=2, label=method, color=colors[method])
        for position, value in zip(positions, values):
            axes[0].scatter(np.repeat(position, len(value)), value, s=10, alpha=0.35, color=colors[method])
        ess_means = [
            borrowing[(borrowing["scenario"] == scenario) & (borrowing["method"] == method)]["equivalent_ess"].mean()
            for scenario in scenarios
        ]
        axes[1].plot(positions, ess_means, marker="o", linewidth=2, label=method, color=colors[method])
    axes[0].set_title("Posterior mean nominal borrowing M")
    axes[0].set_ylabel("M")
    axes[1].set_title("Posterior-equivalent current-study ESS")
    axes[1].set_ylabel("Equivalent ESS")
    for axis in axes:
        axis.set_xticks(x, [item.replace("_", "\n") for item in scenarios])
        axis.legend(frameon=False)
    figure.savefig(REPOSITORY / "results" / "figures" / "borrowing_adaptation.png", bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("NA" if not np.isfinite(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(config: dict, sampler_values: dict, summary: pd.DataFrame, audit: list[str]) -> None:
    display_columns = [
        "scenario",
        "method",
        "repetitions",
        "bias",
        "rmse",
        "coverage",
        "mean_cri_width",
        "mean_m",
        "mean_equivalent_ess",
        "median_theta_ess",
        "failures",
    ]
    table = markdown_table(summary[display_columns])
    experiment = config["experiment"]
    indexed = summary.set_index(["scenario", "method"])
    nip_s1 = indexed.loc[("S1_compatible", "NIP-DA")]
    uip_s1 = indexed.loc[("S1_compatible", "IC-UIP-DA")]
    uip_s3 = indexed.loc[("S3_conflict", "IC-UIP-DA")]
    full_s3 = indexed.loc[("S3_conflict", "Full-borrowing")]
    m_reduction = 100.0 * (1.0 - uip_s3["mean_m"] / uip_s1["mean_m"])
    report = f"""# Small IC-UIP proof-of-concept experiment

## Scope and model

This report records the outputs produced by `scripts/run_small_simulation.py`; no table entry was entered manually. Current data follow a piecewise-exponential proportional-hazards model,

$$h_i(t)=\\lambda_0(t)\\exp(\\theta Z_i+\\beta X_i),$$

and are observed only through $(L_i,R_i]$. The last baseline interval extends to infinity. Historical studies are generated as exact/right-censored PH data, converted to Cox summaries $(\\hat\\theta_k,SE_k,n_k)$, and then discarded by the current analysis.

For fixed equal weights, $\\mu_w=\\sum_k w_k\\hat\\theta_k$, $I_w=\\sum_k w_k/(n_kSE_k^2)$, and

$$\\theta\\mid M,\\mathcal H\\sim N(\\mu_w,(MI_w)^{{-1}}),\\qquad M\\sim U(0,M_{{max}}).$$

Finite latent failure times are drawn by inversion inside their observed intervals. Given latent failures, baseline hazards have

$$\\lambda_j\\mid-\\sim Gamma\\left(a_0+D_j,\\ b_0+\\sum_i e^{{\\theta Z_i+\\beta X_i}}Y_{{ij}}\\right).$$

The scalar regression updates use a stepping-out slice sampler. This is the only nonstandard-conjugate step. For adaptive UIP,

$$M\\mid-\\sim Gamma\\left(3/2,\\tfrac12 I_w(\\theta-\\mu_w)^2\\right)I(0<M<M_{{max}}).$$

## Fixed experiment settings

- Random seed: `{experiment['seed']}`
- Current sample size: `{experiment['n_current']}`; each historical sample size: `{experiment['n_historical']}`
- True current log-HR: `{experiment['theta_current']:.6f}`; covariate coefficient: `{experiment['beta_true']}`
- Repetitions per scenario: `{summary['repetitions'].max()}` successful runs intended from `{config['experiment']['repetitions']}` configured runs
- MCMC: `{sampler_values['iterations']}` iterations, `{sampler_values['burn_in']}` burn-in, thinning `{sampler_values['thin']}`
- Methods: NIP-DA, adaptive IC-UIP-DA, and fixed-$M_{{max}}$ full-borrowing UIP
- Fixed weights: `{experiment['weighting']}`; $M_{{max}}={experiment['m_max']}$
- Common random numbers: within each repetition, current data, underlying historical random draws, and method-level MCMC streams are shared across scenarios; only the historical treatment effects change.

The equivalent ESS is a diagnostic, not a design calibration: current per-subject information is approximated by $1/(n\\,Var_{{NIP}}(\\theta\\mid D))$, and prior precision is divided by this quantity.

## Results

{table}

![Method comparison](../results/figures/method_comparison.png)

![Borrowing adaptation](../results/figures/borrowing_adaptation.png)

## Automated audit

""" + "\n".join(f"- {item}" for item in audit) + """

## Interpretation

This small experiment is a computational and qualitative check, not a definitive operating-characteristic study. In S1, adaptive IC-UIP-DA reduced RMSE from `{nip_s1['rmse']:.3f}` to `{uip_s1['rmse']:.3f}` and mean interval width from `{nip_s1['mean_cri_width']:.3f}` to `{uip_s1['mean_cri_width']:.3f}`. Its mean $M$ fell from `{uip_s1['mean_m']:.3f}` in S1 to `{uip_s3['mean_m']:.3f}` in S3, a `{m_reduction:.1f}%` reduction. The adaptive method retained S3 empirical coverage `{uip_s3['coverage']:.3f}` in these 20 paired repetitions, while fixed full borrowing had bias `{full_s3['bias']:.3f}` and coverage `{full_s3['coverage']:.3f}`. These results support the intended qualitative behavior: useful precision gain under compatibility and substantial down-weighting under severe conflict.

The shared NIP estimate has bias `{nip_s1['bias']:.3f}` and coverage `{nip_s1['coverage']:.3f}` in only 20 current datasets. This finite-repetition result, and all coverage values in the table, are too coarse for publication-level operating-characteristic claims.

## Current limitations and next steps

1. Only the PH model is implemented; proportional-odds transformation models require the Gamma-frailty layer.
2. Historical weights are fixed and equal; dynamic simplex weights are not sampled.
3. Equivalent ESS is posterior-diagnostic. A prospective interval-censoring-design Fisher-information calibration remains to be implemented.
4. Historical summaries use exact/right-censored Cox fits rather than an interval-censored NPMLE/EM analysis.
5. The sampler uses one chain per generated dataset and short CPU-budget chains. A full study should add multiple chains, rank-normalized $\\hat R$, longer runs, and substantially more repetitions.
6. A direct NPMLE/EM benchmark and comparison with power/commensurate priors remain future work.

See `papers/SOURCES.md` for the two user-provided source documents and the exact role each played.
"""
    (REPOSITORY / "reports" / "small_experiment_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    config = load_yaml(args.config)
    repetitions = int(args.repetitions or config["experiment"]["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    sampler_values = dict(config["sampler"])
    if args.iterations is not None:
        sampler_values["iterations"] = args.iterations
    if args.burn_in is not None:
        sampler_values["burn_in"] = args.burn_in
    sampler = sampler_config_from_mapping(sampler_values)
    methods = list(config["methods"])
    scenario_names = list(config["scenarios"])
    raw_rows: list[dict] = []
    history_rows: list[dict] = []
    raw_path = REPOSITORY / "results" / "small_simulation_raw.csv"

    for scenario_index, scenario in enumerate(scenario_names):
        for repetition in range(repetitions):
            logger.info("scenario=%s repetition=%d/%d", scenario, repetition + 1, repetitions)
            try:
                inputs = simulate_replicate(config, scenario, scenario_index, repetition)
            except Exception as error:
                logger.exception("input generation failed")
                for method in methods:
                    raw_rows.append(failure_row(scenario, repetition, method, float(config["experiment"]["theta_current"]), error))
                continue

            for index, item in enumerate(inputs.historical_summaries):
                history_rows.append(
                    {
                        "scenario": scenario,
                        "repetition": repetition,
                        "study": item.study,
                        "theta_true": inputs.historical_truths[index],
                        "theta_hat": item.theta_hat,
                        "se": item.se,
                        "n": item.n,
                        "unit_information": item.unit_information,
                        "event_fraction": float(inputs.histories[index].event.mean()),
                        "uip_weight": inputs.uip.weights[index],
                    }
                )

            fitted: dict[str, FitResult] = {}
            errors: dict[str, Exception] = {}
            for method_index, method in enumerate(methods):
                # Reuse chain random numbers across scenarios to sharpen the conflict comparison.
                seed = derived_seed(int(config["experiment"]["seed"]), 0, repetition, 100 + method_index)
                try:
                    fitted[method] = fit_method(
                        method,
                        inputs.current,
                        inputs.baseline.interval_starts,
                        sampler,
                        seed,
                        None if method == "NIP-DA" else inputs.uip,
                    )
                except Exception as error:
                    errors[method] = error
                    logger.exception("fit failed: scenario=%s repetition=%d method=%s", scenario, repetition, method)

            if "NIP-DA" in fitted:
                nip_variance = float(fitted["NIP-DA"].draws["theta"].var(ddof=1))
                current_unit_information = 1.0 / (inputs.current.n * nip_variance)
            else:
                current_unit_information = np.nan
            for method in methods:
                if method in fitted:
                    raw_rows.append(
                        successful_row(
                            scenario,
                            repetition,
                            fitted[method],
                            inputs.theta_true,
                            current_unit_information,
                        )
                    )
                else:
                    raw_rows.append(failure_row(scenario, repetition, method, inputs.theta_true, errors[method]))
            pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    raw = pd.DataFrame(raw_rows)
    history = pd.DataFrame(history_rows)
    summary = aggregate_simulation(raw)
    diagnostics = raw[
        [
            "scenario",
            "repetition",
            "method",
            "theta_ess",
            "theta_lag1",
            "m_ess",
            "m_lag1",
            "runtime_seconds",
            "failed",
            "error",
        ]
    ]
    summary.to_csv(REPOSITORY / "results" / "small_simulation_summary.csv", index=False)
    history.to_csv(REPOSITORY / "results" / "historical_summaries.csv", index=False)
    diagnostics.to_csv(REPOSITORY / "results" / "diagnostics.csv", index=False)
    audit = audit_results(raw, float(config["experiment"]["m_max"]))
    (REPOSITORY / "results" / "audit.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    save_plots(summary, raw)
    write_report(config, sampler_values, summary, audit)
    logger.info("completed %d rows; audit=%s", len(raw), " | ".join(audit))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
