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
    m_mean = summary.get("m_mean", np.nan)
    prior_precision = 0.0
    if {"m", "uip_unit_information"}.issubset(result.draws.columns):
        prior_precision = float(
            (result.draws["m"] * result.draws["uip_unit_information"]).mean()
        )
    elif "commensurate_precision" in result.draws:
        prior_precision = float(result.draws["commensurate_precision"].mean())
    equivalent_ess = prior_precision / current_unit_information if current_unit_information > 0 else np.nan
    row = {
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
        "m_sd": summary.get("m_sd", np.nan),
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
        "commensurate_precision_mean": summary.get("commensurate_precision_mean", np.nan),
        "commensurate_precision_ess": summary.get("commensurate_precision_ess", np.nan),
        "failed": False,
        "error": "",
    }
    for name, value in summary.items():
        if name.startswith("weight_"):
            row[name] = value
    return row


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
        "commensurate_precision_mean": np.nan,
        "commensurate_precision_ess": np.nan,
        "weight_1_mean": np.nan,
        "weight_1_ess": np.nan,
        "weight_2_mean": np.nan,
        "weight_2_ess": np.nan,
        "failed": True,
        "error": repr(error),
    }


def save_plots(summary: pd.DataFrame, raw: pd.DataFrame) -> None:
    set_plot_style()
    methods = list(dict.fromkeys(summary["method"]))
    scenarios = list(dict.fromkeys(summary["scenario"]))
    colors = {
        "IC-NIP": "#4C78A8",
        "IC-UIP": "#59A14F",
        "IC-CP": "#B279A2",
    }
    metrics = [
        ("bias", "Bias", 0.0),
        ("rmse", "RMSE", None),
        ("coverage", "95% CrI coverage", 0.95),
        ("mean_cri_width", "Mean 95% CrI width", None),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = 0.19
    for axis, (column, label, reference) in zip(axes.ravel(), metrics):
        for index, method in enumerate(methods):
            method_data = summary[summary["method"] == method].set_index("scenario").reindex(scenarios)
            offset = (index - (len(methods) - 1) / 2.0) * width
            axis.bar(x + offset, method_data[column], width, label=method, color=colors[method])
        if reference is not None:
            axis.axhline(reference, color="black", linestyle="--", linewidth=1)
        axis.set_title(label)
        axis.set_xticks(x, [item.replace("_", "\n") for item in scenarios])
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Small-simulation comparison from observed runs")
    figure.savefig(REPOSITORY / "results" / "figures" / "method_comparison.png", bbox_inches="tight")
    plt.close(figure)

    borrowing = raw[(raw["method"].isin(["IC-UIP", "IC-CP"])) & (~raw["failed"])]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    values = [borrowing[(borrowing["scenario"] == scenario) & (borrowing["method"] == "IC-UIP")]["m_mean"] for scenario in scenarios]
    means = [value.mean() for value in values]
    axes[0].plot(x, means, marker="o", linewidth=2, label="IC-UIP", color=colors["IC-UIP"])
    for position, value in zip(x, values):
        axes[0].scatter(np.repeat(position, len(value)), value, s=10, alpha=0.35, color=colors["IC-UIP"])
    for method_index, method in enumerate(["IC-UIP", "IC-CP"]):
        positions = x + (method_index - 0.5) * 0.08
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

    adaptive = raw[(raw["method"] == "IC-UIP") & (~raw["failed"])]
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for study_index, color in [(1, "#59A14F"), (2, "#F28E2B")]:
        column = f"weight_{study_index}_mean"
        means = [adaptive[adaptive["scenario"] == scenario][column].mean() for scenario in scenarios]
        axis.plot(x, means, marker="o", linewidth=2, color=color, label=f"Historical study H{study_index}")
        for position, scenario in zip(x, scenarios):
            values = adaptive[adaptive["scenario"] == scenario][column]
            axis.scatter(np.repeat(position, len(values)), values, s=12, alpha=0.3, color=color)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Posterior mean weight")
    axis.set_title("Dynamic IC-UIP historical-study weights")
    axis.set_xticks(x, [item.replace("_", "\n") for item in scenarios])
    axis.legend(frameon=False)
    figure.savefig(REPOSITORY / "results" / "figures" / "dynamic_weights.png", bbox_inches="tight")
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
        "scenario", "method", "repetitions", "bias", "rmse", "coverage",
        "mean_cri_width", "mean_m", "mean_equivalent_ess", "median_theta_ess",
        "failures",
    ]
    for optional in ["mean_weight_1_mean", "mean_weight_2_mean"]:
        if optional in summary:
            display_columns.append(optional)
    display = summary[display_columns].rename(
        columns={"mean_weight_1_mean": "mean_w1", "mean_weight_2_mean": "mean_w2"}
    )
    table = markdown_table(display)
    experiment = config["experiment"]
    indexed = summary.set_index(["scenario", "method"])
    nip_s1 = indexed.loc[("S1_compatible", "IC-NIP")]
    uip_s1 = indexed.loc[("S1_compatible", "IC-UIP")]
    uip_s2 = indexed.loc[("S2_mixed", "IC-UIP")]
    uip_s3 = indexed.loc[("S3_conflict", "IC-UIP")]
    cp_s3 = indexed.loc[("S3_conflict", "IC-CP")]
    m_reduction = 100.0 * (1.0 - uip_s3["mean_m"] / uip_s1["mean_m"])
    audit_text = "\n".join(f"- {item}" for item in audit)
    report = f"""# Small IC-UIP proof-of-concept experiment

## Scope and model

This report is regenerated from `scripts/run_small_simulation.py`; no result is manually entered. Current interval-censored data follow

$$h_i(t)=\\lambda_0(t)\\exp(\\theta Z_i+\\beta X_i),$$

with a piecewise-constant baseline hazard whose last interval extends to infinity. Historical studies are converted to Cox summaries $(\\hat\\theta_k,SE_k,n_k)$; current analyses never receive historical patient-level observations.

For IC-UIP,

$$\\mu_w=\\sum_k w_k\\hat\\theta_k,\\quad I_w=\\sum_k\\frac{{w_k}}{{n_kSE_k^2}},\\quad \\theta\\mid M,w,\\mathcal H\\sim N(\\mu_w,(MI_w)^{{-1}}).$$

Weights are no longer fixed: $w\\sim Dirichlet(\\gamma)$ and their conditional density is

$$p(w\\mid-)\\propto\\prod_k w_k^{{\\gamma_k-1}}I_w^{{1/2}}\\exp\\left[-\\tfrac12MI_w(\\theta-\\mu_w)^2\\right].$$

The implementation samples additive-log-ratio coordinates one at a time by slice sampling, including the softmax Jacobian. Adaptive $M$ retains the direct truncated-Gamma update

$$M\\mid-\\sim Gamma\\left(3/2,\\tfrac12I_w(\\theta-\\mu_w)^2\\right)I(0<M<M_{{max}}).$$

`IC-CP` is a summary-level commensurate comparator:

$$\\hat\\theta_k\\mid\\theta_H\\sim N(\\theta_H,SE_k^2),\\quad \\theta\\mid\\theta_H,\\tau\\sim N(\\theta_H,\\tau^{{-1}}),\\quad \\tau\\sim Gamma(0.5,0.05).$$

It preserves the repository's summary-only constraint. It is not the matched individual-level random-effects model of Fang et al. (2025).

Finite latent failures are sampled by exact inversion within $(L_i,R_i]$. Baseline hazards have Gamma full conditionals. Regression coefficients and UIP weight coordinates use one-dimensional stepping-out slice updates; $M$, $\\theta_H$, and $\\tau$ have direct standard-distribution updates.

## Experiment settings

- Seed `{experiment['seed']}`; current $n={experiment['n_current']}$; each historical $n_k={experiment['n_historical']}$.
- True current log-HR `{experiment['theta_current']:.6f}`; covariate coefficient `{experiment['beta_true']}`.
- `{summary['repetitions'].max()}` paired repetitions per scenario.
- `{sampler_values['iterations']}` iterations, `{sampler_values['burn_in']}` burn-in, thinning `{sampler_values['thin']}`.
- Methods: `IC-NIP`, `IC-UIP`, and the literature-motivated `IC-CP` comparator.
- UIP weight prior: $Dirichlet({experiment['weight_dirichlet_concentration']})$; $M_{{max}}={experiment['m_max']}$.
- Common random numbers are shared across scenarios within each repetition; only historical effects change.

Equivalent ESS remains a diagnostic: posterior borrowing precision is divided by $1/[n\\,Var_{{IC-NIP}}(\\theta\\mid D)]$.

## Results

{table}

![Method comparison](../results/figures/method_comparison.png)

![Borrowing adaptation](../results/figures/borrowing_adaptation.png)

![Dynamic weights](../results/figures/dynamic_weights.png)

## Automated audit

{audit_text}

## Interpretation

In S1, IC-UIP changed RMSE from `{nip_s1['rmse']:.3f}` under IC-NIP to `{uip_s1['rmse']:.3f}` and mean CrI width from `{nip_s1['mean_cri_width']:.3f}` to `{uip_s1['mean_cri_width']:.3f}`. Mean $M$ decreased from `{uip_s1['mean_m']:.3f}` in S1 to `{uip_s3['mean_m']:.3f}` in S3, a `{m_reduction:.1f}%` reduction. In S2, the posterior mean weights were `w1={uip_s2.get('mean_weight_1_mean', np.nan):.3f}` and `w2={uip_s2.get('mean_weight_2_mean', np.nan):.3f}`, where H1 is the compatible study.

Under S3, IC-UIP had RMSE `{uip_s3['rmse']:.3f}` and coverage `{uip_s3['coverage']:.3f}`; IC-CP had RMSE `{cp_s3['rmse']:.3f}` and coverage `{cp_s3['coverage']:.3f}`. These `{int(summary['repetitions'].max())}`-repetition values are qualitative checks, not publication-level operating characteristics.

## Literature positioning

The closest direct work is Fang et al. (2025), which uses commensurate priors and random effects for matched interval-censored current and historical controls. Murray et al. (2014) provides the right-censored semiparametric commensurate-survival precursor. Gu and Yin (2024) develop a unit-information Dirichlet-process prior for survival distributions, but not this interval-censored regression-effect setting. The reusable search report is in `literature-search-20260727-interval-censored-borrowing/`.

## Limitations and next steps

1. Only PH is implemented; PO/general transformation models require the Gamma-frailty layer.
2. Dynamic weights learn through the treatment-effect UIP kernel; study-level covariate/design discrepancies are not separately modeled.
3. IC-CP is a summary-normal comparator, not a reproduction of Fang et al.'s matched individual-level model.
4. Historical summaries use exact/right-censored Cox fits rather than interval-censored NPMLE/EM fits.
5. Equivalent ESS is posterior-diagnostic rather than prospective IC-design calibration.
6. Short single chains and 20 repetitions are insufficient for final type-I-error or coverage claims.
7. Direct IntCens NPMLE/EM, individual-level Fang-style CP, robust MAP, and normalized power-prior comparisons remain larger follow-up tasks.

See `papers/SOURCES.md` and the literature-search folder for traceable sources and scope cautions.
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
                        "initial_uip_weight": inputs.uip.weights[index],
                        "dirichlet_concentration": inputs.uip.dirichlet_concentration[index],
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
                        inputs.uip if method == "IC-UIP" else None,
                        inputs.commensurate if method == "IC-CP" else None,
                    )
                except Exception as error:
                    errors[method] = error
                    logger.exception("fit failed: scenario=%s repetition=%d method=%s", scenario, repetition, method)

            if "IC-NIP" in fitted:
                nip_variance = float(fitted["IC-NIP"].draws["theta"].var(ddof=1))
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
            "commensurate_precision_ess",
            "weight_1_ess",
            "weight_2_ess",
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
