"""MCMC and repeated-simulation evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def autocorrelation(draws: ArrayLike, lag: int = 1) -> float:
    values = np.asarray(draws, dtype=float)
    if lag <= 0 or lag >= values.size:
        raise ValueError("lag must lie between 1 and n_draws - 1")
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        return 0.0
    return float(centered[:-lag] @ centered[lag:] / denominator)


def effective_sample_size(draws: ArrayLike) -> float:
    """Estimate scalar ESS using Geyer's initial-positive-pair rule."""
    values = np.asarray(draws, dtype=float)
    n = values.size
    if n < 4:
        return float(n)
    centered = values - values.mean()
    variance = float(centered @ centered)
    if variance <= 0:
        return float(n)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:n]
    autocorr = acov / acov[0]
    positive_sum = 0.0
    for lag in range(1, n - 1, 2):
        pair = float(autocorr[lag] + autocorr[lag + 1])
        if pair <= 0:
            break
        positive_sum += pair
    tau = max(1.0, 1.0 + 2.0 * positive_sum)
    return float(min(n, n / tau))


def summarize_chain(draws: pd.DataFrame) -> dict[str, float]:
    theta = draws["theta"].to_numpy()
    result = {
        "theta_mean": float(theta.mean()),
        "theta_sd": float(theta.std(ddof=1)),
        "theta_lower": float(np.quantile(theta, 0.025)),
        "theta_upper": float(np.quantile(theta, 0.975)),
        "theta_ess": effective_sample_size(theta),
        "theta_lag1": autocorrelation(theta, lag=1),
    }
    if "m" in draws:
        m = draws["m"].to_numpy()
        result.update(
            m_mean=float(m.mean()),
            m_sd=float(m.std(ddof=1)),
            m_ess=effective_sample_size(m),
            m_lag1=autocorrelation(m, lag=1),
        )
    if "commensurate_precision" in draws:
        precision = draws["commensurate_precision"].to_numpy()
        result.update(
            commensurate_precision_mean=float(precision.mean()),
            commensurate_precision_sd=float(precision.std(ddof=1)),
            commensurate_precision_ess=effective_sample_size(precision),
            commensurate_precision_lag1=autocorrelation(precision, lag=1),
            historical_mean_mean=float(draws["historical_mean"].mean()),
        )
    for column in [name for name in draws.columns if name.startswith("weight_")]:
        values = draws[column].to_numpy()
        result[f"{column}_mean"] = float(values.mean())
        result[f"{column}_ess"] = effective_sample_size(values)
    if "uip_mean" in draws:
        result["uip_mean_mean"] = float(draws["uip_mean"].mean())
        result["uip_unit_information_mean"] = float(draws["uip_unit_information"].mean())
    return result


def aggregate_simulation(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scenario",
        "method",
        "theta_mean",
        "bias",
        "squared_error",
        "covered",
        "cri_width",
        "m_mean",
        "equivalent_ess",
        "theta_ess",
        "theta_lag1",
        "runtime_seconds",
        "failed",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw simulation results miss columns: {sorted(missing)}")

    clean = raw.loc[~raw["failed"].astype(bool)].copy()
    grouped = clean.groupby(["scenario", "method"], sort=False)
    summary = grouped.agg(
        repetitions=("theta_mean", "size"),
        mean_estimate=("theta_mean", "mean"),
        bias=("bias", "mean"),
        rmse=("squared_error", lambda value: float(np.sqrt(np.mean(value)))),
        coverage=("covered", "mean"),
        mean_cri_width=("cri_width", "mean"),
        mean_m=("m_mean", "mean"),
        mean_equivalent_ess=("equivalent_ess", "mean"),
        median_theta_ess=("theta_ess", "median"),
        mean_theta_lag1=("theta_lag1", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    ).reset_index()
    for column in [name for name in clean.columns if name.startswith("weight_") and name.endswith("_mean")]:
        weight_summary = grouped[column].mean().rename(f"mean_{column}").reset_index()
        summary = summary.merge(weight_summary, on=["scenario", "method"], how="left")
    if "commensurate_precision_mean" in clean:
        cp_summary = (
            grouped["commensurate_precision_mean"]
            .mean()
            .rename("mean_commensurate_precision")
            .reset_index()
        )
        summary = summary.merge(cp_summary, on=["scenario", "method"], how="left")
    failures = raw.groupby(["scenario", "method"], sort=False)["failed"].sum().rename("failures").reset_index()
    return summary.merge(failures, on=["scenario", "method"], how="left")


def audit_results(raw: pd.DataFrame, expected_m_max: float) -> list[str]:
    findings: list[str] = []
    numeric = raw.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        findings.append("FAIL: infinite numeric value detected")
    essential = ["theta_mean", "theta_lower", "theta_upper", "theta_ess", "runtime_seconds"]
    if raw.loc[~raw["failed"].astype(bool), essential].isna().any().any():
        findings.append("FAIL: NaN in an essential successful-fit field")
    adaptive = raw[raw["method"] == "IC-UIP"]
    if not adaptive.empty and ((adaptive["m_mean"] <= 0) | (adaptive["m_mean"] >= expected_m_max)).any():
        findings.append("FAIL: adaptive posterior mean M is outside its open support")
    weight_columns = [column for column in adaptive.columns if column.startswith("weight_") and column.endswith("_mean")]
    if weight_columns:
        weight_values = adaptive[weight_columns]
        if ((weight_values <= 0) | (weight_values >= 1)).any().any():
            findings.append("FAIL: posterior mean UIP weight is outside its open simplex support")
        if not np.allclose(weight_values.sum(axis=1), 1.0, atol=1e-8):
            findings.append("FAIL: posterior mean UIP weights do not sum to one")
    if (raw["theta_ess"].dropna() < 10).any():
        findings.append("WARN: at least one theta chain has ESS below 10")
    if (raw["theta_lag1"].dropna().abs() > 0.98).any():
        findings.append("WARN: at least one theta chain has |lag-1 autocorrelation| above 0.98")
    if int(raw["failed"].sum()) > 0:
        findings.append(f"WARN: {int(raw['failed'].sum())} method fits failed; inspect the log")
    coverage = raw.loc[~raw["failed"].astype(bool)].groupby(["scenario", "method"])["covered"].mean()
    low_coverage = coverage[coverage < 0.80]
    for (scenario, method), value in low_coverage.items():
        findings.append(f"WARN: low empirical coverage ({value:.3f}) for {method} in {scenario}")
    adaptive = raw[(raw["method"] == "IC-UIP") & (~raw["failed"].astype(bool))]
    if {"S1_compatible", "S3_conflict"}.issubset(set(adaptive["scenario"])):
        mean_m = adaptive.groupby("scenario")["m_mean"].mean()
        if mean_m["S3_conflict"] >= mean_m["S1_compatible"]:
            findings.append("WARN: adaptive mean M did not decrease from compatible to severe-conflict scenarios")
    if {"weight_1_mean", "weight_2_mean"}.issubset(adaptive.columns):
        mixed = adaptive[adaptive["scenario"] == "S2_mixed"]
        if not mixed.empty and mixed["weight_1_mean"].mean() <= mixed["weight_2_mean"].mean():
            findings.append("WARN: compatible historical study did not receive the larger mean weight in S2")
    if not findings:
        findings.append("PASS: no NaN/Inf, boundary, low-ESS, stuck-chain, or fit-failure flags")
    return findings
