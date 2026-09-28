"""Unified paired three-outcome experiment for the design-stage rUIP.

The three current likelihoods are kept on their natural one-parameter scales.
For survival, exponential event times are only the data-generating mechanism.
Analysis uses the Cox partial likelihood for a randomized 1:1 treatment
indicator and never reads the baseline hazard. Historical Gaussian summaries
are Cox partial-likelihood MLEs and observed information. PP integrates
independent Beta(1,1) powers over conditionally normalized Gaussian summary
priors, then updates with the exact current likelihood. rMAP, CP, and UIP
retain their frozen Gaussian summary priors, combined with the current Cox
partial likelihood by quadrature.

Every outcome/scenario/replicate dataset is shared by every method.  No prior
builder accepts current data.  The run manifest records a direct invariance
check made by regenerating current data while holding each history fixed.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import digamma, expit, logit, logsumexp, ndtr, polygamma
from scipy.stats import beta as beta_distribution

from ruip.outcomes.survival import (
    CoxRiskSets,
    compress_risk_sets,
    log_partial_likelihood,
    partial_likelihood_summary,
)
from ruip.outcomes.survival import posterior_summary as cox_posterior_summary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "tri_outcome_design_stage"
FROZEN_LIBRARY = ROOT / "csrc" / "frozen_engine_v5" / "libruip_validation.dll"
ABI_SOURCE = ROOT / "src" / "ruip" / "c_abi.py"

SEED = 202609180076
THETA = {"continuous": 0.0, "binary": math.log(0.4 / 0.6), "survival": 0.0}
CURRENT_TARGET_INFORMATION = 100.0
HISTORICAL_TARGET_INFORMATION = np.array([120.0, 80.0, 50.0])
DELTA_CLIN = 0.25
BASELINE_HAZARD = 1.0
CENSORING_HAZARD = 3.0 / 7.0
POWER_BASE_MEAN = 0.0
POWER_BASE_SD = math.sqrt(10.0)
POWER_QUADRATURE_ORDER = 12
METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP no-local", "rUIP no-global", "rUIP")
PRIMARY_METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP")
ALL_OUTCOMES = ("continuous", "binary", "survival")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    global_conflict: str
    local_conflict: str
    delta: float
    local_deviation: tuple[float, float, float]


SCENARIOS = (
    Scenario("GW-LW-0", "weak", "weak", 0.00, (0.00, 0.00, 0.00)),
    Scenario("GW-LW-D", "weak", "weak", 0.10, (0.00, 0.00, 0.00)),
    Scenario("GS-LW-P", "strong", "weak", 0.35, (0.00, 0.00, 0.00)),
    Scenario("GS-LW-N", "strong", "weak", -0.35, (0.00, 0.00, 0.00)),
    Scenario("GW-LS-I", "weak", "strong", 0.00, (0.45, -0.225, -0.225)),
    Scenario("GS-LS-I", "strong", "strong", 0.35, (0.45, -0.225, -0.225)),
)


@dataclass(frozen=True)
class Dataset:
    outcome: str
    theta: float
    history_estimate: np.ndarray
    history_information: np.ndarray
    history_subjects: np.ndarray
    history_events: np.ndarray
    history_exposure: np.ndarray
    current_estimate: float
    current_information: float
    current_subjects: int
    current_events: int
    current_exposure: float
    current_censored: int
    survival_history: tuple[CoxRiskSets, ...] | None = None
    survival_current: CoxRiskSets | None = None


@dataclass(frozen=True)
class Prior:
    means: np.ndarray
    variances: np.ndarray
    log_weights: np.ndarray
    local_weights: np.ndarray | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_engine():
    """Load the C comparator kernels using the checked Python ABI layout."""
    spec = importlib.util.spec_from_file_location("tri_outcome_abi", ABI_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen ABI")
    abi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abi)
    dll = ct.CDLL(str(FROZEN_LIBRARY))
    for name in (
        "ruip_rmap_mixture",
        "ruip_commensurate_mixture",
    ):
        fn = getattr(dll, name)
        fn.argtypes = [ct.POINTER(abi.History), ct.c_size_t, ct.POINTER(abi.Mixture)]
        fn.restype = ct.c_int
    dll.ruip_standard_uip_mixture.argtypes = [
        ct.POINTER(abi.History),
        ct.c_size_t,
        ct.c_double,
        ct.POINTER(abi.Mixture),
    ]
    dll.ruip_standard_uip_mixture.restype = ct.c_int
    dll.ruip_normal_mixture_free.argtypes = [ct.POINTER(abi.Mixture)]
    dll.ruip_normal_mixture_free.restype = None
    return abi, dll


def history_array(dataset: Dataset, abi):
    return (abi.History * 3)(
        *[
            abi.History(
                int(dataset.history_subjects[k]),
                float(dataset.history_estimate[k]),
                float(dataset.history_information[k] / dataset.history_subjects[k]),
            )
            for k in range(3)
        ]
    )


def frozen_prior(dataset: Dataset, method: str, abi, dll) -> Prior:
    histories = history_array(dataset, abi)
    mixture = abi.Mixture()
    if method == "rMAP":
        rc = dll.ruip_rmap_mixture(histories, 3, ct.byref(mixture))
    elif method == "CP":
        rc = dll.ruip_commensurate_mixture(histories, 3, ct.byref(mixture))
    elif method == "UIP":
        # The UIP upper bound is planned current subject count, not information 100.
        rc = dll.ruip_standard_uip_mixture(
            histories, 3, float(dataset.current_subjects), ct.byref(mixture)
        )
    else:
        raise ValueError(method)
    if rc != 0:
        raise RuntimeError(f"frozen prior failed for {method}: rc={rc}")
    try:
        means = np.fromiter(
            (mixture.items[i].mean for i in range(mixture.count)), float
        )
        variances = np.fromiter(
            (mixture.items[i].variance for i in range(mixture.count)), float
        )
        log_weights = np.fromiter(
            (mixture.items[i].log_weight for i in range(mixture.count)), float
        )
    finally:
        dll.ruip_normal_mixture_free(ct.byref(mixture))
    log_weights -= logsumexp(log_weights)
    return Prior(means, variances, log_weights)


def normalized_power_prior(
    dataset: Dataset, order: int = POWER_QUADRATURE_ORDER
) -> Prior:
    """Mix normalized Gaussian historical priors over independent Beta(1,1) powers."""
    if order < 2:
        raise ValueError("power quadrature order must exceed one")
    nodes, weights = leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    powers = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    joint_weights = np.prod(
        np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1),
        axis=-1,
    ).ravel()
    precision = POWER_BASE_SD**-2 + powers @ dataset.history_information
    mean = (
        POWER_BASE_MEAN * POWER_BASE_SD**-2
        + powers @ (dataset.history_information * dataset.history_estimate)
    ) / precision
    return Prior(mean, 1.0 / precision, np.log(joint_weights))


def ruip_prior(dataset: Dataset, method: str) -> Prior:
    h = dataset.history_estimate
    info = dataset.history_information
    if method == "rUIP no-local":
        weights = info / info.sum()
        retained_information = float(info.sum())
    else:
        objectives = np.abs(h[:, None] - h[None, :]).sum(axis=1)
        anchor = min(range(h.size), key=lambda k: (objectives[k], -info[k], k))
        standard_variance = 1.0 / info + 1.0 / info[anchor]
        compatibility = np.exp(-0.5 * (h - h[anchor]) ** 2 / standard_variance)
        compatibility[anchor] = 1.0
        retained = np.minimum(info, info[anchor]) * compatibility
        retained[anchor] = info[anchor]
        retained_information = float(retained.sum())
        weights = retained / retained_information
    center = float(np.dot(weights, h))
    precision = (
        retained_information
        if method == "rUIP no-global"
        else retained_information / (1.0 + DELTA_CLIN**2 * retained_information)
    )
    return Prior(
        np.array([center]),
        np.array([1.0 / precision]),
        np.array([0.0]),
        weights,
    )


def make_dataset(outcome: str, scenario: Scenario, rng: np.random.Generator) -> Dataset:
    """Draw one shared current/history dataset on the outcome's analysis scale."""
    theta = THETA[outcome]
    historical_theta = theta + scenario.delta + np.asarray(scenario.local_deviation)
    if outcome == "continuous":
        h = rng.normal(historical_theta, 1.0 / np.sqrt(HISTORICAL_TARGET_INFORMATION))
        current = float(rng.normal(theta, 1.0 / math.sqrt(CURRENT_TARGET_INFORMATION)))
        return Dataset(
            outcome,
            theta,
            h,
            HISTORICAL_TARGET_INFORMATION.copy(),
            HISTORICAL_TARGET_INFORMATION.astype(int),
            np.zeros(3, int),
            np.zeros(3),
            current,
            CURRENT_TARGET_INFORMATION,
            100,
            0,
            0.0,
            0,
        )
    if outcome == "binary":
        # Planned n is frozen at p=.4 for every scenario.
        current_n = int(round(CURRENT_TARGET_INFORMATION / 0.24))
        history_n = np.rint(HISTORICAL_TARGET_INFORMATION / 0.24).astype(int)
        hp = expit(historical_theta)
        hy = rng.binomial(history_n, hp)
        cy = int(rng.binomial(current_n, 0.4))
        h = np.log((hy + 0.5) / (history_n - hy + 0.5))
        hi = (
            history_n
            * ((hy + 0.5) / (history_n + 1.0))
            * (1.0 - (hy + 0.5) / (history_n + 1.0))
        )
        ce = math.log((cy + 0.5) / (current_n - cy + 0.5))
        ci = (
            current_n
            * ((cy + 0.5) / (current_n + 1.0))
            * (1.0 - (cy + 0.5) / (current_n + 1.0))
        )
        return Dataset(
            outcome,
            theta,
            h,
            hi,
            history_n,
            hy,
            np.zeros(3),
            ce,
            ci,
            current_n,
            cy,
            0.0,
            0,
        )
    if outcome == "survival":
        # At the null, each event contributes about Var(Z)=1/4 information.
        # Counts are even so every trial has exactly balanced randomized arms.
        history_n = 2 * np.ceil(
            HISTORICAL_TARGET_INFORMATION / (0.7 * 0.25) / 2
        ).astype(int)
        hn = []
        he = []
        ht = []
        risk_sets = []
        for n, eta in zip(history_n, historical_theta, strict=True):
            treatment = np.repeat([0, 1], int(n) // 2)
            rng.shuffle(treatment)
            event = rng.exponential(
                1.0 / (BASELINE_HAZARD * np.exp(float(eta) * treatment))
            )
            censor = rng.exponential(1.0 / CENSORING_HAZARD, int(n))
            observed = np.minimum(event, censor)
            status = event <= censor
            d = int(np.sum(status))
            exposure = float(observed.sum())
            compressed = compress_risk_sets(observed, status, treatment)
            estimate, information = partial_likelihood_summary(compressed)
            hn.append(estimate)
            he.append(d)
            ht.append(exposure)
            risk_sets.append(compressed)
        current_n = int(2 * math.ceil(CURRENT_TARGET_INFORMATION / (0.7 * 0.25) / 2))
        treatment = np.repeat([0, 1], current_n // 2)
        rng.shuffle(treatment)
        event = rng.exponential(1.0 / (BASELINE_HAZARD * np.exp(theta * treatment)))
        censor = rng.exponential(1.0 / CENSORING_HAZARD, current_n)
        observed = np.minimum(event, censor)
        status = event <= censor
        d = int(np.sum(status))
        exposure = float(observed.sum())
        current_risk_sets = compress_risk_sets(observed, status, treatment)
        current_estimate, current_information = partial_likelihood_summary(
            current_risk_sets
        )
        return Dataset(
            outcome,
            theta,
            np.asarray(hn),
            np.asarray([partial_likelihood_summary(x)[1] for x in risk_sets]),
            history_n,
            np.asarray(he),
            np.asarray(ht),
            current_estimate,
            current_information,
            current_n,
            d,
            exposure,
            current_n - d,
            tuple(risk_sets),
            current_risk_sets,
        )
    raise ValueError(outcome)


def exact_nip_or_pp(
    dataset: Dataset, method: str
) -> tuple[float, float, float, float, float]:
    if method == "PP":
        return analyze(dataset, "PP", None, None)[:5]
    if method != "NIP":
        raise ValueError(method)
    if dataset.outcome == "continuous":
        mean, variance = dataset.current_estimate, 1.0 / dataset.current_information
        sd = math.sqrt(variance)
        return (
            mean,
            variance,
            mean - 1.959963984540054 * sd,
            mean + 1.959963984540054 * sd,
            0.0,
        )
    if dataset.outcome == "binary":
        alpha = float(dataset.current_events)
        beta = float(dataset.current_subjects - dataset.current_events)
        mean = float(digamma(alpha) - digamma(beta))
        variance = float(polygamma(1, alpha) + polygamma(1, beta))
        lower, upper = logit(beta_distribution.ppf([0.025, 0.975], alpha, beta))
        return mean, variance, float(lower), float(upper), 0.0
    if dataset.survival_current is None or dataset.survival_history is None:
        raise ValueError("survival Cox risk sets are missing")
    return cox_posterior_summary(
        dataset.survival_current,
    )


def log_likelihood(dataset: Dataset, theta: np.ndarray) -> np.ndarray:
    if dataset.outcome == "binary":
        return dataset.current_events * theta - dataset.current_subjects * np.logaddexp(
            0.0, theta
        )
    if dataset.outcome == "survival":
        if dataset.survival_current is None:
            raise ValueError("survival Cox risk sets are missing")
        return log_partial_likelihood(theta, dataset.survival_current)
    return -0.5 * dataset.current_information * (theta - dataset.current_estimate) ** 2


def mixture_log_density(x: np.ndarray, prior: Prior, chunk: int = 256) -> np.ndarray:
    answer = np.full(x.shape, -np.inf)
    for start in range(0, prior.means.size, chunk):
        stop = min(start + chunk, prior.means.size)
        means = prior.means[start:stop, None]
        variances = prior.variances[start:stop, None]
        values = (
            prior.log_weights[start:stop, None]
            - 0.5 * np.log(2.0 * np.pi * variances)
            - 0.5 * (x[None, :] - means) ** 2 / variances
        )
        answer = np.logaddexp(answer, logsumexp(values, axis=0))
    return answer


def numerical_posterior(
    dataset: Dataset, prior: Prior
) -> tuple[float, float, float, float, float]:
    # The current likelihood is log-concave and carries roughly 100 information
    # units. Fifteen current standard errors safely cover its tails, including
    # posterior displacement induced by the bounded borrowing rules.
    current_scale = 1.0 / math.sqrt(max(dataset.current_information, 1e-12))
    lower_bound = max(-20.0, dataset.current_estimate - 15.0 * current_scale)
    upper_bound = min(20.0, dataset.current_estimate + 15.0 * current_scale)
    grid = np.linspace(lower_bound, upper_bound, 401)
    log_kernel = mixture_log_density(grid, prior) + log_likelihood(dataset, grid)
    log_kernel -= float(np.max(log_kernel))

    def integrate(selected_grid: np.ndarray, selected_log_kernel: np.ndarray):
        density = np.exp(selected_log_kernel)
        increments = 0.5 * (density[1:] + density[:-1]) * np.diff(selected_grid)
        mass = float(increments.sum())
        cdf = np.r_[0.0, np.cumsum(increments)] / mass
        weights = density / np.trapezoid(density, selected_grid)
        mean = float(np.trapezoid(selected_grid * weights, selected_grid))
        second = float(np.trapezoid(selected_grid**2 * weights, selected_grid))
        q = np.interp([0.025, 0.975], cdf, selected_grid)
        return mean, max(second - mean * mean, 0.0), float(q[0]), float(q[1]), mass

    fine = integrate(grid, log_kernel)
    coarse = integrate(grid[::2], log_kernel[::2])
    # Prespecified mass error compares 201- and 401-node trapezoid evidence
    # after a shared log-kernel scaling. Quantile interpolation is intentionally
    # excluded from the mass-conservation gate.
    error = abs(coarse[4] / fine[4] - 1.0)
    return fine[0], fine[1], fine[2], fine[3], float(error)


def normal_mixture_quantiles(prior: Prior) -> tuple[float, float]:
    weights = np.exp(prior.log_weights - logsumexp(prior.log_weights))
    sd = np.sqrt(prior.variances)

    def cdf(value: float) -> float:
        return float(np.dot(weights, ndtr((value - prior.means) / sd)))

    lower_bound = float(np.min(prior.means - 12.0 * sd))
    upper_bound = float(np.max(prior.means + 12.0 * sd))
    return (
        float(brentq(lambda value: cdf(value) - 0.025, lower_bound, upper_bound)),
        float(brentq(lambda value: cdf(value) - 0.975, lower_bound, upper_bound)),
    )


def analyze(dataset: Dataset, method: str, abi, dll):
    if method == "NIP":
        return (*exact_nip_or_pp(dataset, method), None)
    if method == "PP":
        prior = normalized_power_prior(dataset)
    elif method.startswith("rUIP"):
        prior = ruip_prior(dataset, method)
    else:
        prior = frozen_prior(dataset, method, abi, dll)
    if dataset.outcome == "continuous":
        current_variance = 1.0 / dataset.current_information
        post_variance = 1.0 / (1.0 / prior.variances + dataset.current_information)
        post_mean = post_variance * (
            prior.means / prior.variances
            + dataset.current_information * dataset.current_estimate
        )
        log_post_weight = prior.log_weights - 0.5 * (
            np.log(2.0 * np.pi * (prior.variances + current_variance))
            + (dataset.current_estimate - prior.means) ** 2
            / (prior.variances + current_variance)
        )
        weight = np.exp(log_post_weight - logsumexp(log_post_weight))
        mean = float(np.dot(weight, post_mean))
        variance = float(np.dot(weight, post_variance + (post_mean - mean) ** 2))
        synthetic = Prior(post_mean, post_variance, np.log(weight))
        lower, upper = normal_mixture_quantiles(synthetic)
        return mean, variance, lower, upper, 0.0, prior
    return (*numerical_posterior(dataset, prior), prior)


def prior_fingerprint(prior: Prior) -> str:
    digest = hashlib.sha256()
    for value in (
        prior.means,
        prior.variances,
        prior.log_weights,
        np.array([]) if prior.local_weights is None else prior.local_weights,
    ):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def exact_power_prior_fingerprint(dataset: Dataset) -> str:
    """Historical-only fingerprint retained for confirmation runner compatibility."""
    return prior_fingerprint(normalized_power_prior(dataset))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["outcome", "scenario_id", "global_conflict", "local_conflict", "method"]
    for key, frame in raw.groupby(keys, sort=False, dropna=False):
        success = frame[frame.status == "success"]
        n = len(success)
        errors = success.error.to_numpy(float)
        sq = errors**2
        covered = success.covered.to_numpy(float)
        width = success.interval_width.to_numpy(float)
        mse = float(sq.mean()) if n else math.nan
        rmse = math.sqrt(mse) if n else math.nan
        rmse_mcse = (
            float(sq.std(ddof=1) / math.sqrt(n) / (2.0 * rmse))
            if n > 1 and rmse > 0
            else math.nan
        )
        coverage = float(covered.mean()) if n else math.nan
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "replications": len(frame),
                "successful_replications": n,
                "failure_count": len(frame) - n,
                "bias": float(errors.mean()) if n else math.nan,
                "bias_mcse": float(errors.std(ddof=1) / math.sqrt(n))
                if n > 1
                else math.nan,
                "rmse": rmse,
                "rmse_mcse": rmse_mcse,
                "coverage": coverage,
                "coverage_mcse": math.sqrt(coverage * (1.0 - coverage) / n)
                if n
                else math.nan,
                "mean_interval_width": float(width.mean()) if n else math.nan,
                "width_mcse": float(width.std(ddof=1) / math.sqrt(n))
                if n > 1
                else math.nan,
                "mean_actual_current_information": float(
                    success.actual_current_information.mean()
                )
                if n
                else math.nan,
                "mean_censoring_fraction": float(success.censoring_fraction.mean())
                if n
                else math.nan,
                "maximum_normalization_mass_discrepancy": float(
                    success.normalization_mass_discrepancy.max()
                )
                if n
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_summary(raw: pd.DataFrame) -> pd.DataFrame:
    success = raw[raw.status == "success"]
    pivot = success.pivot(
        index=["outcome", "scenario_id", "replicate_id"],
        columns="method",
        values="squared_error",
    )
    rows = []
    for (outcome, scenario), frame in pivot.groupby(level=[0, 1], sort=False):
        for comparator in METHODS:
            if comparator == "rUIP":
                continue
            values = (frame["rUIP"] - frame[comparator]).dropna().to_numpy(float)
            # Algebraically identical estimators can differ by a few ulps after
            # CSV round-tripping. Treat such numerical ties as ties rather than
            # directional wins.
            tie_tolerance = 1e-14
            rows.append(
                {
                    "outcome": outcome,
                    "scenario_id": scenario,
                    "comparator": comparator,
                    "paired_replications": values.size,
                    "mean_mse_difference_ruip_minus_comparator": float(values.mean()),
                    "mcse_mse_difference": float(
                        values.std(ddof=1) / math.sqrt(values.size)
                    )
                    if values.size > 1
                    else math.nan,
                    "probability_ruip_lower_squared_error": float(
                        np.mean(values < -tie_tolerance)
                    ),
                }
            )
    return pd.DataFrame(rows)


def quadrant_summary(raw: pd.DataFrame) -> pd.DataFrame:
    success = raw[raw.status == "success"].copy()
    # Pool squared errors before taking a square root; never average scenario RMSEs.
    return (
        success.groupby(
            ["outcome", "global_conflict", "local_conflict", "method"], as_index=False
        )
        .agg(
            mse=("squared_error", "mean"),
            bias=("error", "mean"),
            coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
            replications=("replicate_id", "count"),
        )
        .assign(rmse=lambda x: np.sqrt(x.mse))
    )


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


def main() -> None:
    """Generate one dataset per replication and reuse it across all priors.
    Write replicate-level summaries and a run manifest to the requested output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--outcomes",
        nargs="+",
        choices=ALL_OUTCOMES,
        default=list(ALL_OUTCOMES),
        help=(
            "Outcome subset. Separate processes may run disjoint outcomes in parallel."
        ),
    )
    args = parser.parse_args()
    if args.replications <= 1:
        parser.error("--replications must exceed one")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    abi, dll = load_frozen_engine()
    start = time.perf_counter()
    rows = []
    invariance = []
    outcomes = tuple(args.outcomes)
    for outcome in outcomes:
        oi = ALL_OUTCOMES.index(outcome)
        for si, scenario in enumerate(SCENARIOS):
            for replicate in range(args.replications):
                sequence = np.random.SeedSequence(
                    args.seed, spawn_key=(oi, si, replicate)
                )
                seed_value = int(sequence.generate_state(1, dtype=np.uint64)[0])
                dataset = make_dataset(
                    outcome, scenario, np.random.default_rng(sequence)
                )
                if replicate == 0:
                    regenerated = replace(
                        dataset,
                        current_estimate=dataset.current_estimate + 0.37,
                        current_information=max(
                            1.0, dataset.current_information * 0.83
                        ),
                        current_events=max(1, dataset.current_events - 3),
                        current_exposure=dataset.current_exposure + 2.5,
                        current_censored=min(
                            dataset.current_subjects, dataset.current_censored + 3
                        ),
                    )
                    for method in METHODS[1:]:
                        if method == "PP":
                            original = prior_fingerprint(
                                normalized_power_prior(dataset)
                            )
                            regenerated_current = prior_fingerprint(
                                normalized_power_prior(regenerated)
                            )
                        else:
                            original_prior = (
                                ruip_prior(dataset, method)
                                if method.startswith("rUIP")
                                else frozen_prior(dataset, method, abi, dll)
                            )
                            regenerated_prior = (
                                ruip_prior(regenerated, method)
                                if method.startswith("rUIP")
                                else frozen_prior(regenerated, method, abi, dll)
                            )
                            original = prior_fingerprint(original_prior)
                            regenerated_current = prior_fingerprint(regenerated_prior)
                        invariance.append(
                            {
                                "outcome": outcome,
                                "scenario_id": scenario.scenario_id,
                                "method": method,
                                "original": original,
                                "regenerated_current": regenerated_current,
                                "passed": original == regenerated_current,
                            }
                        )
                for method in METHODS:
                    row = {
                        "outcome": outcome,
                        "scenario_id": scenario.scenario_id,
                        "global_conflict": scenario.global_conflict,
                        "local_conflict": scenario.local_conflict,
                        "replicate_id": replicate,
                        "data_seed": seed_value,
                        "method": method,
                    }
                    try:
                        mean, variance, lower, upper, qerr, prior = analyze(
                            dataset, method, abi, dll
                        )
                        finite = np.all(
                            np.isfinite([mean, variance, lower, upper, qerr])
                        )
                        if not finite or variance < 0 or lower > upper:
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
                            prior_precision=(
                                float(1.0 / prior.variances[0])
                                if prior is not None and prior.means.size == 1
                                else math.nan
                            ),
                        )
                        if prior is not None and prior.local_weights is not None:
                            for k in range(3):
                                row[f"q{k + 1}"] = float(prior.local_weights[k])
                    except Exception as exc:  # retained in raw failure audit
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
    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    paired = paired_summary(raw)
    quadrants = quadrant_summary(raw)
    raw_path = output / "replicate_results.csv.gz"
    summary_path = output / "scenario_method_summary.csv"
    paired_path = output / "paired_mse_summary.csv"
    quadrant_path = output / "quadrant_summary.csv"
    invariance_path = output / "prior_invariance.csv"
    raw.to_csv(raw_path, index=False, compression="gzip")
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    quadrants.to_csv(quadrant_path, index=False)
    pd.DataFrame(invariance).to_csv(invariance_path, index=False)
    failures = int((raw.status != "success").sum())
    maximum_mass_discrepancy = float(raw.normalization_mass_discrepancy.max())
    elapsed = time.perf_counter() - start
    manifest = {
        "status": "PASS"
        if failures == 0 and maximum_mass_discrepancy < 1e-4
        else "FAIL",
        "seed": args.seed,
        "replications": args.replications,
        "paired_replications": True,
        "outcomes": list(outcomes),
        "scenarios": [s.scenario_id for s in SCENARIOS],
        "methods": list(METHODS),
        "datasets": len(outcomes) * len(SCENARIOS) * args.replications,
        "analyses": len(raw),
        "failure_count": failures,
        "maximum_normalization_mass_discrepancy": maximum_mass_discrepancy,
        "normalization_mass_discrepancy_tolerance": 1e-4,
        # Retained only so the already-frozen artifact merger can read newly
        # generated manifests; the two canonical fields above name the
        # diagnostic correctly.
        "maximum_quadrature_error": maximum_mass_discrepancy,
        "quadrature_tolerance": 1e-4,
        "deprecated_quadrature_field_note": (
            "compatibility alias for normalization-mass discrepancy; it is "
            "not an interval-endpoint accuracy measure"
        ),
        "numerical_gate_definition": (
            "relative discrepancy between 201- and 401-node trapezoid "
            "normalization masses under shared log-kernel scaling; this is "
            "not an interval-endpoint accuracy bound"
        ),
        "prior_invariance_checks": len(invariance),
        "prior_invariance_failures": sum(not x["passed"] for x in invariance),
        "binary_planned_subjects": int(round(CURRENT_TARGET_INFORMATION / 0.24)),
        "survival_planned_subjects": int(
            2 * math.ceil(CURRENT_TARGET_INFORMATION / (0.7 * 0.25) / 2)
        ),
        "survival_baseline_hazard": BASELINE_HAZARD,
        "survival_censoring_hazard": CENSORING_HAZARD,
        "survival_model": "Cox proportional hazards with randomized 1:1 treatment",
        "survival_baseline_role": "exponential DGP only; unavailable to analysis",
        "survival_summary_rule": "Cox partial-likelihood MLE and observed information",
        "power_prior": {
            "historical_likelihood": "independent Gaussian summary likelihoods",
            "powers": "independent Beta(1,1)",
            "normalization": "conditional on each power vector before mixing",
            "base_prior": {"mean": POWER_BASE_MEAN, "sd": POWER_BASE_SD},
            "quadrature": f"tensor Gauss-Legendre order {POWER_QUADRATURE_ORDER}",
            "binary_current_likelihood": "exact binomial logit",
            "survival_current_likelihood": "exact Cox partial likelihood",
        },
        "frozen_comparator_library": str(FROZEN_LIBRARY.relative_to(ROOT)),
        "frozen_comparator_sha256": sha256(FROZEN_LIBRARY),
        "runner_sha256": sha256(Path(__file__)),
        "git_head": git_head(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_seconds": elapsed,
        "files": {},
    }
    for path in (raw_path, summary_path, paired_path, quadrant_path, invariance_path):
        manifest["files"][path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0 if manifest["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
