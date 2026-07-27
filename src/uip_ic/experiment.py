"""Shared experiment assembly for the demo and repeated simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_generation import (
    ExactSurvivalData,
    IntervalCensoredData,
    generate_historical_summaries,
    generate_interval_censored_data,
)
from .interval_ph import PiecewiseBaseline
from .priors import HistoricalSummary, UIPSpecification, build_uip
from .samplers import SamplerConfig
from .utils import derived_seed


@dataclass(frozen=True)
class ReplicateInputs:
    current: IntervalCensoredData
    histories: tuple[ExactSurvivalData, ...]
    historical_summaries: tuple[HistoricalSummary, ...]
    uip: UIPSpecification
    baseline: PiecewiseBaseline
    theta_true: float
    historical_truths: np.ndarray


def sampler_config_from_mapping(values: dict[str, Any]) -> SamplerConfig:
    fields = SamplerConfig.__dataclass_fields__
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown sampler configuration fields: {sorted(unknown)}")
    return SamplerConfig(**values)


def simulate_replicate(
    config: dict[str, Any],
    scenario_name: str,
    scenario_index: int,
    repetition: int,
) -> ReplicateInputs:
    experiment = config["experiment"]
    scenario = config["scenarios"][scenario_name]
    baseline = PiecewiseBaseline(
        np.asarray(experiment["baseline_interval_starts"], dtype=float),
        np.asarray(experiment["baseline_hazards"], dtype=float),
    )
    theta_true = float(experiment["theta_current"])
    beta_true = float(experiment["beta_true"])
    base_seed = int(experiment["seed"])
    # Common random numbers across scenarios isolate the historical-effect shift:
    # current data and underlying historical uniforms match within a repetition.
    _ = scenario_index
    current_rng = np.random.default_rng(derived_seed(base_seed, 0, repetition, 0))
    history_rng = np.random.default_rng(derived_seed(base_seed, 0, repetition, 1))
    current = generate_interval_censored_data(
        current_rng,
        int(experiment["n_current"]),
        baseline,
        theta_true,
        beta_true,
        experiment["inspection_times"],
        bool(experiment["include_covariate"]),
    )
    historical_truths = theta_true + np.asarray(scenario["historical_offsets"], dtype=float)
    summaries, histories = generate_historical_summaries(
        history_rng,
        int(experiment["n_historical"]),
        baseline,
        historical_truths,
        beta_true,
        float(experiment["administrative_time"]),
        bool(experiment["include_covariate"]),
    )
    uip = build_uip(
        summaries,
        float(experiment["m_max"]),
        str(experiment.get("weighting", "equal")),
    )
    return ReplicateInputs(
        current=current,
        histories=tuple(histories),
        historical_summaries=tuple(summaries),
        uip=uip,
        baseline=baseline,
        theta_true=theta_true,
        historical_truths=historical_truths,
    )
