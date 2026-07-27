"""Uniform interface to the proof-of-concept comparison methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .data_generation import IntervalCensoredData
from .evaluation import summarize_chain
from .priors import UIPSpecification
from .samplers import SamplerConfig, run_da_sampler


@dataclass(frozen=True)
class FitResult:
    method: str
    draws: pd.DataFrame
    summary: dict[str, float]
    runtime_seconds: float
    uip_mean: float | None
    uip_unit_information: float | None
    m_max: float | None


def fit_method(
    method: str,
    data: IntervalCensoredData,
    interval_starts: np.ndarray,
    config: SamplerConfig,
    seed: int,
    uip: UIPSpecification | None = None,
) -> FitResult:
    start = perf_counter()
    draws = run_da_sampler(data, interval_starts, method, config, np.random.default_rng(seed), uip)
    elapsed = perf_counter() - start
    return FitResult(
        method=method,
        draws=draws,
        summary=summarize_chain(draws),
        runtime_seconds=float(elapsed),
        uip_mean=None if uip is None else uip.mean,
        uip_unit_information=None if uip is None else uip.unit_information,
        m_max=None if uip is None else uip.m_max,
    )
