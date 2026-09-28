"""The standard unit-information-prior comparator (``UIP2`` in code only)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .common import HistoricalSummary, validated_histories


@dataclass(frozen=True, slots=True)
class StandardUIPHyperparameters:
    """Frozen standard-UIP hyperparameters induced by historical sizes and n_C."""

    gammas: tuple[float, ...]
    M_max: float


@dataclass(frozen=True, slots=True)
class StandardUIPConditional:
    """Conditional Normal UIP given Dirichlet weights and a borrowing size.

    ``precision=0`` represents the valid ``M=0`` zero-borrowing boundary. It
    is deliberately not a ``NormalPrior``, whose contract requires a proper
    positive precision.
    """

    mean: float
    precision: float
    weights: tuple[float, ...]
    M: float
    M_max: float

    @property
    def is_zero_borrowing(self) -> bool:
        return self.precision == 0.0


@dataclass(frozen=True, slots=True)
class StandardUIPMeanSummary:
    """Expected conditional parameters, not the full random standard UIP."""

    mean: float
    expected_precision: float
    expected_weights: tuple[float, ...]
    expected_M: float
    M_max: float


def standard_uip_hyperparameters(
    histories: Iterable[HistoricalSummary], *, planned_n_control: float
) -> StandardUIPHyperparameters:
    """Return gamma_k=min(1,n_k/n_C) and M_max=min(n_C,sum n_k)."""

    values = validated_histories(histories)
    if not isfinite(float(planned_n_control)) or float(planned_n_control) <= 0:
        raise ValueError("planned_n_control must be finite and positive")
    n_control = float(planned_n_control)
    gammas = tuple(min(1.0, item.n_k / n_control) for item in values)
    return StandardUIPHyperparameters(
        gammas=gammas,
        M_max=min(n_control, float(sum(item.n_k for item in values))),
    )


def standard_uip_conditional(
    histories: Iterable[HistoricalSummary],
    *,
    weights: Iterable[float],
    M: float,
    hyperparameters: StandardUIPHyperparameters,
) -> StandardUIPConditional:
    """Compute mu=sum(w_k h_k), J=M sum(w_k I_Uk) for a valid draw."""

    values = validated_histories(histories)
    w = tuple(float(value) for value in weights)
    if len(w) != len(values):
        raise ValueError("weights must have one entry per historical study")
    if not all(isfinite(value) and value >= 0 for value in w):
        raise ValueError("weights must be finite and non-negative")
    if not np.isclose(sum(w), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("weights must lie on the unit simplex")
    if len(hyperparameters.gammas) != len(values) or any(
        gamma <= 0 or not isfinite(gamma) for gamma in hyperparameters.gammas
    ):
        raise ValueError("hyperparameters do not match valid historical studies")
    size = float(M)
    if not isfinite(size) or not 0.0 <= size <= hyperparameters.M_max:
        raise ValueError("M must be finite and satisfy 0 <= M <= M_max")
    mean = sum(weight * item.h_k for weight, item in zip(w, values, strict=True))
    unit_information = sum(
        weight * item.I_Uk for weight, item in zip(w, values, strict=True)
    )
    return StandardUIPConditional(
        mean=mean,
        precision=size * unit_information,
        weights=w,
        M=size,
        M_max=hyperparameters.M_max,
    )


def sample_standard_uip_conditional(
    histories: Iterable[HistoricalSummary],
    *,
    planned_n_control: float,
    rng: np.random.Generator,
) -> StandardUIPConditional:
    """Sample w~Dirichlet(gamma), M~Uniform(0,M_max) using explicit RNG."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    values = validated_histories(histories)
    hyper = standard_uip_hyperparameters(values, planned_n_control=planned_n_control)
    return standard_uip_conditional(
        values,
        weights=rng.dirichlet(np.asarray(hyper.gammas, dtype=float)),
        M=float(rng.uniform(0.0, hyper.M_max)),
        hyperparameters=hyper,
    )


def standard_uip_mean_summary(
    histories: Iterable[HistoricalSummary], *, planned_n_control: float
) -> StandardUIPMeanSummary:
    """Summarize E[w]=gamma/sum(gamma), E[M]=M_max/2 without collapsing UIP."""

    values = validated_histories(histories)
    hyper = standard_uip_hyperparameters(values, planned_n_control=planned_n_control)
    gamma_total = sum(hyper.gammas)
    expected_weights = tuple(gamma / gamma_total for gamma in hyper.gammas)
    mean = sum(
        weight * item.h_k for weight, item in zip(expected_weights, values, strict=True)
    )
    expected_precision = (hyper.M_max / 2.0) * sum(
        weight * item.I_Uk
        for weight, item in zip(expected_weights, values, strict=True)
    )
    return StandardUIPMeanSummary(
        mean=mean,
        expected_precision=expected_precision,
        expected_weights=expected_weights,
        expected_M=hyper.M_max / 2.0,
        M_max=hyper.M_max,
    )
