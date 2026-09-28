"""Shared, history-only data contracts for rUIP prior construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class HistoricalSummary:
    """A single historical-study summary on the analysis scale.

    ``h_k`` is the study estimate, ``I_Uk`` its per-unit information, and
    ``n_k`` the historical sample size.  The deliberately small contract has
    no current-trial fields, so it can be safely supplied to rUIP without
    leaking a planned design quantity into the prior.
    """

    source_id: str
    h_k: float
    n_k: int
    I_Uk: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        for name, value in (("h_k", self.h_k), ("I_Uk", self.I_Uk)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.n_k, bool) or not isinstance(self.n_k, Integral):
            raise ValueError("n_k must be an integer sample size")
        if self.n_k < 1:
            raise ValueError("n_k must be at least one")
        if float(self.I_Uk) <= 0:
            raise ValueError("I_Uk must be positive")


def validated_histories(
    histories: Iterable[HistoricalSummary],
) -> tuple[HistoricalSummary, ...]:
    """Materialize and validate a non-empty collection of unique studies."""

    values = tuple(histories)
    if not values:
        raise ValueError("at least one historical summary is required")
    if not all(isinstance(item, HistoricalSummary) for item in values):
        raise TypeError("histories must contain HistoricalSummary objects")
    ids = [item.source_id for item in values]
    if len(set(ids)) != len(ids):
        raise ValueError("historical source_id values must be unique")
    return values


@dataclass(frozen=True, slots=True)
class NormalPrior:
    """A proper Normal prior represented by its location and precision."""

    mean: float
    precision: float

    def __post_init__(self) -> None:
        if not isfinite(float(self.mean)):
            raise ValueError("mean must be finite")
        if not isfinite(float(self.precision)) or float(self.precision) <= 0:
            raise ValueError("precision must be finite and positive")

    @property
    def variance(self) -> float:
        return 1.0 / self.precision
