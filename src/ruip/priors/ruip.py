"""Authoritative scalar capped-medoid robust unit information prior."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, isfinite, log, pi

from .common import HistoricalSummary, validated_histories

_LOG_2PI = log(2.0 * pi)


@dataclass(frozen=True, slots=True)
class DesignStageRUIPPrior:
    """Single Normal rUIP prior frozen before current outcomes are observed."""

    histories: tuple[HistoricalSummary, ...]
    delta_clin: float
    anchor_index: int
    compatibility: tuple[float, ...]
    effective_information: tuple[float, ...]
    working_variances: tuple[float, ...]
    weights: tuple[float, ...]
    mean: float
    local_variance: float
    local_precision: float
    global_discount: float
    precision: float
    average_unit_information: float
    effective_information_units: float

    @property
    def variance(self) -> float:
        """Prior variance, equal to ``local_variance + delta_clin**2``."""

        return 1.0 / self.precision

    def log_density(self, theta: float) -> float:
        """Evaluate the normalized Normal log density."""

        value = float(theta)
        if not isfinite(value):
            raise ValueError("theta must be finite")
        return -0.5 * (
            _LOG_2PI + log(self.variance) + (value - self.mean) ** 2 / self.variance
        )

    def density(self, theta: float) -> float:
        """Evaluate the normalized Normal density."""

        return exp(self.log_density(theta))


def design_stage_ruip_prior(
    histories: Iterable[HistoricalSummary], *, delta_clin: float
) -> DesignStageRUIPPrior:
    """Build the authoritative scalar rUIP from histories and a frozen margin.

    The source medoid anchors a source-count consensus.  Gaussian pairwise
    compatibility scores downweight discordance, and each non-anchor source's
    effective information is capped at the anchor information.  The prior
    variance is ``1/sum_k R_k + delta_clin**2``.  The current outcomes are not
    used.
    """

    values = validated_histories(histories)
    margin = float(delta_clin)
    if not isfinite(margin) or margin <= 0.0:
        raise ValueError("delta_clin must be finite and positive")

    information = tuple(float(item.n_k) * float(item.I_Uk) for item in values)
    estimates = tuple(float(item.h_k) for item in values)
    objectives = tuple(
        sum(abs(value - other) for other in estimates) for value in estimates
    )
    anchor = min(
        range(len(values)),
        key=lambda index: (
            objectives[index],
            -information[index],
            values[index].source_id,
        ),
    )
    compatibility = []
    contributions = []
    for index, estimate in enumerate(estimates):
        if index == anchor:
            score = 1.0
        else:
            variance = 1.0 / information[index] + 1.0 / information[anchor]
            score = exp(-0.5 * (estimate - estimates[anchor]) ** 2 / variance)
        compatibility.append(score)
        contributions.append(min(information[index], information[anchor]) * score)
    contributions[anchor] = information[anchor]
    local_precision = sum(contributions)
    weights = tuple(value / local_precision for value in contributions)
    working = tuple(1.0 / value for value in contributions)
    mean = sum(
        weight * float(item.h_k) for weight, item in zip(weights, values, strict=True)
    )
    local_variance = 1.0 / local_precision
    variance = local_variance + margin * margin
    precision = 1.0 / variance
    average_unit_information = sum(
        weight * float(item.I_Uk) for weight, item in zip(weights, values, strict=True)
    )
    return DesignStageRUIPPrior(
        histories=values,
        delta_clin=margin,
        anchor_index=anchor,
        compatibility=tuple(compatibility),
        effective_information=tuple(contributions),
        working_variances=working,
        weights=weights,
        mean=mean,
        local_variance=local_variance,
        local_precision=local_precision,
        global_discount=local_variance / variance,
        precision=precision,
        average_unit_information=average_unit_information,
        effective_information_units=precision / average_unit_information,
    )


def robust_uip_prior(
    histories: Iterable[HistoricalSummary], *, delta_clin: float
) -> DesignStageRUIPPrior:
    """Public rUIP constructor; an explicit alias for the design-stage rule."""

    return design_stage_ruip_prior(histories, delta_clin=delta_clin)
