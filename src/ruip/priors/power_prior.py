"""Fixed-power Gaussian summary-data comparator.

The comparator uses the common, prespecified power ``a0 = 0.5``.  For
historical Normal summaries with information ``J_k = n_k I_{Uk}``, the
resulting prior is Normal with pooled information ``a0 sum_k J_k``.  The
same construction is used on the corrected-logit summary scale for binary
outcomes; the current likelihood itself remains exact Binomial.
"""

from __future__ import annotations

from collections.abc import Iterable

from .common import HistoricalSummary, NormalPrior, validated_histories

POWER_WEIGHT = 0.5


def power_prior(
    histories: Iterable[HistoricalSummary], *, a0: float = POWER_WEIGHT
) -> NormalPrior:
    """Construct the fixed-power Normal prior from historical summaries.

    ``a0`` is exposed only to make the mathematical contract testable.  The
    production dispatch always uses the common value ``0.5``.
    """
    if not 0.0 < a0 <= 1.0:
        raise ValueError("a0 must lie in (0, 1]")
    values = validated_histories(histories)
    information = [item.n_k * item.I_Uk for item in values]
    total_information = sum(information)
    mean = (
        sum(j * item.h_k for j, item in zip(information, values, strict=True))
        / total_information
    )
    return NormalPrior(mean=mean, precision=a0 * total_information)
