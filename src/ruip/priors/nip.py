"""No-information-prior interface for current-data analyses."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class NoInformationPrior:
    """A named flat-prior convention, with no historical-data input.

    This object intentionally has no density calculation: an improper flat
    prior is a modeling convention and is combined with an outcome likelihood
    in later milestones.
    """

    location: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(float(self.location)):
            raise ValueError("location must be finite")

    @property
    def is_improper_flat(self) -> bool:
        return True


def no_information_prior(*, location: float = 0.0) -> NoInformationPrior:
    """Return the NIP comparator without accepting histories or design data."""

    return NoInformationPrior(location=location)
