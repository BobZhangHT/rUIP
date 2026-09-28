"""Shared mechanics for the precision-leverage confirmation experiment.

The functions in this module deliberately receive the authoritative
three-outcome implementation as an argument.  This keeps the scientific
invariants in one place while allowing the command-line runner to load its
frozen simulation authority explicitly:

* historical information may be changed for a prespecified design cell;
* current outcomes never enter construction of a borrowing prior; and
* every comparator uses the same dataset and posterior implementation.

The module contains no scenario selection, file I/O, or reporting logic.
"""

from __future__ import annotations

from dataclasses import replace


def make_dataset(tri, outcome: str, scenario, information, rng):
    """Generate one paired dataset at a prespecified information profile.

    ``tri.make_dataset`` reads its historical information target from the
    authoritative module.  The override is restored in ``finally`` so a
    failed replicate cannot contaminate later simulations.
    """

    original = tri.HISTORICAL_TARGET_INFORMATION
    try:
        tri.HISTORICAL_TARGET_INFORMATION = information.copy()
        return tri.make_dataset(outcome, scenario, rng)
    finally:
        tri.HISTORICAL_TARGET_INFORMATION = original


def prior_invariant_to_current(tri, dataset, method: str, abi, dll) -> bool:
    """Check that a borrowing prior is fixed before current outcomes are seen."""

    changed = replace(
        dataset,
        current_estimate=dataset.current_estimate + 0.51,
        current_information=max(1.0, dataset.current_information * 0.74),
        current_events=max(1, dataset.current_events - 2),
    )
    if method == "NIP":
        return True
    if method == "PP":
        return tri.prior_fingerprint(
            tri.normalized_power_prior(dataset)
        ) == tri.prior_fingerprint(tri.normalized_power_prior(changed))
    before = (
        tri.ruip_prior(dataset, method)
        if method == "rUIP"
        else tri.frozen_prior(dataset, method, abi, dll)
    )
    after = (
        tri.ruip_prior(changed, method)
        if method == "rUIP"
        else tri.frozen_prior(changed, method, abi, dll)
    )
    return tri.prior_fingerprint(before) == tri.prior_fingerprint(after)


def analyze(tri, dataset, method: str, abi, dll):
    """Analyze one paired dataset with the authoritative method implementation."""

    return tri.analyze(dataset, method, abi, dll)
