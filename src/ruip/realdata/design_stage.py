"""Design-stage clinical analyses from published arm-level summaries."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.special import logsumexp, roots_legendre

from ruip.analysis.scalable_comparators import (
    _binary_from_control_nodes,
    cp_binary,
    cp_continuous,
    uip_binary,
    uip_continuous,
)
from ruip.outcomes import (
    BinomialSummary,
    CurrentGroupSummary,
    nip_posterior,
    rmap_posterior,
    ruip_posterior,
)
from ruip.outcomes.continuous import _mixture_result, _NormalComponent
from ruip.priors import HistoricalSummary, rmap_prior, robust_uip_prior
from ruip.realdata.normalized_power import (
    log_mixture_density,
    normalized_power_components,
)

METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP")


@dataclass(frozen=True, slots=True)
class ClinicalAnalysisResult:
    application: str
    source_set: str
    omitted_source: str
    method: str
    status: str
    margin: float | None
    margin_label: str
    estimate: float
    posterior_sd: float
    interval_lower: float
    interval_upper: float
    favorable_probability: float
    risk_difference_mean: float | None = None
    risk_difference_lower: float | None = None
    risk_difference_upper: float | None = None
    prior_mean: float | None = None
    prior_precision: float | None = None
    local_variance: float | None = None
    global_discount: float | None = None
    effective_information_units: float | None = None
    source_weights_json: str = ""
    diagnostics_json: str = "{}"

    def row(self) -> dict[str, object]:
        return asdict(self)


def _matrix(histories: Iterable[HistoricalSummary]) -> np.ndarray:
    return np.asarray(
        [(item.n_k, item.h_k, item.I_Uk) for item in histories], dtype=float
    )


def _continuous_row(
    source_set: str,
    omitted_source: str,
    method: str,
    result,
    *,
    margin: float | None = None,
    margin_label: str = "not_applicable",
) -> ClinicalAnalysisResult:
    return ClinicalAnalysisResult(
        "memantine_npi",
        source_set,
        omitted_source,
        method,
        "success",
        margin if method == "rUIP" else None,
        margin_label if method == "rUIP" else "not_applicable",
        float(result.delta_mean),
        math.sqrt(float(result.delta_var)),
        float(result.delta_q025),
        float(result.delta_q975),
        float(1.0 - result.probability_delta_positive),
        diagnostics_json=json.dumps(result.diagnostics, sort_keys=True),
    )


def _continuous_mapping(
    source_set: str,
    omitted_source: str,
    method: str,
    result: dict[str, object],
) -> ClinicalAnalysisResult:
    return ClinicalAnalysisResult(
        "memantine_npi",
        source_set,
        omitted_source,
        method,
        "success",
        None,
        "not_applicable",
        float(result["estimate"]),
        float(result["posterior_sd"]),
        float(result["interval_lower"]),
        float(result["interval_upper"]),
        1.0 - float(result["posterior_probability_positive"]),
        diagnostics_json=json.dumps(result.get("diagnostics", {}), sort_keys=True),
    )


def analyze_memantine(
    histories: Iterable[HistoricalSummary],
    control: CurrentGroupSummary,
    treatment: CurrentGroupSummary,
    *,
    delta_clin: float,
    margin_label: str,
    source_set: str = "full",
    omitted_source: str = "",
    uip_seed: int = 202609187901,
    uip_scrambles: int = 4,
    uip_initial_power: int = 11,
    uip_convergence_power: int = 12,
    pp_power: int = 14,
) -> list[ClinicalAnalysisResult]:
    """Analyze treatment-minus-control NPI change; negative values favor treatment."""

    values = tuple(histories)
    matrix = _matrix(values)
    pp_mean, pp_variance, pp_weight, pp_diagnostics = normalized_power_components(
        values, qmc_power=pp_power
    )
    pp = _mixture_result(
        "power_prior",
        (
            _NormalComponent(float(m), float(v), float(w), {})
            for m, v, w in zip(pp_mean, pp_variance, pp_weight, strict=True)
        ),
        control,
        treatment,
    )
    pp = replace(pp, diagnostics={**pp.diagnostics, **pp_diagnostics})
    rows = [
        _continuous_row(
            source_set,
            omitted_source,
            "NIP",
            nip_posterior(control, treatment),
        ),
        _continuous_row(
            source_set,
            omitted_source,
            "PP",
            pp,
        ),
        _continuous_row(
            source_set,
            omitted_source,
            "rMAP",
            rmap_posterior(values, control, treatment),
        ),
        _continuous_mapping(
            source_set,
            omitted_source,
            "CP",
            cp_continuous(
                matrix,
                (control.n, control.mean),
                (treatment.n, treatment.mean),
                control.known_variance,
            ),
        ),
        _continuous_mapping(
            source_set,
            omitted_source,
            "UIP",
            uip_continuous(
                matrix,
                (control.n, control.mean),
                (treatment.n, treatment.mean),
                control.known_variance,
                control.n,
                scrambles=uip_scrambles,
                initial_power=uip_initial_power,
                convergence_power=uip_convergence_power,
                seed=uip_seed,
            ),
        ),
    ]
    result = ruip_posterior(values, control, treatment, delta_clin=delta_clin)
    prior = robust_uip_prior(values, delta_clin=delta_clin)
    base = _continuous_row(
        source_set,
        omitted_source,
        "rUIP",
        result,
        margin=delta_clin,
        margin_label=margin_label,
    )
    rows.append(
        ClinicalAnalysisResult(
            **{
                **base.row(),
                "prior_mean": prior.mean,
                "prior_precision": prior.precision,
                "local_variance": prior.local_variance,
                "global_discount": prior.global_discount,
                "effective_information_units": prior.effective_information_units,
                "source_weights_json": json.dumps(
                    {
                        item.source_id: weight
                        for item, weight in zip(
                            prior.histories, prior.weights, strict=True
                        )
                    },
                    sort_keys=True,
                ),
            }
        )
    )
    return rows


def _grid(log_prior, control: BinomialSummary, bound: float, order: int):
    raw, weights = roots_legendre(order)
    eta = bound * raw
    log_mass = (
        np.log(weights * bound)
        + np.asarray(log_prior(eta), dtype=float)
        + control.events * eta
        - control.n * np.logaddexp(0.0, eta)
    )
    return eta, np.exp(log_mass - logsumexp(log_mass))


def _binary_row(
    source_set: str,
    omitted_source: str,
    method: str,
    result: dict[str, object],
    *,
    margin: float | None = None,
    margin_label: str = "not_applicable",
    prior=None,
) -> ClinicalAnalysisResult:
    row = ClinicalAnalysisResult(
        "secukinumab_asas20",
        source_set,
        omitted_source,
        method,
        "success",
        margin if method == "rUIP" else None,
        margin_label if method == "rUIP" else "not_applicable",
        float(result["estimate"]),
        float(result["posterior_sd"]),
        float(result["interval_lower"]),
        float(result["interval_upper"]),
        float(result["posterior_probability_positive"]),
        float(result["risk_difference_mean"]),
        float(result["risk_difference_q025"]),
        float(result["risk_difference_q975"]),
        diagnostics_json=json.dumps(result.get("diagnostics", {}), sort_keys=True),
    )
    if prior is None:
        return row
    return ClinicalAnalysisResult(
        **{
            **row.row(),
            "prior_mean": prior.mean,
            "prior_precision": prior.precision,
            "local_variance": prior.local_variance,
            "global_discount": prior.global_discount,
            "effective_information_units": prior.effective_information_units,
            "source_weights_json": json.dumps(
                {
                    item.source_id: weight
                    for item, weight in zip(prior.histories, prior.weights, strict=True)
                },
                sort_keys=True,
            ),
        }
    )


def analyze_secukinumab(
    histories: Iterable[HistoricalSummary],
    historical_events: Iterable[int],
    control: BinomialSummary,
    treatment: BinomialSummary,
    *,
    delta_clin: float,
    margin_label: str,
    source_set: str = "full",
    omitted_source: str = "",
    uip_seed: int = 202609187902,
    uip_scrambles: int = 4,
    uip_initial_power: int = 11,
    uip_convergence_power: int = 12,
    theta_order: int = 768,
    pp_power: int = 14,
) -> list[ClinicalAnalysisResult]:
    """Analyze log odds ratio and risk difference; positive values favor treatment."""

    values = tuple(histories)
    event = np.asarray(tuple(historical_events), dtype=float)
    matrix = _matrix(values)
    if len(event) != len(values):
        raise ValueError("historical event counts must match historical summaries")
    eta, mass = _grid(lambda x: np.zeros_like(x), control, 24.0, theta_order)
    rows = [
        _binary_row(
            source_set,
            omitted_source,
            "NIP",
            _binary_from_control_nodes(eta, mass, (treatment.n, treatment.events)),
        )
    ]
    pp_mean, pp_variance, pp_weight, pp_diagnostics = normalized_power_components(
        values, qmc_power=pp_power
    )
    eta, mass = _grid(
        lambda x: log_mixture_density(x, pp_mean, pp_variance, pp_weight),
        control,
        24.0,
        theta_order,
    )
    result = _binary_from_control_nodes(eta, mass, (treatment.n, treatment.events))
    result["diagnostics"] = {
        **pp_diagnostics,
        "historical_likelihood": "Gaussian_corrected_logit_summary",
        "current_likelihood": "exact_binomial",
    }
    rows.append(_binary_row(source_set, omitted_source, "PP", result))

    rmap = rmap_prior(values)
    eta, mass = _grid(
        lambda x: np.asarray([math.log(rmap.density(float(v))) for v in x]),
        control,
        24.0,
        theta_order,
    )
    result = _binary_from_control_nodes(eta, mass, (treatment.n, treatment.events))
    result["diagnostics"] = {
        "robust_weight": rmap.robust_weight,
        "tau_quadrature_nodes": len(rmap.tau_nodes),
    }
    rows.append(_binary_row(source_set, omitted_source, "rMAP", result))
    rows.append(
        _binary_row(
            source_set,
            omitted_source,
            "CP",
            cp_binary(
                matrix,
                (control.n, control.events),
                (treatment.n, treatment.events),
                order=theta_order,
            ),
        )
    )
    rows.append(
        _binary_row(
            source_set,
            omitted_source,
            "UIP",
            uip_binary(
                matrix,
                (control.n, control.events),
                (treatment.n, treatment.events),
                control.n,
                scrambles=uip_scrambles,
                initial_power=uip_initial_power,
                convergence_power=uip_convergence_power,
                seed=uip_seed,
                theta_order=theta_order,
            ),
        )
    )
    prior = robust_uip_prior(values, delta_clin=delta_clin)
    eta, mass = _grid(
        lambda x: (
            -0.5
            * (
                math.log(2.0 * math.pi * prior.variance)
                + (x - prior.mean) ** 2 / prior.variance
            )
        ),
        control,
        24.0,
        theta_order,
    )
    result = _binary_from_control_nodes(eta, mass, (treatment.n, treatment.events))
    result["diagnostics"] = {
        "integration": "Gauss-Legendre_exact_binomial_likelihood",
        "theta_order": theta_order,
        "theta_bound": 24.0,
    }
    rows.append(
        _binary_row(
            source_set,
            omitted_source,
            "rUIP",
            result,
            margin=delta_clin,
            margin_label=margin_label,
            prior=prior,
        )
    )
    return rows


def validate_rows(rows: Iterable[ClinicalAnalysisResult]) -> None:
    values = tuple(rows)
    if not values:
        raise ValueError("no clinical analysis rows")
    for row in values:
        numeric = (
            row.estimate,
            row.posterior_sd,
            row.interval_lower,
            row.interval_upper,
            row.favorable_probability,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite result: {row.application}/{row.method}")
        if row.posterior_sd < 0 or row.interval_lower > row.interval_upper:
            raise ValueError(f"invalid interval: {row.application}/{row.method}")
        if not 0.0 <= row.favorable_probability <= 1.0:
            raise ValueError(f"invalid probability: {row.application}/{row.method}")
