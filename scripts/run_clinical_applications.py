"""Run the current scalar rUIP and five comparators on two public applications."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ruip.outcomes import BinomialSummary, CurrentGroupSummary  # noqa: E402
from ruip.priors import HistoricalSummary  # noqa: E402
from ruip.realdata import (  # noqa: E402
    METHODS,
    analyze_memantine,
    analyze_secukinumab,
    validate_rows,
)
from ruip.realdata.input_summaries import materialize_inputs  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results" / "realdata"
MEMANTINE_INPUT = ROOT / "data" / "processed" / "clinical_memantine_npi.csv"
SECUKINUMAB_INPUT = ROOT / "data" / "processed" / "clinical_secukinumab_asas20.csv"
MEMANTINE_MARGINS = (4.0, 8.0, 12.0)
SECUKINUMAB_OR_MARGINS = (1.25, 1.5, 2.0)
PRIMARY_MEMANTINE_MARGIN = 8.0
PRIMARY_SECUKINUMAB_OR = 1.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def load_memantine():
    """Convert published arm summaries to historical and current contrasts."""
    # Create the published aggregate input locally when checking out source only.
    materialize_inputs(ROOT)
    frame = pd.read_csv(MEMANTINE_INPUT)
    historical = frame[(frame.role == "historical") & (frame.arm == "placebo")]
    current = frame[frame.role == "current"].set_index("arm")
    histories = tuple(
        HistoricalSummary(
            str(row.study),
            float(row.mean_change),
            int(row.n),
            1.0 / float(row.sd) ** 2,
        )
        for row in historical.itertuples(index=False)
    )
    control_row, treatment_row = current.loc["placebo"], current.loc["treatment"]
    pooled_variance = (
        (int(control_row.n) - 1) * float(control_row.sd) ** 2
        + (int(treatment_row.n) - 1) * float(treatment_row.sd) ** 2
    ) / (int(control_row.n) + int(treatment_row.n) - 2)
    return (
        histories,
        CurrentGroupSummary(
            int(control_row.n), float(control_row.mean_change), pooled_variance
        ),
        CurrentGroupSummary(
            int(treatment_row.n), float(treatment_row.mean_change), pooled_variance
        ),
        pooled_variance,
    )


def load_secukinumab():
    """Convert published ASAS20 event counts to the binary analysis input."""
    # Both studies use the same source-controlled input materializer.
    materialize_inputs(ROOT)
    frame = pd.read_csv(SECUKINUMAB_INPUT)
    historical = frame[frame.role == "historical"]
    current = frame[frame.role == "current"].set_index("study")
    histories, events = [], []
    for row in historical.itertuples(index=False):
        p_tilde = (int(row.events) + 0.5) / (int(row.total) + 1.0)
        histories.append(
            HistoricalSummary(
                str(row.study),
                float(logit(p_tilde)),
                int(row.total),
                float(p_tilde * (1.0 - p_tilde)),
            )
        )
        events.append(int(row.events))
    placebo = current.loc["Baeten 2013 placebo"]
    treatment = current.loc["Baeten 2013 secukinumab"]
    return (
        tuple(histories),
        tuple(events),
        BinomialSummary(int(placebo.total), int(placebo.events)),
        BinomialSummary(int(treatment.total), int(treatment.events)),
    )


def _publish_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def _margin_specification() -> dict[str, object]:
    return {
        "frozen_before_analysis": True,
        "prohibition": (
            "No margin was selected from current outcomes or from the amount of "
            "borrowing produced by the analysis."
        ),
        "memantine_npi": {
            "analysis_scale": "full Neuropsychiatric Inventory change, 0-144",
            "primary_delta_clin": PRIMARY_MEMANTINE_MARGIN,
            "sensitivity_grid": list(MEMANTINE_MARGINS),
            "rationale": (
                "Howard et al. reported an 8-point NPI MCID for DOMINO, based on "
                "0.4 SD of change in the first 127 completers. The 4/8/12 grid "
                "uses one-half, the cited anchor, and 1.5 times that anchor."
            ),
            "source": {
                "citation": (
                    "Howard R et al. Determining the minimum clinically important "
                    "differences for outcomes in the DOMINO trial. Int J Geriatr "
                    "Psychiatry. 2011;26:812-817."
                ),
                "doi": "10.1002/gps.2607",
                "pmid": "20848576",
                "url": "https://pubmed.ncbi.nlm.nih.gov/20848576/",
                "scale_warning": "This is NPI, not the shorter NPI-Q scale.",
            },
        },
        "secukinumab_asas20": {
            "analysis_scale": "historical-control log odds",
            "primary_delta_clin": math.log(PRIMARY_SECUKINUMAB_OR),
            "primary_odds_ratio": PRIMARY_SECUKINUMAB_OR,
            "sensitivity_odds_ratio_grid": list(SECUKINUMAB_OR_MARGINS),
            "sensitivity_delta_grid": [
                math.log(value) for value in SECUKINUMAB_OR_MARGINS
            ],
            "rationale": (
                "No canonical common-displacement MCID was identified for this "
                "historical-control log-odds transport problem. OR 1.25/1.5/2.0 "
                "is a transparent prespecified outcome-scale sensitivity grid "
                "and must not be described as a validated clinical margin."
            ),
            "source": None,
        },
    }


def main() -> None:
    """Materialize the published aggregate inputs, freeze all historical priors,
    then update them with the current study likelihood."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uip-scrambles", type=int, default=4)
    parser.add_argument("--uip-initial-power", type=int, default=11)
    parser.add_argument("--uip-convergence-power", type=int, default=12)
    parser.add_argument("--theta-order", type=int, default=768)
    parser.add_argument("--pp-power", type=int, default=14)
    args = parser.parse_args()
    materialize_inputs(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hashes_before = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (MEMANTINE_INPUT, SECUKINUMAB_INPUT)
    }
    mem_h, mem_c, mem_t, pooled_variance = load_memantine()
    sec_h, sec_y, sec_c, sec_t = load_secukinumab()

    common = {
        "uip_scrambles": args.uip_scrambles,
        "uip_initial_power": args.uip_initial_power,
        "uip_convergence_power": args.uip_convergence_power,
        "pp_power": args.pp_power,
    }
    full = analyze_memantine(
        mem_h,
        mem_c,
        mem_t,
        delta_clin=PRIMARY_MEMANTINE_MARGIN,
        margin_label="primary_external_NPI_MCID",
        **common,
    )
    full += analyze_secukinumab(
        sec_h,
        sec_y,
        sec_c,
        sec_t,
        delta_clin=math.log(PRIMARY_SECUKINUMAB_OR),
        margin_label="primary_prespecified_OR_1.5_assumption",
        theta_order=args.theta_order,
        **common,
    )

    loo = []
    for index, source in enumerate(mem_h):
        keep = tuple(item for item in mem_h if item.source_id != source.source_id)
        loo += analyze_memantine(
            keep,
            mem_c,
            mem_t,
            delta_clin=PRIMARY_MEMANTINE_MARGIN,
            margin_label="primary_external_NPI_MCID",
            source_set="leave_one_source_out",
            omitted_source=source.source_id,
            uip_seed=202609187910 + index,
            **common,
        )
    for index, source in enumerate(sec_h):
        keep = tuple(item for item in sec_h if item.source_id != source.source_id)
        keep_events = tuple(
            value for position, value in enumerate(sec_y) if position != index
        )
        loo += analyze_secukinumab(
            keep,
            keep_events,
            sec_c,
            sec_t,
            delta_clin=math.log(PRIMARY_SECUKINUMAB_OR),
            margin_label="primary_prespecified_OR_1.5_assumption",
            source_set="leave_one_source_out",
            omitted_source=source.source_id,
            uip_seed=202609187920 + index,
            theta_order=args.theta_order,
            **common,
        )

    sensitivity: list[dict[str, object]] = []
    full_by_application = {
        application: [row for row in full if row.application == application]
        for application in ("memantine_npi", "secukinumab_asas20")
    }
    for index, margin in enumerate(MEMANTINE_MARGINS):
        if margin == PRIMARY_MEMANTINE_MARGIN:
            varied_ruip = next(
                row
                for row in full_by_application["memantine_npi"]
                if row.method == "rUIP"
            )
        else:
            varied_ruip = next(
                row
                for row in analyze_memantine(
                    mem_h,
                    mem_c,
                    mem_t,
                    delta_clin=margin,
                    margin_label=f"NPI_{margin:g}_point_sensitivity",
                    uip_seed=202609187930 + index,
                    **common,
                )
                if row.method == "rUIP"
            )
        rows = [
            varied_ruip if row.method == "rUIP" else row
            for row in full_by_application["memantine_npi"]
        ]
        for row in rows:
            payload = row.row()
            payload["sensitivity_margin"] = margin
            payload["sensitivity_margin_scale"] = "NPI points"
            sensitivity.append(payload)
    for index, odds_ratio in enumerate(SECUKINUMAB_OR_MARGINS):
        margin = math.log(odds_ratio)
        if odds_ratio == PRIMARY_SECUKINUMAB_OR:
            varied_ruip = next(
                row
                for row in full_by_application["secukinumab_asas20"]
                if row.method == "rUIP"
            )
        else:
            varied_ruip = next(
                row
                for row in analyze_secukinumab(
                    sec_h,
                    sec_y,
                    sec_c,
                    sec_t,
                    delta_clin=margin,
                    margin_label=f"OR_{odds_ratio:g}_sensitivity_assumption",
                    uip_seed=202609187940 + index,
                    theta_order=args.theta_order,
                    **common,
                )
                if row.method == "rUIP"
            )
        rows = [
            varied_ruip if row.method == "rUIP" else row
            for row in full_by_application["secukinumab_asas20"]
        ]
        for row in rows:
            payload = row.row()
            payload["sensitivity_margin"] = odds_ratio
            payload["sensitivity_margin_scale"] = "odds ratio"
            sensitivity.append(payload)

    validate_rows(full)
    validate_rows(loo)
    sensitivity_objects = [
        replace(
            next(
                row
                for row in full
                if row.application == item["application"]
                and row.method == item["method"]
            ),
            estimate=float(item["estimate"]),
            posterior_sd=float(item["posterior_sd"]),
            interval_lower=float(item["interval_lower"]),
            interval_upper=float(item["interval_upper"]),
            favorable_probability=float(item["favorable_probability"]),
        )
        for item in sensitivity
    ]
    validate_rows(sensitivity_objects)

    full_path = output / "full_source.csv"
    loo_path = output / "leave_one_source_out.csv"
    sensitivity_path = output / "margin_sensitivity.csv"
    margin_path = output / "margin_specification.json"
    _publish_csv(full_path, [row.row() for row in full])
    _publish_csv(loo_path, [row.row() for row in loo])
    _publish_csv(sensitivity_path, sensitivity)
    margin_path.write_text(
        json.dumps(_margin_specification(), indent=2) + "\n", encoding="utf-8"
    )

    hashes_after = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (MEMANTINE_INPUT, SECUKINUMAB_INPUT)
    }
    expected = {"full_source": 12, "leave_one_source_out": 78, "sensitivity": 36}
    observed = {
        "full_source": len(full),
        "leave_one_source_out": len(loo),
        "sensitivity": len(sensitivity),
    }
    full_frame = pd.DataFrame([row.row() for row in full])
    loo_frame = pd.DataFrame([row.row() for row in loo])
    sensitivity_frame = pd.DataFrame(sensitivity)
    expected_full_keys = {
        (application, method)
        for application in ("memantine_npi", "secukinumab_asas20")
        for method in METHODS
    }
    expected_loo_keys = {
        (application, source.source_id, method)
        for application, histories in (
            ("memantine_npi", mem_h),
            ("secukinumab_asas20", sec_h),
        )
        for source in histories
        for method in METHODS
    }
    expected_sensitivity_keys = {
        (application, margin, method)
        for application, margins in (
            ("memantine_npi", MEMANTINE_MARGINS),
            ("secukinumab_asas20", SECUKINUMAB_OR_MARGINS),
        )
        for margin in margins
        for method in METHODS
    }
    full_keys = set(zip(full_frame.application, full_frame.method, strict=True))
    loo_keys = set(
        zip(
            loo_frame.application,
            loo_frame.omitted_source,
            loo_frame.method,
            strict=True,
        )
    )
    sensitivity_keys = set(
        zip(
            sensitivity_frame.application,
            sensitivity_frame.sensitivity_margin,
            sensitivity_frame.method,
            strict=True,
        )
    )
    identity_checks = {
        "full_unique_complete": bool(
            len(full_frame) == len(full_keys) and full_keys == expected_full_keys
        ),
        "leave_one_source_out_unique_complete": bool(
            len(loo_frame) == len(loo_keys) and loo_keys == expected_loo_keys
        ),
        "sensitivity_unique_complete": bool(
            len(sensitivity_frame) == len(sensitivity_keys)
            and sensitivity_keys == expected_sensitivity_keys
        ),
        "all_status_success": bool(
            full_frame.status.eq("success").all()
            and loo_frame.status.eq("success").all()
            and sensitivity_frame.status.eq("success").all()
        ),
    }
    method_sets = {
        application: sorted(
            row.method for row in full if row.application == application
        )
        for application in ("memantine_npi", "secukinumab_asas20")
    }
    diagnostics = [
        json.loads(row.diagnostics_json)
        for row in [*full, *loo]
        if row.diagnostics_json
    ]
    qmc_level_changes = [
        float(item["max_level_change"])
        for item in diagnostics
        if "max_level_change" in item
    ]
    qmc_standard_errors = []
    for item in diagnostics:
        if "levels" not in item or "convergence_power" not in item:
            continue
        level = item["levels"][str(item["convergence_power"])]
        qmc_standard_errors.extend(
            float(metric["qmc_se"])
            for metric in level.values()
            if metric["qmc_se"] is not None
        )
    bound_changes = [
        float(item["bound_check_max_abs"])
        for item in diagnostics
        if "bound_check_max_abs" in item
    ]
    numerical_checks = {
        "maximum_uip_level_change": max(qmc_level_changes, default=0.0),
        "maximum_uip_qmc_standard_error": max(qmc_standard_errors, default=0.0),
        "maximum_binary_bound_change": max(bound_changes, default=0.0),
        "uip_level_change_limit": 0.005,
        "uip_qmc_standard_error_limit": 0.002,
        "binary_bound_change_limit": 1e-5,
    }
    numerical_checks["passed"] = (
        numerical_checks["maximum_uip_level_change"]
        <= numerical_checks["uip_level_change_limit"]
        and numerical_checks["maximum_uip_qmc_standard_error"]
        <= numerical_checks["uip_qmc_standard_error_limit"]
        and numerical_checks["maximum_binary_bound_change"]
        <= numerical_checks["binary_bound_change_limit"]
    )
    passed = (
        hashes_before == hashes_after
        and expected == observed
        and all(sorted(METHODS) == values for values in method_sets.values())
        and all(identity_checks.values())
        and numerical_checks["passed"]
    )
    manifest = {
        "status": "PASS" if passed else "FAIL",
        "generated_utc": datetime.now(UTC).isoformat(),
        "analysis_identity": "authoritative scalar design-stage rUIP",
        "legacy_mixture_used": False,
        "current_outcomes_used_in_prior_construction": False,
        "published_arm_level_inputs_only": True,
        "methods": list(METHODS),
        "method_availability": {method: "available" for method in METHODS},
        "row_counts_expected": expected,
        "row_counts_observed": observed,
        "method_sets": method_sets,
        "identity_checks": identity_checks,
        "numerical_checks": numerical_checks,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "inputs_unchanged": hashes_before == hashes_after,
        "pooled_current_per_observation_variance": pooled_variance,
        "pooled_current_design_sd": math.sqrt(pooled_variance),
        "margins": _margin_specification(),
        "numerical_settings": {
            "pp_qmc_power": args.pp_power,
            "uip": {
                "scrambles": args.uip_scrambles,
                "initial_power": args.uip_initial_power,
                "convergence_power": args.uip_convergence_power,
            },
            "binary_theta_order": args.theta_order,
            "binary_theta_bound": 24.0,
            "rmap_tau_nodes": 256,
            "cp_kappa_nodes": 81,
        },
        "code": {
            "runner": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": sha256(Path(__file__)),
            "ruip_core": "src/ruip/priors/ruip.py",
            "ruip_core_sha256": sha256(ROOT / "src/ruip/priors/ruip.py"),
            "clinical_interface": "src/ruip/realdata/design_stage.py",
            "clinical_interface_sha256": sha256(
                ROOT / "src/ruip/realdata/design_stage.py"
            ),
            "comparator_interface": "src/ruip/analysis/scalable_comparators.py",
            "comparator_interface_sha256": sha256(
                ROOT / "src/ruip/analysis/scalable_comparators.py"
            ),
            "canonical_cp": "src/ruip/priors/commensurate.py",
            "canonical_cp_sha256": sha256(ROOT / "src/ruip/priors/commensurate.py"),
            "git_head": git_head(),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "files": {},
    }
    for path in (full_path, loo_path, sensitivity_path, margin_path):
        manifest["files"][path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
