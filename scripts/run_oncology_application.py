"""Run the prespecified oncology Gaussian-summary application."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ruip.realdata import (  # noqa: E402
    GaussianSummary,
    analyze_gaussian_summary,
    validate_gaussian_rows,
)
from ruip.realdata.input_summaries import materialize_inputs  # noqa: E402

INPUT = ROOT / "data" / "processed" / "clinical_oncology_hazard.csv"
PROVENANCE = ROOT / "data" / "processed" / "clinical_oncology_hazard.provenance.json"
DEFAULT_OUTPUT = ROOT / "results" / "realdata_oncology"
DELTA_GRID = (math.log(1.25), math.log(1.5), math.log(2.0))
PRIMARY_DELTA = math.log(1.5)
METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP")
LOO_METHODS = METHODS[1:]


def historical_information_cap(histories) -> int:
    """Return a training-history-only UIP information cap."""
    values = [int(history.n_k) for history in histories]
    if not values:
        raise ValueError("at least one training history is required")
    return max(1, round(float(np.median(values))))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path: Path = INPUT):
    """Validate the nine historical and one current event/exposure summaries."""
    # A fresh code checkout has no CSV; transcribed summaries live in source.
    materialize_inputs(ROOT)
    frame = pd.read_csv(path)
    required = {"role", "study", "events", "exposure"}
    if set(frame.columns) != required or len(frame) != 10:
        raise ValueError("oncology input must have the specified ten arm-level rows")
    if (
        frame.role.tolist().count("current") != 1
        or frame.role.tolist().count("historical") != 9
    ):
        raise ValueError(
            "oncology input must contain nine historical and one current row"
        )
    summaries = tuple(
        GaussianSummary(str(x.study), int(x.events), float(x.exposure))
        for x in frame.itertuples(index=False)
    )
    if any(x.events < 1 or x.exposure <= 0 for x in summaries):
        raise ValueError("events must be positive and exposure must be positive")
    current = next(
        x for x, role in zip(summaries, frame.role, strict=True) if role == "current"
    )
    histories = tuple(
        x.historical()
        for x, role in zip(summaries, frame.role, strict=True)
        if role == "historical"
    )
    return frame, summaries, histories, current


def publish_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def _rows(results, **extra):
    return [{**extra, **result.row()} for result in results]


def main() -> None:
    """Materialize the published ten-study summaries and evaluate the fixed
    clinical analysis plus leave-one-history-out prediction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uip-scrambles", type=int, default=4)
    parser.add_argument("--uip-power", type=int, default=13)
    parser.add_argument("--pp-power", type=int, default=14)
    args = parser.parse_args()
    materialize_inputs(ROOT)
    if args.uip_scrambles < 1 or args.uip_power < 1:
        raise ValueError("UIP scrambles and power must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    input_hash_before = sha256(INPUT)
    provenance_hash_before = sha256(PROVENANCE)
    frame, all_studies, histories, current = load_data()

    posterior, ruip = analyze_gaussian_summary(
        histories,
        current,
        delta_clin=PRIMARY_DELTA,
        planned_events=historical_information_cap(histories),
        uip_scrambles=args.uip_scrambles,
        uip_power=args.uip_power,
        pp_power=args.pp_power,
    )
    validate_gaussian_rows(posterior)
    posterior_rows = _rows(posterior, application="oncology_hazard", source_set="full")

    loo_rows: list[dict[str, object]] = []
    for index, held_out in enumerate(all_studies):
        keep = tuple(
            item.historical() for pos, item in enumerate(all_studies) if pos != index
        )
        results, _ = analyze_gaussian_summary(
            keep,
            held_out,
            delta_clin=PRIMARY_DELTA,
            planned_events=historical_information_cap(keep),
            uip_scrambles=args.uip_scrambles,
            uip_power=args.uip_power,
            uip_seed=202609188401 + index,
            pp_power=args.pp_power,
        )
        selected = [x for x in results if x.method in LOO_METHODS]
        validate_gaussian_rows(selected)
        loo_rows += _rows(
            selected, held_out=held_out.source_id, source_set="leave_one_trial_out"
        )

    sensitivity_rows: list[dict[str, object]] = []
    for index, delta in enumerate(DELTA_GRID):
        results, _ = analyze_gaussian_summary(
            histories,
            current,
            delta_clin=delta,
            planned_events=historical_information_cap(histories),
            uip_scrambles=args.uip_scrambles,
            uip_power=args.uip_power,
            uip_seed=202609188501 + index,
            pp_power=args.pp_power,
        )
        result = next(x for x in results if x.method == "rUIP")
        sensitivity_rows += _rows([result], delta_clin=delta)

    source_weight_rows = []
    for history, weight in zip(ruip.histories, ruip.weights, strict=True):
        source_weight_rows.append(
            {
                "source_id": history.source_id,
                "weight": weight,
                "historical_log_hazard": history.h_k,
                "events": history.n_k,
                "prior_mean": ruip.mean,
                "prior_precision": ruip.precision,
                "local_variance": ruip.local_variance,
                "global_discount": ruip.global_discount,
                "effective_information_units": ruip.effective_information_units,
                "delta_clin": PRIMARY_DELTA,
            }
        )

    paths = {
        "posterior.csv": output / "posterior.csv",
        "loo_predictive.csv": output / "loo_predictive.csv",
        "source_weights.csv": output / "source_weights.csv",
        "sensitivity.csv": output / "sensitivity.csv",
    }
    publish_csv(paths["posterior.csv"], posterior_rows)
    publish_csv(paths["loo_predictive.csv"], loo_rows)
    publish_csv(paths["source_weights.csv"], source_weight_rows)
    publish_csv(paths["sensitivity.csv"], sensitivity_rows)

    expected = {
        "posterior": 6,
        "loo_predictive": 50,
        "source_weights": 9,
        "sensitivity": 3,
    }
    observed = {
        "posterior": len(posterior_rows),
        "loo_predictive": len(loo_rows),
        "source_weights": len(source_weight_rows),
        "sensitivity": len(sensitivity_rows),
    }
    posterior_keys = {(x["source_set"], x["method"]) for x in posterior_rows}
    loo_keys = {(x["held_out"], x["method"]) for x in loo_rows}
    sensitivity_keys = {x["delta_clin"] for x in sensitivity_rows}
    checks = {
        "row_counts": expected == observed,
        "posterior_complete": posterior_keys == {("full", x) for x in METHODS},
        "loo_complete": loo_keys
        == {(x.source_id, method) for x in all_studies for method in LOO_METHODS},
        "loo_scores_finite": all(
            math.isfinite(float(x["prior_predictive_log_density"])) for x in loo_rows
        ),
        "sensitivity_complete": sensitivity_keys == set(DELTA_GRID),
        "all_status_success": all(
            x["status"] == "success"
            for x in [*posterior_rows, *loo_rows, *sensitivity_rows]
        ),
        "weights_sum_to_one": math.isclose(
            sum(x["weight"] for x in source_weight_rows), 1.0, abs_tol=1e-12
        ),
        "inputs_unchanged": input_hash_before == sha256(INPUT),
        "provenance_unchanged": provenance_hash_before == sha256(PROVENANCE),
    }
    manifest = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_utc": datetime.now(UTC).isoformat(),
        "analysis_identity": "oncology exponential-model Gaussian log-hazard summary",
        "current_outcomes_used_in_prior_construction": False,
        "prior_construction": (
            "Historical log(events/exposure), variance 1/events; current outcome "
            "enters only Normal likelihood."
        ),
        "uip_information_cap_rule": (
            "Rounded median event count among the training historical trials; "
            "recomputed without the held-out trial in each predictive fold."
        ),
        "primary_delta_clin": PRIMARY_DELTA,
        "sensitivity_delta_clin": list(DELTA_GRID),
        "numerical_settings": {
            "uip_scrambles": args.uip_scrambles,
            "uip_power": args.uip_power,
            "pp_power": args.pp_power,
            "uip_draws_per_scramble": 2**args.uip_power,
        },
        "row_counts_expected": expected,
        "row_counts_observed": observed,
        "identity_checks": checks,
        "input_hash_before": input_hash_before,
        "input_hash_after": sha256(INPUT),
        "provenance_file": str(PROVENANCE.relative_to(ROOT)),
        "provenance_hash_before": provenance_hash_before,
        "provenance_hash_after": sha256(PROVENANCE),
        "input_rows": frame.to_dict(orient="records"),
        "code": {
            "runner": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": sha256(Path(__file__)),
            "gaussian_interface": "src/ruip/realdata/gaussian_summary.py",
            "gaussian_interface_sha256": sha256(
                ROOT / "src" / "ruip" / "realdata" / "gaussian_summary.py"
            ),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
        },
        "files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    raise SystemExit(0 if manifest["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
