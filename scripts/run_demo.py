"""Run one reproducible IC-UIP comparison and save its posterior summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from uip_ic.experiment import sampler_config_from_mapping, simulate_replicate
from uip_ic.methods import fit_method
from uip_ic.utils import derived_seed, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs" / "small_experiment.yaml")
    parser.add_argument("--scenario", default="S2_mixed")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--burn-in", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    sampler_values = dict(config["sampler"])
    if args.iterations is not None:
        sampler_values["iterations"] = args.iterations
    if args.burn_in is not None:
        sampler_values["burn_in"] = args.burn_in
    sampler = sampler_config_from_mapping(sampler_values)
    scenario_names = list(config["scenarios"])
    if args.scenario not in scenario_names:
        raise ValueError(f"unknown scenario {args.scenario!r}")
    scenario_index = scenario_names.index(args.scenario)
    inputs = simulate_replicate(config, args.scenario, scenario_index, repetition=0)

    rows = []
    for method_index, method in enumerate(config["methods"]):
        seed = derived_seed(int(config["experiment"]["seed"]), scenario_index, 0, 100 + method_index)
        result = fit_method(
            method,
            inputs.current,
            inputs.baseline.interval_starts,
            sampler,
            seed,
            None if method == "NIP-DA" else inputs.uip,
        )
        rows.append({"scenario": args.scenario, "method": method, **result.summary, "runtime_seconds": result.runtime_seconds})

    output = REPOSITORY / "results" / "demo_posterior_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    history_output = REPOSITORY / "results" / "demo_historical_summaries.csv"
    pd.DataFrame(
        [
            {
                "study": item.study,
                "theta_true": inputs.historical_truths[index],
                "theta_hat": item.theta_hat,
                "se": item.se,
                "n": item.n,
                "unit_information": item.unit_information,
            }
            for index, item in enumerate(inputs.historical_summaries)
        ]
    ).to_csv(history_output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {output}")
    print(f"Saved {history_output}")


if __name__ == "__main__":
    main()
