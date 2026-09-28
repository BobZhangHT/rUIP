"""Verify current joint-conflict outputs and their paired simulation design.

This verifier is intentionally separate from the locked simulation runner.  It
checks provenance, file digests, the complete paired key space, and the
random-stream mapping. The historical unfiltered execution identity remains
in the external audit archive; current outputs explicitly identify derivation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "joint_conflict_design.json"
RUNNER = ROOT / "scripts" / "run_joint_conflict.py"
AUTHORITY = ROOT / "scripts" / "run_outcome_simulations.py"
DLL = ROOT / "csrc" / "frozen_engine_v5" / "libruip_validation.dll"
SURVIVAL = ROOT / "src" / "ruip" / "outcomes" / "survival.py"
ABI = ROOT / "src" / "ruip" / "c_abi.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("joint_formal_locked", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load locked runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_manifest_files(directory: Path, manifest: dict) -> None:
    for name, record in manifest["files"].items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing declared file: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"byte-count mismatch: {path}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")


def verify_one(directory: Path, config: dict, joint) -> dict:
    """Validate one outcome run against its manifest and protocol key space."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest_files(directory, manifest)
    if manifest["status"] != "PASS":
        raise ValueError(f"numerical manifest is not PASS: {directory}")
    if manifest["sentinel"] or manifest["seed"] != config["formal_seed"]:
        raise ValueError(f"not a locked formal stream: {directory}")
    if manifest["replications"] != config["replications"]:
        raise ValueError(f"replication mismatch: {directory}")
    if len(manifest["outcomes"]) != 1:
        raise ValueError(f"each input must contain exactly one outcome: {directory}")
    outcome = manifest["outcomes"][0]
    if manifest["config_sha256"] != sha256(CONFIG):
        raise ValueError(f"config provenance mismatch: {directory}")
    if manifest["runner_sha256"] != sha256(RUNNER):
        raise ValueError(f"runner provenance mismatch: {directory}")
    if manifest["authority_runner_sha256"] != sha256(AUTHORITY):
        raise ValueError(f"authority provenance mismatch: {directory}")
    if manifest["frozen_comparator_sha256"] != sha256(DLL):
        raise ValueError(f"comparator provenance mismatch: {directory}")
    if manifest.get("provenance_kind") != "derived_filter_and_reanalysis":
        raise ValueError(f"missing explicit derivation provenance: {directory}")
    if manifest.get("derivation", {}).get("removed_method_count") != 1:
        raise ValueError(f"unexpected derivation: {directory}")

    raw = pd.read_csv(directory / "replicate_results.csv.gz")
    scenarios = joint.scenario_grid(config, outcome)
    scenario_ids = [row["scenario_id"] for row in scenarios]
    methods = list(config["methods"])
    reps = int(config["replications"])
    expected_rows = len(scenario_ids) * reps * len(methods)
    if len(raw) != expected_rows or int(manifest["analyses"]) != expected_rows:
        raise ValueError(f"row-count mismatch: {directory}")
    if set(raw["outcome"]) != {outcome}:
        raise ValueError(f"outcome-column mismatch: {directory}")
    if set(raw["scenario_id"]) != set(scenario_ids):
        raise ValueError(f"scenario set mismatch: {directory}")
    if set(raw["method"]) != set(methods):
        raise ValueError(f"method set mismatch: {directory}")
    if set(raw["replicate_id"]) != set(range(reps)):
        raise ValueError(f"replicate set mismatch: {directory}")
    keys = ["outcome", "scenario_id", "replicate_id", "method"]
    if raw.duplicated(keys).any() or raw.groupby(keys).size().ne(1).any():
        raise ValueError(f"duplicate paired key: {directory}")
    dataset_keys = ["outcome", "scenario_id", "replicate_id"]
    counts = raw.groupby(dataset_keys, sort=False).size()
    if not counts.eq(len(methods)).all():
        raise ValueError(f"incomplete method pairing: {directory}")
    if raw.groupby(dataset_keys, sort=False)["data_seed"].nunique().ne(1).any():
        raise ValueError(f"method-specific data seed detected: {directory}")

    oi = list(config["outcomes"]).index(outcome)
    observed = raw.groupby(dataset_keys, sort=False)["data_seed"].first()
    for si, scenario_id in enumerate(scenario_ids):
        for replicate in range(reps):
            expected_seed = int(
                np.random.SeedSequence(
                    config["formal_seed"], spawn_key=(oi, si, replicate)
                ).generate_state(1, dtype=np.uint64)[0]
            )
            actual_seed = int(observed.loc[(outcome, scenario_id, replicate)])
            if actual_seed != expected_seed:
                raise ValueError(
                    f"seed-map mismatch: {outcome}/{scenario_id}/{replicate}"
                )
    return {
        "outcome": outcome,
        "rows": len(raw),
        "datasets": len(scenario_ids) * reps,
        "manifest_sha256": sha256(manifest_path),
        "replicate_results_sha256": sha256(directory / "replicate_results.csv.gz"),
    }


def verify_combined(combined: Path, config: dict) -> dict:
    manifest = json.loads((combined / "manifest.json").read_text(encoding="utf-8"))
    verify_manifest_files(combined, manifest)
    report = json.loads(
        (combined / "confirmation_gate_report.json").read_text(encoding="utf-8")
    )
    numerical_pass = manifest["status"] == "PASS"
    current_gates_pass = bool(report["all_current_gates_pass"])
    headline_pass = bool(
        report["gates"].get("headline_mse_reduction_at_least_7_5_percent", False)
    )
    return {
        "numerical_manifest_pass": numerical_pass,
        "all_current_gates_pass": current_gates_pass,
        "headline_7_5_percent_gate_pass": headline_pass,
        "current_reanalysis_pass": numerical_pass and current_gates_pass,
    }


def main() -> None:
    """Check expected rows, scenario keys, paired seeds, and output digests
    without changing or regenerating analysis results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    current = {
        "scripts/joint_conflict_design.json": sha256(CONFIG),
        "scripts/run_joint_conflict.py": sha256(RUNNER),
        "scripts/run_outcome_simulations.py": sha256(AUTHORITY),
        "csrc/frozen_engine_v5/libruip_validation.dll": sha256(DLL),
    }
    identity = None
    if args.lock is not None:
        identity = json.loads(args.lock.read_text(encoding="utf-8"))
        if identity.get("artifacts") != current:
            raise ValueError("current execution-identity artifact hashes do not match")
    joint = load_runner()
    records = [
        verify_one(path.resolve(), config, joint) for path in args.input_dirs
    ]
    if {row["outcome"] for row in records} != set(config["outcomes"]):
        raise ValueError("formal inputs do not cover each frozen outcome exactly once")
    combined = verify_combined(args.combined_dir.resolve(), config)
    result = {
        "status": "PASS" if combined["current_reanalysis_pass"] else "FAIL",
        "meaning": (
            "PASS applies to the current five-classical-comparator reanalysis: "
            "complete provenance/key/seed verification, a numerical manifest "
            "PASS, and every current primary gate. Historical acceptance is "
            "separate; its original supplementary gate failed."
        ),
        "current_artifacts": current,
        "execution_identity_type": (
            identity.get("record_type", "unspecified")
            if identity is not None
            else (
                "post-review filtered reanalysis; historical full outputs "
                "archived separately"
            )
        ),
        "execution_identity": str(args.lock.resolve()) if args.lock else None,
        "supplementary_runtime_dependencies": {
            str(SURVIVAL.relative_to(ROOT)).replace("\\", "/"): sha256(SURVIVAL),
            str(ABI.relative_to(ROOT)).replace("\\", "/"): sha256(ABI),
        },
        "outcomes": records,
        **combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
