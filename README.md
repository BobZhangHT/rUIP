# rUIP reproduction code

Source code for the robust unit information prior (rUIP) experiments. This public package contains Python and C source, configuration, tests, and aggregate clinical inputs encoded in `src/ruip/realdata/input_summaries.py`. It contains no manuscript, review material, generated results, tables, or figures. All outputs are created locally in ignored directories.

## Setup (Windows)

Python 3.12, CMake 3.20+, and a C17 compiler are required. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e .
cmake -S . -B build
cmake --build build --config Release --target ruip_validation_shared
New-Item -ItemType Directory -Force csrc/frozen_engine_v5 | Out-Null
Copy-Item build/csrc/Release/ruip_validation.dll csrc/frozen_engine_v5/libruip_validation.dll
```

Some generators place the DLL at `build/csrc/ruip_validation.dll`; use that path for the final `Copy-Item` command if needed. The library is built from included C source and is not uploaded. `pytest -q` runs the focused Python checks. Clinical scripts create their published aggregate input CSVs locally on first use.

## Analyses

Each script has `--help` for options and output paths. The key entry points are:

| Script | Analysis |
|---|---|
| `run_outcome_simulations.py` | Six-scenario continuous, binary, and survival comparison |
| `run_joint_conflict.py` | Formal joint-conflict comparison |
| `run_precision_conflict.py` | High-precision source-conflict experiment |
| `run_component_ablation.py` | Local and global component ablation |
| `run_transport_sensitivity.py` | Transport-scale sensitivity |
| `run_clinical_applications.py` | Memantine and secukinumab applications |
| `run_oncology_application.py` | Oncology application and leave-one-history-out evaluation |
| `verify_joint_conflict.py` | Formal output checks |
| `make_figures.py` | Figures generated from local experiment results |

For example:

```powershell
python scripts/run_outcome_simulations.py --outcomes continuous --replications 1000 --output-dir results/tri_outcome_continuous
python scripts/run_precision_conflict.py --outcome continuous --replications 1000 --output-dir results/precision_leverage_continuous
python scripts/run_clinical_applications.py --output results/realdata
python scripts/run_oncology_application.py --output results/oncology
```

The three-outcome experiments use common generated datasets across methods within each replication. Prior construction uses historical summaries and prespecified settings, not current outcomes. Output CSVs, generated figures, manifests, and compiled libraries are local artifacts and are ignored by Git.

## Code structure

- `src/ruip/priors/`: rUIP and benchmark prior implementations.
- `src/ruip/outcomes/`: outcome likelihoods and posterior calculations.
- `src/ruip/realdata/`: application logic and transcribed aggregate inputs.
- `csrc/`: C17 comparator and simulation kernels.
- `scripts/`: one entry point per analysis or verification task, two design JSON files, and one consolidated figure generator.
- `tests/`: seven focused test files covering the rUIP rule, comparators, outcomes, and clinical input mapping.

The aggregate clinical inputs are transcriptions of published summaries; see `input_summaries.py` and the oncology provenance record materialized by that module. They are illustrative aggregate analyses and do not reconstruct individual participant data.
