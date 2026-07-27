# UIP-IC

Reproducible CPU proof-of-concept for Bayesian data augmentation with a unit information prior (UIP) in interval-censored proportional-hazards survival data.

The repository compares:

- `IC-NIP`: diffuse Normal prior without historical borrowing;
- `IC-UIP`: Normal UIP from historical Cox summary statistics, dynamically updated study weights, and adaptive continuous borrowing amount `M`;
- `IC-CP`: a summary-level commensurate-prior comparator motivated by the historical-borrowing literature.

This is a method sanity check, not a publication-level operating-characteristic study. It uses 20 paired repetitions per scenario and short single chains.

## What is implemented

- Piecewise-exponential PH data generation with treatment and one Normal covariate.
- Conversion of exact event times into general intervals `(L, R]`, including left and right censoring induced by scheduled inspections.
- Historical exact/right-censored Cox fits yielding only `theta_hat`, `SE`, and `n`; patient-level historical data are not passed to the current Bayesian analysis.
- Direct inverse sampling of finite latent failure times inside `(L, R]`.
- Gamma-conjugate baseline hazard updates.
- Stable one-dimensional stepping-out slice updates for treatment and covariate effects.
- Direct truncated-Gamma update for adaptive `M`.
- Dirichlet prior and additive-log-ratio slice updates for historical-study weights.
- Bias, RMSE, 95% CrI coverage and width, borrowing, equivalent ESS, scalar MCMC ESS, lag-1 autocorrelation, timing, figures, logs, and automated audit.
- Common random numbers across scenarios so that only historical treatment-effect conflict changes within each repetition.

The PH model sets frailty to one. Proportional odds and general transformation models are intentionally out of scope for this first implementation.

## Observed small-experiment result

The checked-in results were generated with seed `20260727`, `n=80`, two historical studies of `n_k=100`, 20 paired repetitions per scenario, and 800 iterations with 400 burn-in.

| Scenario | Method | RMSE | Coverage | Mean CrI width | Mean M |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 compatible | IC-NIP | 0.276 | 1.00 | 1.212 | 0.0 |
| S1 compatible | IC-UIP | 0.192 | 1.00 | 1.000 | 44.2 |
| S2 mixed | IC-UIP | 0.174 | 1.00 | 1.007 | 41.7 |
| S3 severe conflict | IC-UIP | 0.252 | 1.00 | 1.187 | 17.1 |
| S3 severe conflict | IC-CP | 0.244 | 1.00 | 1.219 | NA |

The IC-UIP posterior mean `M` fell by 61.4% from S1 to S3 and decreased in all 20 paired repetitions. In S2 its mean posterior weights were 0.572 for the compatible study and 0.428 for the conflicting study. All 180 fits completed; the minimum treatment-effect ESS was 54.0, the largest absolute lag-1 autocorrelation was 0.67, and the automated audit passed. With only 20 repetitions, coverage estimates remain coarse and should be interpreted as a method sanity check.

See the full [experiment report](reports/small_experiment_report.md), [summary CSV](results/small_simulation_summary.csv), and [audit](results/audit.txt).

## Windows PowerShell quick start

```powershell
Set-Location "$env:USERPROFILE\Desktop\UIP-IC"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

Run a single demonstration:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --scenario S2_mixed
```

Reproduce the checked-in small experiment (about two minutes on the development CPU):

```powershell
.\.venv\Scripts\python.exe scripts\run_small_simulation.py
```

For a faster development run:

```powershell
.\.venv\Scripts\python.exe scripts\run_small_simulation.py --repetitions 2 --iterations 300 --burn-in 150
```

The simulation command overwrites generated CSV, figures, logs, audit, and report with outputs from the requested configuration.

## Repository layout

```text
configs/small_experiment.yaml       Fixed experiment and MCMC settings
src/uip_ic/                         Data, PH, DA, UIP, sampler, and evaluation modules
scripts/run_demo.py                 One-dataset reproducible comparison
scripts/run_small_simulation.py     Repeated experiment, plots, audit, and report
tests/                              Numerical, boundary, borrowing, and reproducibility tests
results/                            Checked-in generated outputs and diagnostics
reports/small_experiment_report.md  Model, conditionals, results, interpretation, limitations
papers/SOURCES.md                   Verified local source metadata and use statement
```

## UIP used in this repository

For historical summaries `(theta_hat_k, SE_k, n_k)` and dynamically sampled weights `w_k`,

```text
I_Uk = 1 / (n_k SE_k^2)
mu_w = sum_k w_k theta_hat_k
I_w  = sum_k w_k I_Uk
theta | M, H ~ Normal(mu_w, 1 / (M I_w))
M ~ Uniform(0, M_max)
w ~ Dirichlet(gamma)
```

Consequently, `M | theta, H` is a shape-`3/2` Gamma distribution with rate `0.5 I_w (theta - mu_w)^2`, truncated to `(0, M_max)`. This update drives borrowing down when the current treatment effect conflicts with the historical weighted mean.

The reported equivalent ESS is diagnostic only. It divides posterior mean prior precision by an NIP-posterior approximation to current per-subject information; it is not yet the prospective IC-design ESS calibration proposed for future work.

## References and integrity

The two user-provided PDFs were checked locally. Their titles, DOI, hashes, and exact roles are documented in [papers/SOURCES.md](papers/SOURCES.md). The broader comparison is documented in [the literature search report](literature-search-20260727-interval-censored-borrowing/papers.md). PDF full text is ignored by Git and is not redistributed.

No result is hard-coded into the experiment scripts. Tables, figures, diagnostics, audit messages, and the report are generated from the saved run outputs. Users should review the code and numerical claims before using this work in a manuscript, and follow the journal or institution's current AI-disclosure policy.
