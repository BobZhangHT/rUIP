# Generated results

Running `scripts/run_small_simulation.py` creates or overwrites:

- `small_simulation_raw.csv`: one row per scenario, repetition, and method;
- `small_simulation_summary.csv`: grouped performance metrics;
- `historical_summaries.csv`: historical truth, Cox estimate, SE, unit information, event fraction, and fixed UIP weight;
- `diagnostics.csv`: scalar ESS, lag-1 autocorrelation, timing, and any fit error;
- `audit.txt`: automated numerical and operating-characteristic warnings;
- `figures/method_comparison.png`: bias, RMSE, coverage, and CrI-width comparison;
- `figures/borrowing_adaptation.png`: posterior mean `M` and equivalent ESS by conflict scenario;
- `logs/small_experiment.log`: progress and retained failures.

`demo_*.csv` files come from `scripts/run_demo.py` and are separate from the 20-repetition experiment.

The checked-in batch used the exact settings in `configs/small_experiment.yaml`. The report at `reports/small_experiment_report.md` is generated from these files.
