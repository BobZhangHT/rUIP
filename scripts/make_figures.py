"""Generate the four manuscript figures from verified experiment outputs.

Usage:
    python scripts/make_figures.py all
    python scripts/make_figures.py precision-leverage
    python scripts/make_figures.py joint-conflict
    python scripts/make_figures.py component-ablation
    python scripts/make_figures.py oncology

The precision command accepts --inputs CONTINUOUS BINARY SURVIVAL to override
the default formal confirmation result directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "manuscript" / "figures"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Precision and leverage confirmation
PRECISION_OUTCOMES = ("continuous", "binary", "survival")
PRECISION_METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP")
PRECISION_COLORS = {"continuous": "#0072B2", "binary": "#D55E00", "survival": "#009E73"}


def load_precision_data(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path, outcome in zip(paths, PRECISION_OUTCOMES, strict=True):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "PASS"
            or manifest.get("replications") != 1000
            or manifest.get("outcome") != outcome
            or manifest.get("failure_count") != 0
        ):
            raise RuntimeError(f"formal confirmation gate failed: {path}")
        frame = pd.read_csv(path / "replicate_results.csv.gz")
        if (
            len(frame) != 7 * 1000 * len(PRECISION_METHODS)
            or frame.status.ne("success").any()
        ):
            raise RuntimeError(f"row/failure gate failed: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_precision_leverage(input_dirs: list[Path]) -> None:
    """Plot error and coverage for the high-information conflict experiment."""
    raw = load_precision_data([path.resolve() for path in input_dirs])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    comparators = ("NIP", "PP", "rMAP", "CP", "UIP")
    ratio_matrix = np.empty((len(PRECISION_OUTCOMES), len(comparators)))
    coverage_matrix = np.empty((len(PRECISION_OUTCOMES), len(PRECISION_METHODS)))
    width_matrix = np.empty((len(PRECISION_OUTCOMES), len(PRECISION_METHODS)))
    for row, outcome in enumerate(PRECISION_OUTCOMES):
        joint = raw[raw.outcome.eq(outcome) & raw.scenario_id.str.startswith("joint")]
        mse = joint.groupby("method").squared_error.mean()
        coverage = joint.groupby("method").covered.mean()
        ratio_matrix[row] = [mse["rUIP"] / mse[method] for method in comparators]
        coverage_matrix[row] = [coverage[method] for method in PRECISION_METHODS]
        widths = joint.groupby("method").interval_width.mean()
        width_matrix[row] = [
            widths[method] / widths["NIP"] for method in PRECISION_METHODS
        ]

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 7.0))
    bar_width = 0.23
    offsets = (-bar_width, 0.0, bar_width)
    panels = (
        (
            axes[0],
            comparators,
            100.0 * (1.0 - ratio_matrix),
            "A. Estimation error",
            "rUIP MSE reduction (%)",
        ),
        (
            axes[1],
            PRECISION_METHODS,
            100.0 * (coverage_matrix - 0.95),
            "B. Interval calibration",
            "Coverage minus 95% (points)",
        ),
        (
            axes[2],
            PRECISION_METHODS,
            100.0 * (1.0 - width_matrix),
            "C. Interval precision",
            "Width reduction vs NIP (%)",
        ),
    )
    hatches = ("", "//", "..")
    for axis, methods, values, title, ylabel in panels:
        positions = np.arange(len(methods))
        for outcome_index, outcome in enumerate(PRECISION_OUTCOMES):
            axis.bar(
                positions + offsets[outcome_index],
                values[outcome_index],
                width=bar_width,
                color=PRECISION_COLORS[outcome],
                edgecolor="0.25",
                linewidth=0.45,
                hatch=hatches[outcome_index],
                label=outcome.capitalize(),
            )
        axis.axhline(0.0, color="0.25", lw=1)
        axis.set_xticks(positions, methods)
        for tick in axis.get_xticklabels():
            if tick.get_text() == "rUIP":
                tick.set_fontweight("bold")
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.88", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis, _, values, _, _ in panels:
        spread = max(1.0, float(np.ptp(values)))
        axis.set_ylim(
            min(0.0, float(np.min(values))) - 0.09 * spread,
            max(0.0, float(np.max(values))) + 0.13 * spread,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.0)
    output = ROOT / "manuscript" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"figure_precision_leverage_confirmation.{suffix}")
    plt.close(fig)
    generator = Path(__file__).resolve()
    input_records = []
    for directory in input_dirs:
        for name in ("manifest.json", "replicate_results.csv.gz"):
            path = directory.resolve() / name
            input_records.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    for suffix in ("pdf", "png"):
        artifact = output / f"figure_precision_leverage_confirmation.{suffix}"
        record = {
            "generator": str(generator),
            "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
            "inputs": input_records,
            "output": str(artifact.resolve()),
            "output_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        artifact.with_name(artifact.name + ".provenance.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {"status": "PASS", "figure": "figure_precision_leverage_confirmation"},
            indent=2,
        )
    )


# Joint conflict confirmation
JOINT_RESULTS = ROOT / "results" / "ruip_joint_conflict_confirmation"


def make_joint_conflict() -> None:
    """Plot the paired joint-conflict results from verified local outputs."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    report_path = JOINT_RESULTS / "confirmation_gate_report.json"
    comparisons_path = JOINT_RESULTS / "confirmation_comparisons.csv"
    scenario_path = JOINT_RESULTS / "scenario_method_summary.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparisons = pd.read_csv(comparisons_path)
    scenarios = pd.read_csv(scenario_path)

    outcomes = ["continuous", "binary", "survival"]
    labels = ["C", "B", "S"]
    ratios = np.array(
        [report["outcome_mse_ratios_vs_strongest_classic"][key] for key in outcomes]
    )
    coverage = {row["outcome"]: row for row in report["coverage"]}
    cover_points = np.array([coverage[key]["coverage"] for key in outcomes])
    cover_lower = np.array(
        [coverage[key]["bonferroni_one_sided_wilson_lower"] for key in outcomes]
    )

    display_methods = ["NIP", "PP", "rMAP", "CP", "UIP"]
    rows = comparisons.set_index("comparator").loc[display_methods]
    combined = rows["combined_mse_ratio"].to_numpy()
    upper = rows["familywise_one_sided_ratio_upper"].to_numpy()
    target = scenarios.loc[
        scenarios.region.eq("target") & scenarios.method.isin(("rUIP", "rMAP"))
    ]
    paired_cells = target.pivot(
        index=["outcome", "scenario_id"], columns="method", values="mse"
    )
    if paired_cells.isna().any().any() or any(
        len(paired_cells.loc[outcome]) != 16 for outcome in outcomes
    ):
        raise ValueError("Expected 16 complete target cells per outcome")
    wins = np.array(
        [
            int(
                (
                    paired_cells.loc[outcome, "rUIP"]
                    < paired_cells.loc[outcome, "rMAP"]
                ).sum()
            )
            for outcome in outcomes
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.45))
    ax_a, ax_b, ax_c, ax_d = axes.flat
    blue = "#2C6DB2"

    x = np.arange(3)
    ax_a.axhspan(0, 1, color="#EAF2FA", zorder=0)
    ax_a.axhline(1, color="0.35", ls="--", lw=0.9)
    ax_a.scatter(x, ratios, color=blue, s=34, zorder=3)
    ax_a.set(
        title="A. Accuracy versus rMAP",
        ylabel="MSE ratio: rUIP / rMAP",
        xticks=x,
        xticklabels=labels,
        ylim=(0.84, 1.015),
    )

    yerr = cover_points - cover_lower
    ax_b.axhspan(0.94, 0.97, color="#EAF2FA", zorder=0)
    ax_b.axhline(0.95, color="0.35", ls="--", lw=0.9)
    ax_b.errorbar(
        x,
        cover_points,
        yerr=np.vstack([yerr, np.zeros_like(yerr)]),
        fmt="o",
        color=blue,
        capsize=3,
        lw=1.1,
    )
    ax_b.set(
        title="B. Interval calibration",
        ylabel="Coverage",
        xticks=x,
        xticklabels=labels,
        ylim=(0.92, 0.975),
    )

    y = np.arange(len(display_methods))
    ax_c.axvspan(0, 1, color="#EAF2FA", zorder=0)
    ax_c.axvline(1, color="0.35", ls="--", lw=0.9)
    for pos, method, point, bound in zip(
        y, display_methods, combined, upper, strict=True
    ):
        color = "#D55E00" if method == "UIP" else blue
        ax_c.plot([point, bound], [pos, pos], color=color, lw=1.5)
        ax_c.scatter(point, pos, color=color, marker="o", s=28, zorder=3)
        ax_c.scatter(
            bound,
            pos,
            facecolor="white",
            edgecolor=color,
            marker="o",
            s=26,
            zorder=3,
        )
    ax_c.set(
        title="C. Comparison with classical priors",
        xlabel="MSE ratio: rUIP / comparator\n(<1 favors rUIP)",
        yticks=y,
        yticklabels=display_methods,
        xlim=(0.26, 1.02),
    )
    ax_c.invert_yaxis()

    ax_d.bar(x, [16] * len(x), color="#E7EBF0", width=0.6)
    ax_d.bar(x, wins, color=blue, width=0.6)
    for position, count in zip(x, wins, strict=True):
        ax_d.text(position, count + 0.35, f"{count}/16", ha="center", fontsize=9)
    ax_d.set(
        title="D. Improvement across target cells",
        ylabel="Cells with lower MSE than rMAP",
        xticks=x,
        xticklabels=labels,
        yticks=[0, 4, 8, 12, 16],
        ylim=(0, 17),
    )

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", lw=0.6)
        axis.title.set_fontsize(10)
        axis.xaxis.label.set_fontsize(9)
        axis.yaxis.label.set_fontsize(9)
        axis.tick_params(labelsize=9)
    fig.tight_layout(w_pad=1.1, h_pad=1.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "figure_joint_conflict_confirmation.pdf"
    png = FIGURES / "figure_joint_conflict_confirmation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "generator": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "generator_sha256": sha256(Path(__file__)),
        "inputs": {
            str(report_path.relative_to(ROOT)).replace("\\", "/"): sha256(report_path),
            str(comparisons_path.relative_to(ROOT)).replace("\\", "/"): sha256(
                comparisons_path
            ),
            str(scenario_path.relative_to(ROOT)).replace("\\", "/"): sha256(
                scenario_path
            ),
        },
        "outputs": {
            str(pdf.relative_to(ROOT)).replace("\\", "/"): sha256(pdf),
            str(png.relative_to(ROOT)).replace("\\", "/"): sha256(png),
        },
        "scientific_status": (
            "post-review five-method analysis of the original paired "
            "simulations"
        ),
    }
    (FIGURES / "figure_joint_conflict_confirmation.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


# Component ablation
ABLATION_INPUT = ROOT / "results" / "component_ablation_20260928"
ABLATION_OUTPUT = ROOT / "manuscript" / "figures"
ABLATION_OUTCOMES = ("continuous", "binary", "survival")
ABLATION_SCENARIOS = (
    "compatible",
    "global-negative",
    "global-positive",
    "local-negative",
    "local-positive",
    "joint-negative",
    "joint-positive",
)
ABLATION_SHORT = ("C", "G−", "G+", "L−", "L+", "J−", "J+")
ABLATION_METHODS = ("weighted-anchor", "no-score", "no-cap", "no-global")
ABLATION_TITLES = (
    "A. Precision-weighted anchor",
    "B. No compatibility score",
    "C. No information cap",
    "D. No transport variance",
)
ABLATION_COLORS = {
    "continuous": "#2763A6",
    "binary": "#C46014",
    "survival": "#22846A",
}
ABLATION_MARKERS = {"continuous": "o", "binary": "s", "survival": "^"}


def make_component_ablation() -> None:
    """Show what changes when one rUIP component is removed at a time."""
    inputs = [
        ABLATION_INPUT / o / "paired_component_summary.csv" for o in ABLATION_OUTCOMES
    ]
    coverage_inputs = [
        ABLATION_INPUT / o / "scenario_method_summary.csv" for o in ABLATION_OUTCOMES
    ]
    inputs += coverage_inputs
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("all three outcome summaries are required")
    data = pd.concat(
        [pd.read_csv(path) for path in inputs[: len(ABLATION_OUTCOMES)]],
        ignore_index=True,
    )
    coverage_data = pd.concat(
        [pd.read_csv(path) for path in coverage_inputs], ignore_index=True
    )
    expected = len(ABLATION_OUTCOMES) * len(ABLATION_SCENARIOS) * len(ABLATION_METHODS)
    if len(data) != expected:
        raise ValueError(f"expected {expected} summary rows, got {len(data)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(12.1, 6.1), sharex="col")
    x = np.arange(len(ABLATION_SCENARIOS))
    for col, (method, title) in enumerate(
        zip(ABLATION_METHODS, ABLATION_TITLES, strict=True)
    ):
        top, bottom = axes[:, col]
        for outcome in ABLATION_OUTCOMES:
            subset = (
                data[data.ablation.eq(method) & data.outcome.eq(outcome)]
                .set_index("scenario")
                .loc[list(ABLATION_SCENARIOS)]
            )
            top.plot(
                x,
                subset.mse_ratio_ablation_over_ruip,
                color=ABLATION_COLORS[outcome],
                marker=ABLATION_MARKERS[outcome],
                ms=3.7,
                lw=1.15,
                label=outcome.capitalize(),
            )
            ablation_coverage = (
                coverage_data[
                    coverage_data.method.eq(method) & coverage_data.outcome.eq(outcome)
                ]
                .set_index("scenario")
                .loc[list(ABLATION_SCENARIOS), "coverage"]
            )
            ruip_coverage = (
                coverage_data[
                    coverage_data.method.eq("rUIP") & coverage_data.outcome.eq(outcome)
                ]
                .set_index("scenario")
                .loc[list(ABLATION_SCENARIOS), "coverage"]
            )
            bottom.plot(
                x,
                ruip_coverage,
                color=ABLATION_COLORS[outcome],
                ls="--",
                lw=1.0,
                alpha=0.55,
                zorder=2,
            )
            bottom.plot(
                x,
                ablation_coverage,
                color=ABLATION_COLORS[outcome],
                marker=ABLATION_MARKERS[outcome],
                ms=3.7,
                lw=1.15,
                zorder=3,
            )
        top.axhline(1.0, color="0.45", ls="--", lw=0.8)
        bottom.axhline(0.95, color="0.25", ls=":", lw=1.0, zorder=1)
        top.set_title(title, loc="left", pad=9)
        bottom.set_xticks(x, ABLATION_SHORT)
        bottom.set_xlabel("Conflict setting")
        for ax in (top, bottom):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.88", lw=0.55)
            ax.tick_params(axis="x", pad=3)
        if col == 0:
            top.set_ylabel("Ablation/rUIP MSE ratio\n(above 1 favors rUIP)")
            bottom.set_ylabel("95% interval coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="0.3", lw=1.2, label="Ablation: solid"),
        Line2D([0], [0], color="0.3", lw=1.2, ls="--", label="rUIP: dashed"),
        Line2D([0], [0], color="0.3", lw=1.2, ls=":", label="Nominal: dotted"),
    ]
    labels += ["Ablation: solid", "rUIP: dashed", "Nominal: dotted"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=6,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.13, top=0.88, wspace=0.30, hspace=0.35
    )
    ABLATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = ABLATION_OUTPUT / "figure_component_ablation"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    provenance = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__)),
        "role": "exploratory supplementary component ablation",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (stem.with_suffix(".pdf"), stem.with_suffix(".png"))
        },
    }
    (ABLATION_OUTPUT / "figure_component_ablation.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(stem.with_suffix(".pdf"))


# Oncology application
ONCOLOGY_RESULTS = ROOT / "results" / "realdata_oncology_capped_medoid"
ONCOLOGY_DATA = ROOT / "data" / "processed" / "clinical_oncology_hazard.csv"
ONCOLOGY_METHODS = ("NIP", "PP", "rMAP", "CP", "UIP", "rUIP")
ONCOLOGY_BORROWING = ONCOLOGY_METHODS[1:]
ONCOLOGY_COLORS = {
    "NIP": "#777777",
    "PP": "#D55E00",
    "rMAP": "#009E73",
    "CP": "#CC79A7",
    "UIP": "#E69F00",
    "rUIP": "#0072B2",
}


def verify_oncology_inputs() -> dict:
    manifest_path = ONCOLOGY_RESULTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not all(
        manifest.get("identity_checks", {}).values()
    ):
        raise RuntimeError("oncology manifest did not pass")
    if sha256(ONCOLOGY_DATA) != manifest["input_hash_after"]:
        raise RuntimeError("oncology input hash mismatch")
    for name, metadata in manifest["files"].items():
        path = ONCOLOGY_RESULTS / name
        if (
            not path.is_file()
            or sha256(path) != metadata["sha256"]
            or path.stat().st_size != metadata["bytes"]
        ):
            raise RuntimeError(f"oncology result mismatch: {path}")
    for name in ("runner", "gaussian_interface"):
        path = ROOT / manifest["code"][name]
        if sha256(path) != manifest["code"][f"{name}_sha256"]:
            raise RuntimeError(f"oncology code mismatch: {path}")
    return manifest


def set_oncology_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.8,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def make_oncology_application() -> None:
    """Display historical source weights and leave-one-source-out prediction."""
    manifest = verify_oncology_inputs()
    set_oncology_style()
    data = pd.read_csv(ONCOLOGY_DATA)
    posterior = pd.read_csv(ONCOLOGY_RESULTS / "posterior.csv").set_index("method")
    weights = pd.read_csv(ONCOLOGY_RESULTS / "source_weights.csv")
    loo = pd.read_csv(ONCOLOGY_RESULTS / "loo_predictive.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 6.75))

    # Panel A: observed rate summaries. The fifth history is the local outlier.
    ax = axes[0, 0]
    estimate = np.log(data.events / data.exposure)
    se = 1.0 / np.sqrt(data.events)
    rate = np.exp(estimate)
    lower, upper = np.exp(estimate - 1.96 * se), np.exp(estimate + 1.96 * se)
    labels = [f"H{i}" if i != 5 else "H5 (outlier)" for i in range(1, 10)] + ["Current"]
    y = np.arange(len(data))[::-1]
    colors = ["#D55E00" if i == 4 else "#777777" for i in range(9)] + ["#0072B2"]
    for yi, x, lo, hi, color in zip(y, rate, lower, upper, colors, strict=True):
        ax.errorbar(
            x,
            yi,
            xerr=[[x - lo], [hi - x]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            lw=0.9,
            ms=4,
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Observed hazard rate per person-year (95% interval)")
    ax.set_title("A  Observed trial rates (descriptive)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)

    # Panel B: design-stage source reliability weights.
    ax = axes[0, 1]
    wy = np.arange(len(weights))[::-1]
    wcolors = ["#D55E00" if i == 4 else "#0072B2" for i in range(len(weights))]
    ax.barh(wy, weights.weight, color=wcolors, height=0.65)
    # The pooled CP center uses these unadjusted information shares. Displaying
    # them alongside rUIP shares isolates the visible effect of the local rule.
    precision_only_share = weights.events.to_numpy(float) / weights.events.sum()
    ax.scatter(
        precision_only_share,
        wy,
        marker="D",
        s=25,
        facecolor="white",
        edgecolor="#303030",
        linewidth=1.0,
        zorder=4,
    )
    ax.scatter(
        [weights.weight.iloc[4]], [wy[4]], marker="s", s=22, color="#D55E00", zorder=5
    )
    ax.set_yticks(wy, [f"H{i}" for i in range(1, 10)])
    ax.set_xlabel(
        "Share of historical information\nbars: rUIP retained; diamonds: precision-only"
    )
    ax.set_title(
        "B  Local weighting changes source influence", loc="left", fontweight="bold"
    )
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.set_xlim(0, max(0.36, float(weights.weight.max()) + 0.05))

    # Panel C: current-study posterior under every method.
    ax = axes[1, 0]
    py = np.arange(len(ONCOLOGY_METHODS))[::-1]
    for yi, method in zip(py, ONCOLOGY_METHODS, strict=True):
        row = posterior.loc[method]
        x, lo, hi = (
            np.exp(row.estimate),
            np.exp(row.interval_lower),
            np.exp(row.interval_upper),
        )
        ax.errorbar(
            x,
            yi,
            xerr=[[x - lo], [hi - x]],
            fmt="o",
            color=ONCOLOGY_COLORS[method],
            ecolor=ONCOLOGY_COLORS[method],
            capsize=2,
            lw=1.0,
            ms=4.2,
        )
    display_methods = {
        "NIP": "NIP (current data only)",
        "PP": "PP",
        "rMAP": "rMAP",
        "CP": "CP",
        "UIP": "UIP",
        "rUIP": "rUIP",
    }
    ax.set_yticks(py, [display_methods[method] for method in ONCOLOGY_METHODS])
    ax.set_xlabel("Exp(posterior mean log hazard) and transformed 95% CrI")
    ax.set_title("C  Posterior estimates (descriptive)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)

    # Panel D: leave-one-trial-out predictive evaluation.
    ax = axes[1, 1]
    ly = np.arange(len(ONCOLOGY_BORROWING))[::-1]
    ruip_y = ly[list(ONCOLOGY_BORROWING).index("rUIP")]
    ax.axhspan(ruip_y - 0.42, ruip_y + 0.42, color="#DCEAF7", zorder=0)
    for yi, method in zip(ly, ONCOLOGY_BORROWING, strict=True):
        values = loo.loc[
            loo.method.eq(method), "prior_predictive_log_density"
        ].to_numpy()
        mean = float(np.mean(values))
        ax.scatter(values, np.full(values.size, yi), s=12, color="#B5B5B5", zorder=2)
        ax.scatter(
            [mean],
            [yi],
            marker="D",
            s=32,
            color=ONCOLOGY_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.text(mean + 0.035, yi + 0.12, f"mean {mean:.3f}", fontsize=8.3)
    ax.set_yticks(ly, ONCOLOGY_BORROWING)
    ax.set_xlabel("Leave-one-trial-out log predictive density (higher is better)")
    ax.set_title("D  LOO prediction (higher is better)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.95,
        bottom=0.08,
        wspace=0.34,
        hspace=0.34,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = [
        FIGURES / "figure_oncology_application.pdf",
        FIGURES / "figure_oncology_application.png",
    ]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1], dpi=300)
    plt.close(fig)
    inputs = [
        ONCOLOGY_DATA,
        ONCOLOGY_RESULTS / "manifest.json",
        ONCOLOGY_RESULTS / "posterior.csv",
        ONCOLOGY_RESULTS / "loo_predictive.csv",
        ONCOLOGY_RESULTS / "source_weights.csv",
    ]
    for output in outputs:
        payload = {
            "artifact": str(output.relative_to(ROOT)),
            "sha256": sha256(output),
            "generator": str(Path(__file__).relative_to(ROOT)),
            "generator_sha256": sha256(Path(__file__)),
            "analysis_manifest_status": manifest["status"],
            "inputs": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in inputs
            ],
        }
        output.with_suffix(output.suffix + ".provenance.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps({"status": "PASS", "outputs": [str(x) for x in outputs]}, indent=2)
    )


def precision_input_dirs() -> list[Path]:
    return [
        ROOT / "results" / f"precision_leverage_confirmation_{outcome}"
        for outcome in ("continuous", "binary", "survival")
    ]


def main() -> None:
    """Read local experiment outputs and generate the paper figures; the
    generated images remain local and are never part of the source release."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="figure", required=True)
    precision = sub.add_parser("precision-leverage")
    precision.add_argument(
        "--inputs",
        nargs=3,
        type=Path,
        metavar=("CONTINUOUS", "BINARY", "SURVIVAL"),
        default=precision_input_dirs(),
        help="formal confirmation result directories in outcome order",
    )
    for name in ("joint-conflict", "component-ablation", "oncology", "all"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.figure in ("precision-leverage", "all"):
        make_precision_leverage(
            args.inputs
            if args.figure == "precision-leverage"
            else precision_input_dirs()
        )
    if args.figure in ("joint-conflict", "all"):
        make_joint_conflict()
    if args.figure in ("component-ablation", "all"):
        make_component_ablation()
    if args.figure in ("oncology", "all"):
        make_oncology_application()


if __name__ == "__main__":
    main()
