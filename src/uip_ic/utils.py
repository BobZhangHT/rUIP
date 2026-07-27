"""Configuration, plotting, and deterministic seed helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def derived_seed(base_seed: int, scenario_index: int, repetition: int, stream: int) -> int:
    sequence = __import__("numpy").random.SeedSequence([base_seed, scenario_index, repetition, stream])
    return int(sequence.generate_state(1, dtype="uint32")[0])


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
