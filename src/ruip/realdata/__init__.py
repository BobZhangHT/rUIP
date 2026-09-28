"""Real-data application interfaces."""

from .design_stage import (
    METHODS,
    ClinicalAnalysisResult,
    analyze_memantine,
    analyze_secukinumab,
    validate_rows,
)
from .gaussian_summary import (
    GaussianPosteriorResult,
    GaussianSummary,
    analyze_gaussian_summary,
    freeze_priors,
    validate_gaussian_rows,
)

__all__ = [
    "METHODS",
    "ClinicalAnalysisResult",
    "analyze_memantine",
    "analyze_secukinumab",
    "validate_rows",
    "GaussianPosteriorResult",
    "GaussianSummary",
    "analyze_gaussian_summary",
    "freeze_priors",
    "validate_gaussian_rows",
]
