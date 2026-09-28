# ruff: noqa: E501 -- preserve the exact published aggregate input rows.
"""Published aggregate input summaries for the illustrative analyses.

These are source data transcriptions, not simulation results or fitted outputs.
The materializer writes CSV/JSON files only into the ignored local data folder.
"""

from pathlib import Path

INPUTS = {
    'clinical_memantine_npi.csv': """role,study,arm,n,mean_change,sd
historical,LU-99679,treatment,146,-0.36,10.40
historical,LU-99679,placebo,64,-2.23,9.55
historical,MEM-MD-01,treatment,133,-2.11,15.12
historical,MEM-MD-01,placebo,127,0.51,13.75
historical,MEM-MD-02,treatment,171,-0.75,11.03
historical,MEM-MD-02,placebo,152,2.78,13.48
historical,MEM-MD-10,treatment,107,0.77,12.06
historical,MEM-MD-10,placebo,118,2.83,15.70
historical,MRZ-9605,treatment,97,0.09,15.92
historical,MRZ-9605,placebo,84,2.89,16.13
current,MEM-MD-12,treatment,136,0.97,11.26
current,MEM-MD-12,placebo,125,0.86,11.08
""",
    'clinical_secukinumab_asas20.csv': """role,study,historical_active_drug,events,total,count_status,assessment_window,assessment_window_source
historical,ATLAS,adalimumab,23,107,reported,within 2--12 weeks; exact study visit not encoded,McLeod et al. (2007) Figure 1
historical,Canadian AS,adalimumab,12,44,reported,within 2--12 weeks; exact study visit not encoded,McLeod et al. (2007) Figure 1
historical,Wyeth 0881A3-314,etanercept,19,51,reported,within 2--12 weeks; exact study visit not encoded,McLeod et al. (2007) Figure 1
historical,Calin,etanercept,9,39,reported,within 2--12 weeks; exact study visit not encoded,McLeod et al. (2007) Figure 1
historical,Davis,etanercept,39,139,reported,within 2--12 weeks; exact study visit not encoded,McLeod et al. (2007) Figure 1
historical,Gorman,etanercept,6,20,reported,16 weeks; included with 12-week analysis,McLeod et al. (2007) Figure 1 and text
historical,ASSERT,infliximab,9,78,estimated_from_percentage,within 2--12 weeks; exact study visit not encoded,McLeod et al. (2007) Figure 1
historical,Braun,infliximab,10,35,estimated_from_percentage,within 2--12 weeks; exact study visit not encoded,McLeod et al. (2007) Figure 1
current,Baeten 2013 placebo,,1,6,reported,6 weeks,Baeten et al. (2013) Table 2
current,Baeten 2013 secukinumab,,14,23,reported,6 weeks,Baeten et al. (2013) Table 2
""",
    'clinical_oncology_hazard.csv': """role,study,events,exposure
historical,Roychoudhury-Neuenschwander-01,14,45.0
historical,Roychoudhury-Neuenschwander-02,32,110.8
historical,Roychoudhury-Neuenschwander-03,29,114.7
historical,Roychoudhury-Neuenschwander-04,13,25.3
historical,Roychoudhury-Neuenschwander-05,22,23.7
historical,Roychoudhury-Neuenschwander-06,31,86.4
historical,Roychoudhury-Neuenschwander-07,18,36.7
historical,Roychoudhury-Neuenschwander-08,10,48.7
historical,Roychoudhury-Neuenschwander-09,10,25.4
current,Zhang-current,32,117.6
""",
    'clinical_oncology_hazard.provenance.json': """{
  "analysis_status": "exploratory retrospective illustration",
  "selection_date": "2026-09-18",
  "selection_note": "The dataset was selected after the manuscript project began and was not prospectively registered. It was chosen because the source analysis designates one current trial, provides nine historical trials estimating the same scalar log hazard, and contains visible between-trial heterogeneity. Method ranking was not treated as confirmatory evidence.",
  "primary_source": {
    "citation": "Roychoudhury S, Neuenschwander B. Bayesian leveraging of historical control data for a clinical trial with time-to-event endpoint. Statistics in Medicine. 2020;39:984-995.",
    "doi": "10.1002/sim.8456"
  },
  "secondary_source": {
    "citation": "Zhang H, Shen Y, Li J, Ye H, Chiang AY. Adaptively leveraging external data with robust meta-analytical-predictive prior using empirical Bayes. Pharmaceutical Statistics. 2023;22:846-860.",
    "doi": "10.1002/pst.2315"
  },
  "transcribed_fields": [
    "historical/current role",
    "trial order",
    "event count",
    "person-years of exposure"
  ],
  "model_reduction": "Each trial is represented by log(events/exposure) with observed-information variance approximation 1/events under a constant-hazard exponential model.",
  "limitations": [
    "The archived file contains aggregate summaries rather than individual event times.",
    "The analysis does not recover piecewise baseline hazards.",
    "The dataset is an illustrative case study, not a prospectively selected confirmatory benchmark."
  ]
}
""",
}


def materialize_inputs(root: Path) -> None:
    """Write missing published aggregate inputs without replacing local edits."""
    output = root / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in INPUTS.items():
        path = output / name
        if not path.exists():
            path.write_text(contents, encoding="utf-8", newline="")
