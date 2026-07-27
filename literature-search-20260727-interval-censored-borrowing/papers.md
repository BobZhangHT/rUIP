# Interval-censored survival borrowing: focused literature map

Search date: 2026-07-27. Scope: Bayesian historical information borrowing, interval-censored survival inference, and computational methods directly informing IC-UIP. This is a focused positioning review rather than a systematic review.

## Main finding

Fang et al. (2025) is the closest direct precedent: it combines interval-censored survival data with commensurate historical borrowing in a piecewise-exponential proportional-hazards model. Consequently, IC-UIP should not be described as the first Bayesian historical-borrowing method for interval censoring. Its more defensible distinction is the combination of multiple historical treatment-effect summaries, unit-information scaling, jointly adaptive total borrowing and source weights, and a summary-only data-augmentation implementation.

## Evidence map

| Work | Role in comparison | Relation to IC-UIP | Evidence |
| --- | --- | --- | --- |
| Fang et al. (2025), *Commensurate prior models with random effects for interval-censored data to accommodate historical controls* | Closest direct comparator | Interval-censored PH, matched historical controls, random effects and commensurate priors; primarily individual-level historical-control borrowing rather than multiple treatment-effect summaries | [PubMed](https://pubmed.ncbi.nlm.nih.gov/42445526/), [DOI](https://doi.org/10.1080/03610918.2025.2593950) |
| Jin & Yin (2021), *Unit information prior for adaptive information borrowing from multiple historical datasets* | UIP foundation | Defines adaptive total information and study-specific weighting across multiple historical datasets | [DOI](https://doi.org/10.1002/sim.9146) |
| Zhang & Yin (2023), *Unit information prior for incorporating real-world evidence into randomized controlled trials* | Survival-UIP foundation | Applies summary-level UIP borrowing to Cox treatment effects, but not general interval censoring | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10147603/), [DOI](https://doi.org/10.1177/09622802221133555) |
| Gu & Yin (2024), *Unit information Dirichlet process prior* | Nonparametric UIP neighbor | Borrows survival distributions through a unit-information DP; does not target interval-censored regression-effect summaries | [DOI](https://doi.org/10.1093/biomtc/ujae091) |
| Murray et al. (2014), *Semiparametric Bayesian commensurate survival model for post-market medical device surveillance with non-exchangeable historical data* | Survival commensurate-prior precursor | Piecewise-exponential, right-censored survival borrowing and nonexchangeability | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4007051/), [DOI](https://doi.org/10.1111/biom.12115) |
| Hobbs et al. (2012), *Commensurate priors for incorporating historical information in clinical trials using general and generalized linear models* | General CP foundation | Establishes adaptive commensurability modeling used by later survival approaches | [DOI](https://doi.org/10.1214/12-BA722) |
| Ibrahim et al. (2015), *The power prior: theory and applications* | Alternative borrowing family | Provides the main likelihood-discounting comparator family | [DOI](https://doi.org/10.1002/sim.6728) |
| Schmidli et al. (2014), *Robust meta-analytic-predictive priors in clinical trials with historical control information* | Robust borrowing benchmark | Mixture-based protection against prior–data conflict; not interval-censoring-specific | [PubMed](https://pubmed.ncbi.nlm.nih.gov/25355546/), [DOI](https://doi.org/10.1111/biom.12242) |
| Zeng, Mao & Lin (2016), *Maximum likelihood estimation for semiparametric transformation models with interval-censored data* | Core interval-censoring benchmark | NPMLE/EM framework for interval-censored transformation models; no historical borrowing | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4803648/), [DOI](https://doi.org/10.1093/biomet/asw013) |
| Wang et al. (2016), *Analysis of interval-censored survival data via Poisson data augmentation* | Computational comparator | Poisson augmentation/EM for interval-censored PH inference; no information prior | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4803649/), [DOI](https://doi.org/10.1111/biom.12389) |
| Henschel et al. (2009), *Bayesian inference for interval-censored survival data in the presence of frailty* | Bayesian DA precedent | Bayesian PH interval-censoring with data augmentation and frailty; no adaptive historical borrowing | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2654881/), [DOI](https://doi.org/10.1186/1471-2288-9-9) |
| Pan (2000), *A multiple imputation approach to Cox regression with interval-censored data* | Earlier latent-time approach | Multiple imputation for interval-censored Cox regression; no Bayesian borrowing | [DOI](https://doi.org/10.1111/j.0006-341X.2000.00199.x) |

## Recommended comparison hierarchy

1. `IC-NIP`: isolates the gain and risk from borrowing.
2. `IC-UIP`: proposed summary-only method with dynamic `M` and dynamic weights.
3. `IC-CP`: feasible summary-normal commensurate comparator in this proof of concept.
4. Future full study: reproduce Fang et al. with individual-level matched controls; add robust MAP or normalized power prior; benchmark NPMLE/EM.

The implemented `IC-CP` is intentionally labeled a summary-level approximation. It is not claimed to reproduce Fang et al.'s random-effects model.

## Research gap supported by this map

The literature already covers interval-censored Bayesian modeling, data augmentation, general historical borrowing, and a recent interval-censored commensurate-prior model. The remaining methodological opportunity is narrower: adaptive borrowing from multiple historical regression-effect summaries on an interval-censoring-aware unit-information scale, while estimating source weights and total borrowing inside one computationally transparent sampler.
