# Reference sources

The proof-of-concept was implemented after checking two user-provided local PDF copies. The PDFs remain local and are excluded from Git to avoid redistributing copyrighted full text.

1. Huaqing Jin and Guosheng Yin (2021), "Unit information prior for adaptive information borrowing from multiple historical datasets", *Statistics in Medicine*. DOI: `10.1002/sim.9146`.
   - Local working-copy name: `jin_yin_unit_information_prior.pdf`
   - SHA-256: `13d8987db2385bb87cdde12efed5269de29113731c289961b5850d0115b178d5`
   - Used for: the weighted UIP mean, per-observation Fisher information, Dirichlet modeling of study weights, and a bounded uniform hyperprior for nominal borrowed information `M`.

2. Hengtao Zhang and Guosheng Yin (2023), "Unit information prior for incorporating real-world evidence into randomized controlled trials", *Statistical Methods in Medical Research*, 32(2), 229–241. DOI: `10.1177/09622802221133555`.
   - Local working-copy name: `uip_reference.pdf`
   - SHA-256: `5e868893b802c978416e0eaf5e6cf225028ae03ff541e322a3b48f336bfa594d`
   - Used for: summary-statistic borrowing for a Cox treatment effect and the piecewise-exponential PH baseline construction.

The interval-censoring data-augmentation extension and dynamic weight sampler in this repository are proof-of-concept implementations supplied for the present project. They should not be attributed to either paper without independent verification. Broader related work is documented in `literature-search-20260727-interval-censored-borrowing/`.
