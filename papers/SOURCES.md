# Reference sources

The proof-of-concept was implemented after checking the following user-provided local PDF copies. The PDF files are retained locally in this directory for the user's use but excluded from Git to avoid redistributing copyrighted full text.

1. Huaqing Jin and Guosheng Yin (2021), "Unit information prior for adaptive information borrowing from multiple historical datasets", *Statistics in Medicine*. DOI: `10.1002/sim.9146`.
   - Local read-only source found at: `C:\Users\bobzh\Desktop\PS-UIP\UIP\Jin和Yin - Unit information prior for adaptive information borrowing from multiple historical datasets.pdf`
   - Local working copy: `jin_yin_unit_information_prior.pdf`
   - SHA-256: `13d8987db2385bb87cdde12efed5269de29113731c289961b5850d0115b178d5`
   - Used for: the weighted UIP mean, weighted per-observation Fisher information, fixed study weights, and a bounded uniform hyperprior for the nominal borrowed information `M`.

2. Hengtao Zhang and Guosheng Yin (2023), "Unit information prior for incorporating real-world evidence into randomized controlled trials", *Statistical Methods in Medical Research*, 32(2), 229-241. DOI: `10.1177/09622802221133555`.
   - Local read-only source found at: `C:\Users\bobzh\Documents\学术研究\[202403] 国自然\张恒韬\Paper\UIP.pdf`
   - Local working copy: `uip_reference.pdf`
   - SHA-256: `5e868893b802c978416e0eaf5e6cf225028ae03ff541e322a3b48f336bfa594d`
   - Used for: summary-statistic borrowing for a Cox treatment effect and the piecewise-exponential PH baseline construction.

The interval-censoring data-augmentation extension in this repository is a proof-of-concept implementation supplied for the present project. It should not be attributed to either paper without independent verification.
