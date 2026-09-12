# References

Every methodological choice in this pipeline is mapped below to its
source, with a public link wherever one exists. Venue type is noted
explicitly — some foundational papers in this space were published in
top-tier CS/engineering conferences rather than journals, and those are
labeled as such rather than misrepresented as journal publications.

## Core algorithm sources (used directly in code)

- **Toole, J. L., Colak, S., Sturt, B., Alexander, L. P., Evsukoff, A., &
  González, M. C. (2015).** "The path most traveled: Travel demand
  estimation using big data resources." *Transportation Research Part C:
  Emerging Technologies*, 58, 162–177.
  https://doi.org/10.1016/j.trc.2015.04.022 · **[Q1 journal]**
  — `stay_points.py` implements this paper's stay-point algorithm
  (Algorithms 2–5: candidate stays, grid-based agglomerative clustering,
  final snapping pass) and `label_places.py` implements its OD Creation
  Algorithm's home/work detection rule (Algorithm 6) exactly.

- **Zheng, Y. (2015).** "Trajectory Data Mining: An Overview." *ACM
  Transactions on Intelligent Systems and Technology*, 6(3), Article 29.
  Open-access PDF (Microsoft Research):
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/TrajectoryDataMining-tist-yuzheng_published.pdf
  · **[Q1 journal]** — source for the Kalman/particle-filter noise
  reduction step in `noise_reduction.py`; Zheng's survey identifies
  noise filtering as the first trajectory pre-processing stage and
  recommends Kalman or particle filters over simple mean/median filters
  for low, irregularly-sampled trajectories (the case here).

- **de Montjoye, Y.-A., Rocher, L., & Pentland, A. S. (2016).** "bandicoot:
  a Python toolbox for mobile phone metadata." *Journal of Machine
  Learning Research*, 17(175), 1–5.
  https://www.jmlr.org/papers/v17/15-593.html · **[Q1 journal]** — the
  behavioral-indicator schema (`bandicoot_features.py`) reproduces this
  toolbox's standard individual/spatial/social feature taxonomy.

- **Steele, J. E., Sundsøy, P. R., Pezzulo, C., et al. (2017).** "Mapping
  poverty using mobile phone and satellite data." *Journal of the Royal
  Society Interface*, 14(127), 20160690.
  https://doi.org/10.1098/rsif.2016.0690 · **[Q1 journal]** — grounds the
  recharge-frequency-vs-income direction encoded in `RECHARGE_PROFILE`
  (generate_bengaluru_data.py).

- **Fellegi, I. P., & Sunter, A. B. (1969).** "A theory for record
  linkage." *Journal of the American Statistical Association*, 64(328),
  1183–1210. https://doi.org/10.1080/01621459.1969.10501049 · **[Q1
  journal]** — the probabilistic record linkage framework needed to
  replace exact phone-number matching for real CDR deployment (see
  `docs/DATA_LINEAGE.md`).

- **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.
  (2002).** "SMOTE: Synthetic minority over-sampling technique."
  *Journal of Artificial Intelligence Research*, 16, 321–357.
  https://doi.org/10.1613/jair.953 · **[Q1 journal]** — the per-fold
  oversampling strategy in `train.py`.

- **González, M. C., Hidalgo, C. A., & Barabási, A.-L. (2008).**
  "Understanding individual human mobility patterns." *Nature*, 453,
  779–782. https://doi.org/10.1038/nature06958 · **[Q1 journal]** —
  radius-of-gyration and individual mobility-signature concepts.

- **Newman, M. E. J. (2003).** "The structure and function of complex
  networks." *SIAM Review*, 45(2), 167–256.
  https://doi.org/10.1137/S003614450342480 · **[Q1 journal]** — graph
  structural feature definitions, relevant if the optional three-graph
  extension (see Developer Handbook) is reinstated.

- **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin,
  A. (2018).** "CatBoost: unbiased boosting with categorical features."
  *NeurIPS 31*. Preprint: https://arxiv.org/abs/1706.09516
  · **[Top-tier ML conference, not a journal]** — flagged explicitly, since
  a journal-only reference list would need a substitute or an explicit note.

## IIT Bombay Civil Engineering Department — transportation research

1. **Mathew, T. V.** *Travel Demand Modeling* (course notes, Transportation
   Systems Engineering, Department of Civil Engineering, IIT Bombay).
   https://www.civil.iitb.ac.in/~vmtom/1100_LnTse/900_allln/ce_tvm_tse_ln.pdf
   — covers the classical four-step model (trip generation, distribution,
   modal split, assignment); the home/work anchor detection and OD-matrix
   framing in this pipeline sit conceptually upstream of this material
   (population synthesis feeding trip generation).

2. **Mathew, T. V., Joshi, G. J., & Velaga, N. R. (Eds.) (2019).**
   *Transportation Research: Proceedings of CTRG 2017.* Lecture Notes in
   Civil Engineering, Springer. https://doi.org/10.1007/978-981-32-9042-6
   — proceedings of the Conference of the Transportation Research Group
   of India, co-edited by two IIT Bombay Civil Engineering faculty
   (Mathew and Velaga).

3. **Yadav, A. K., Pawar, N. M., & Velaga, N. R. (2024).** "Modeling the
   Influence of Smartphone Distraction and Pedestrian Characteristics on
   Pedestrian Road Crossing Behavior." *Transportation Research Record.*
   https://doi.org/10.1177/03611981231189499 — IIT Bombay Civil
   Engineering (Transportation Systems Engineering), on behavioral data
   collected via smartphone use, relevant to the phone-behavior framing
   of this project.

4. **Singh, V. K., Kumar, V., & Jana, A. (2021).** "Spatial Distribution
   of Socioeconomic Factors and Its Impact on Urban Land Use Dynamics: An
   Agent Based Modeling Approach." In *Urban Science and Engineering*,
   Lecture Notes in Civil Engineering, vol. 121, Springer.
   https://doi.org/10.1007/978-981-33-4114-2_3 — IIT Bombay Civil
   Engineering (Centre for Urban Science and Engineering), directly
   relevant to the socio-demographic + agent-based framing used for the
   synthetic population here.

5. **Thomas, N., Jana, A., & Bandyopadhyay, S. (2021).** "Do Socioeconomic
   Characteristics Affect Travel Time and Transport Perception? Insights
   from Mumbai, India." In *Urban Science and Engineering*, Lecture Notes
   in Civil Engineering, vol. 121, Springer.
   https://doi.org/10.1007/978-981-33-4114-2_14 — IIT Bombay Civil
   Engineering, socioeconomic characteristics vs. travel behavior in an
   Indian metro (Mumbai), the closest published analog in the department
   to this project's target variables.

6. **Mittal, A., Talebpour, A., & Mahmassani, H. (2017).** "Network Flow
   Relations and Travel Time Reliability in a Connected Environment."
   *Transportation Research Record*, 2622, 24–37.
   https://doi.org/10.3141/2622-03 — by the SRFP fellowship's guide, Dr.
   Archak Mittal (now IIT Bombay Civil Engineering), from his PhD-era
   research at Northwestern University.

7. **Mittal, A., Kim, E., Mahmassani, H. S., & Hong, Z. (2018).**
   "Predictive Dynamic Speed Limit in a Connected Environment for a
   Weather Affected Traffic Network: A Case Study of Chicago."
   *Transportation Research Record.*
   https://doi.org/10.1177/0361198118791668 — also by Dr. Archak Mittal.

## Note on citation completeness

This list covers methods actually implemented in the code, plus the
requested departmental papers. It does not attempt a full literature
review of the CDR/mobility field — Blondel, Decuyper & Krings (2015),
"A survey of results on mobile phone datasets analysis," *EPJ Data
Science*, 4(1), https://doi.org/10.1140/epjds/s13688-015-0046-0
**[Q1 journal]**, is a good starting survey for that purpose.
