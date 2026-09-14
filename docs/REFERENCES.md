# References

Methodological foundations, academic literature, and research publications supporting the pipeline's algorithms and modeling frameworks.

## Core Algorithm Sources

- **Toole, J. L., Colak, S., Sturt, B., Alexander, L. P., Evsukoff, A., &
  González, M. C. (2015).** "The path most traveled: Travel demand
  estimation using big data resources." *Transportation Research Part C:
  Emerging Technologies*, 58, 162–177.
  https://doi.org/10.1016/j.trc.2015.04.022
  — Source for stay-point detection (`stay_points.py`, Algorithms 2–5)
  and origin-destination home/work place inference (`label_places.py`,
  Algorithm 6).

- **Zheng, Y. (2015).** "Trajectory Data Mining: An Overview." *ACM
  Transactions on Intelligent Systems and Technology*, 6(3), Article 29.
  https://doi.org/10.1145/2743025
  — Formulation for trajectory pre-processing and noise filtering
  using constant-velocity Kalman and particle filtering (`noise_reduction.py`).

- **de Montjoye, Y.-A., Rocher, L., & Pentland, A. S. (2016).** "bandicoot:
  a Python toolbox for mobile phone metadata." *Journal of Machine
  Learning Research*, 17(175), 1–5.
  https://www.jmlr.org/papers/v17/15-593.html
  — Reference taxonomy for behavioral metadata indicators across spatial,
  temporal, social, and recharge dimensions (`bandicoot_features.py`).

- **Steele, J. E., Sundsøy, P. R., Pezzulo, C., et al. (2017).** "Mapping
  poverty using mobile phone and satellite data." *Journal of the Royal
  Society Interface*, 14(127), 20160690.
  https://doi.org/10.1098/rsif.2016.0690
  — Empirical basis for modeling airtime recharge frequency and denomination
  profiles across socio-economic groups (`generate_bengaluru_data.py`).

- **Fellegi, I. P., & Sunter, A. B. (1969).** "A theory for record
  linkage." *Journal of the American Statistical Association*, 64(328),
  1183–1210. https://doi.org/10.1080/01621459.1969.10501049
  — Probabilistic record linkage framework for identifier resolution in
  empirical telecom-census data reconciliation.

- **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.
  (2002).** "SMOTE: Synthetic minority over-sampling technique."
  *Journal of Artificial Intelligence Research*, 16, 321–357.
  https://doi.org/10.1613/jair.953
  — Stratified cross-validation oversampling strategy for imbalanced
  demographic classes (`train.py`).

- **González, M. C., Hidalgo, C. A., & Barabási, A.-L. (2008).**
  "Understanding individual human mobility patterns." *Nature*, 453,
  779–782. https://doi.org/10.1038/nature06958
  — Theoretical basis for radius of gyration and individual mobility
  entropy metrics.

- **Newman, M. E. J. (2003).** "The structure and function of complex
  networks." *SIAM Review*, 45(2), 167–256.
  https://doi.org/10.1137/S003614450342480
  — Graph topology and centrality formulations for communication and
  co-location networks.

- **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin,
  A. (2018).** "CatBoost: unbiased boosting with categorical features."
  *Advances in Neural Information Processing Systems (NeurIPS 31)*.
  https://arxiv.org/abs/1706.09516
  — Categorical gradient boosted decision trees utilized in `train.py`.

## IIT Bombay Transportation Systems Engineering Research

Foundational research publications and course curriculum from the Transportation Systems Engineering group, Department of Civil Engineering, IIT Bombay:

1. **Mathew, T. V.** *Travel Demand Modeling* (Course lecture notes,
   Transportation Systems Engineering, Department of Civil Engineering,
   IIT Bombay).
   https://www.civil.iitb.ac.in/~vmtom/1100_LnTse/900_allln/ce_tvm_tse_ln.pdf
   — Classical four-step travel demand modeling (trip generation, trip
   distribution, modal split, traffic assignment), providing the direct
   context for synthetic population synthesis.

2. **Mathew, T. V., Joshi, G. J., & Velaga, N. R. (Eds.) (2019).**
   *Transportation Research: Proceedings of CTRG 2017.* Lecture Notes in
   Civil Engineering, Springer. https://doi.org/10.1007/978-981-32-9042-6
   — Conference of the Transportation Research Group of India proceedings.

3. **Yadav, A. K., Pawar, N. M., & Velaga, N. R. (2024).** "Modeling the
   Influence of Smartphone Distraction and Pedestrian Characteristics on
   Pedestrian Road Crossing Behavior." *Transportation Research Record.*
   https://doi.org/10.1177/03611981231189499
   — Empirical analysis of mobile-device behavioral characteristics in
   urban pedestrian dynamics.

4. **Singh, V. K., Kumar, V., & Jana, A. (2021).** "Spatial Distribution
   of Socioeconomic Factors and Its Impact on Urban Land Use Dynamics: An
   Agent Based Modeling Approach." In *Urban Science and Engineering*,
   Lecture Notes in Civil Engineering, vol. 121, Springer.
   https://doi.org/10.1007/978-981-33-4114-2_3
   — Agent-based modeling of socio-economic distributions in urban spatial
   analysis.

5. **Thomas, N., Jana, A., & Bandyopadhyay, S. (2021).** "Do Socioeconomic
   Characteristics Affect Travel Time and Transport Perception? Insights
   from Mumbai, India." In *Urban Science and Engineering*, Lecture Notes
   in Civil Engineering, vol. 121, Springer.
   https://doi.org/10.1007/978-981-33-4114-2_14
   — Relationship between socio-economic attributes and travel patterns in
   Indian metropolitan environments.

6. **Mittal, A., Talebpour, A., & Mahmassani, H. (2017).** "Network Flow
   Relations and Travel Time Reliability in a Connected Environment."
   *Transportation Research Record*, 2622, 24–37.
   https://doi.org/10.3141/2622-03
   — Network reliability and flow dynamics under connected mobility.

7. **Mittal, A., Kim, E., Mahmassani, H. S., & Hong, Z. (2018).**
   "Predictive Dynamic Speed Limit in a Connected Environment for a
   Weather Affected Traffic Network: A Case Study of Chicago."
   *Transportation Research Record.*
   https://doi.org/10.1177/0361198118791668
   — Predictive traffic network management and dynamic modeling.

*Note*: Items 1–5 represent IIT Bombay Transportation Systems Engineering publications and course curriculum; items 6–7 cover transportation network reliability and traffic dynamics research by Dr. Archak Mittal.

## Additional Survey Literature

- **Blondel, V. D., Decuyper, A., & Krings, G. (2015).** "A survey of results on mobile phone datasets analysis." *EPJ Data Science*, 4(1), 10. https://doi.org/10.1140/epjds/s13688-015-0046-0 — Comprehensive review of telecommunications data analysis for human dynamics and mobility.
