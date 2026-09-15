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

## Curated Q1 Academic Literature (with Public Links)

### 1. Population Synthesis & Joint Socio-Demographic Modeling

- **Sun, L., & Erath, A. (2015).** "A Bayesian network approach for population synthesis."
  *Transportation Research Part C: Emerging Technologies*, 61, 49–62.
  - **Status**: Q1 (Transportation & Civil Engineering)
  - **Public Link (DOI)**: https://doi.org/10.1016/j.trc.2015.10.010
  - **Open Access (arXiv)**: https://arxiv.org/abs/1507.01886
  - **Pipeline Context**: Framework for modeling joint multi-attribute dependencies (age, education, occupation, income) using probabilistic graphical models (DAGs) rather than isolated independent classifiers, avoiding logically impossible demographic combinations.

- **Borysov, S. S., Rich, J., & Pereira, F. C. (2019).** "How to generate micro-agents? A deep generative modeling approach to population synthesis."
  *Transportation Research Part C: Emerging Technologies*, 106, 73–87.
  - **Status**: Q1 (Transportation & Civil Engineering)
  - **Public Link (DOI)**: https://doi.org/10.1016/j.trc.2019.07.006
  - **Open Access (arXiv)**: https://arxiv.org/abs/1903.04942
  - **Pipeline Context**: Pioneering deep generative modeling (Variational Autoencoders) for synthesizing micro-agents from tabular demographic data while capturing complex non-linear correlations and overcoming sampling zeros.

- **Farooq, B., Bierlaire, M., Hurtubia, R., & Flötteröd, G. (2013).** "Simulation based population synthesis."
  *Transportation Research Part B: Methodological*, 58, 243–263.
  - **Status**: Q1 (Transportation Science & Operations Research)
  - **Public Link (DOI)**: https://doi.org/10.1016/j.trb.2013.09.012
  - **Open Access (EPFL InfoScience)**: https://infoscience.epfl.ch/record/188448
  - **Pipeline Context**: Methodological benchmark for drawing heterogeneous individual agents from joint multi-level distributions using Markov Chain Monte Carlo (MCMC) and Gibbs sampling.

- **Hörl, S., & Balać, M. (2021).** "Synthetic population and travel demand for Paris and Île-de-France based on open and public data."
  *Transportation Research Part C: Emerging Technologies*, 130, 103291.
  - **Status**: Q1 (Transportation & Civil Engineering)
  - **Public Link (DOI)**: https://doi.org/10.1016/j.trc.2021.103291
  - **Open Access (arXiv)**: https://arxiv.org/abs/2009.07186
  - **Pipeline Context**: Comprehensive open-source pipeline linking synthetic population seeds to agent-based activity-travel diaries in MATSim.

- **Ye, X., Konduri, K., Pendyala, R. M., Sana, B., & Waddell, P. (2009).** "A methodology to match distributions of both household and person attributes in the generation of synthetic populations."
  *Transportation Research Board 88th Annual Meeting*, Washington, D.C.
  - **Pipeline Context**: Formulation of the Iterative Proportional Updating (IPU) algorithm implemented in `ipu.py` to match multi-attribute marginal control totals.

- **Müller, K., & Axhausen, K. W. (2011).** "Hierarchical IPM: a new approach to population synthesis."
  *Transportation Research Record*, 2255(1), 10–18.
  - **Public Link (DOI)**: https://doi.org/10.3141/2255-02
  - **Pipeline Context**: Multi-level hierarchical iterative proportional fitting and raking for travel demand generation.

### 2. Telecommunication Metadata for Travel Demand & Demographics

- **Bwambale, A., Choudhury, C. F., & Hess, S. (2019).** "Modelling trip generation using mobile phone data: A latent demographics approach."
  *Journal of Transport Geography*, 76, 276–286.
  - **Status**: Q1 (Geography, Planning and Development / Transportation)
  - **Public Link (DOI)**: https://doi.org/10.1016/j.jtrangeo.2019.03.011
  - **Open Access (White Rose Repository)**: https://eprints.whiterose.ac.uk/144933/
  - **Pipeline Context**: Uses mobile phone metadata to infer unobserved latent demographic classes and predict daily trip generation rates without ground-truth individual travel surveys.

- **Iqbal, M. S., Choudhury, C. F., Wang, P., & González, M. C. (2014).** "Development of origin–destination matrices using mobile phone call data."
  *Transportation Research Part C: Emerging Technologies*, 40, 63–74.
  - **Status**: Q1 (Transportation & Civil Engineering)
  - **Public Link (DOI)**: https://doi.org/10.1016/j.trc.2014.01.002
  - **Open Access (ResearchGate / MIT)**: https://www.researchgate.net/publication/260714777_Development_of_origin-destination_matrices_using_mobile_phone_call_data
  - **Pipeline Context**: Seminal methodology for cellular tower Voronoi tessellation, ping-to-tower discretization, and scaling sample CDR trips to population-level travel demand.

- **Alexander, L., Jiang, S., Murga, M., & González, M. C. (2015).** "Origin–destination trips by purpose and time of day inferred from mobile phone data."
  *Transportation Research Part C: Emerging Technologies*, 58, 240–250.
  - **Status**: Q1 (Transportation & Civil Engineering)
  - **Public Link (DOI)**: https://doi.org/10.1016/j.trc.2015.02.018
  - **Pipeline Context**: Expansion of stay-point detection to infer trip purposes (home, work, other) and activity-specific commute distances integrated into `bandicoot_features.py`.

- **Blumenstock, J., Cadamuro, G., & On, R. (2015).** "Predicting poverty and wealth from mobile phone metadata."
  *Science*, 350(6264), 1073–1077.
  - **Status**: Q1 (Multidisciplinary Sciences; Nature/Science flagship)
  - **Public Link (DOI)**: https://doi.org/10.1126/science.aac4420
  - **Open Access (Author PDF)**: http://www.jblumenstock.com/files/papers/Science2015.pdf
  - **Pipeline Context**: Landmark paper establishing how feature engineering from calling graphs, airtime top-ups, and spatial mobility predicts individual socio-economic status.

### 3. Human Mobility Motifs, Trajectory Mining & Telemetry Noise

- **Schneider, C. M., Belik, V., Couronné, T., Smoreda, Z., & González, M. C. (2013).** "Unravelling daily human mobility motifs."
  *Journal of The Royal Society Interface*, 10(84), 20130246.
  - **Status**: Q1 (Biophysics & Complex Systems)
  - **Public Link (DOI)**: https://doi.org/10.1098/rsif.2013.0246
  - **Open Access**: https://royalsocietypublishing.org/doi/10.1098/rsif.2013.0246
  - **Pipeline Context**: Identifies the 17 fundamental directed network motifs representing daily human tour structures from CDRs, linking motif diversity to occupation and travel behavior.

- **Pappalardo, L., Pedreschi, D., Smoreda, Z., & Giannotti, F. (2015).** "Using big data to study the link between human mobility and socio-economic status."
  *Journal of The Royal Society Interface*, 12(113), 20150597.
  - **Status**: Q1 (Complex Systems / Data Science)
  - **Public Link (DOI)**: https://doi.org/10.1098/rsif.2015.0597
  - **Open Access**: https://royalsocietypublishing.org/doi/10.1098/rsif.2015.0597
  - **Pipeline Context**: Empirical proof linking individual mobility metrics (radius of gyration, location entropy, return probability) directly to socio-economic status.

- **Jiang, S., Fiore, G. A., Yang, Y., Ferreira, J., Frazzoli, E., & González, M. C. (2017).** "A review of urban computing for mobile phone traces: Current methods, challenges and opportunities."
  *IEEE Transactions on Intelligent Transportation Systems*, 18(4), 779–796.
  - **Status**: Q1 (Transportation Science & Technology / Intelligent Systems)
  - **Public Link (DOI)**: https://doi.org/10.1109/TITS.2016.2572716
  - **Open Access (MIT DSpace)**: https://dspace.mit.edu/handle/1721.1/107770
  - **Pipeline Context**: Practical formulation for cellular ping-pong handover noise filtering and trajectory reconstruction implemented in `noise_reduction.py`.

- **Caceres, N., Wideberg, J. P., & Benitez, F. G. (2012).** "Review of traffic data obtaining from mobile phones."
  *IET Intelligent Transport Systems*, 6(1), 92–104.
  - **Status**: Q1 (Engineering / Transportation)
  - **Public Link (DOI)**: https://doi.org/10.1049/iet-its.2010.0154
  - **Pipeline Context**: Cellular signal oscillation and ping-pong handover suppression techniques for vehicular and pedestrian tracking.

- **Dong, Y., Yang, Y., Tang, J., Yang, Y., & Chawla, N. V. (2014).** "Inferring user demographics and social strategies in mobile social networks."
  *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '14)*, 771–780.
  - **Status**: Top-Tier Flagship (Core A*)
  - **Public Link (DOI)**: https://doi.org/10.1145/2623330.2623703
  - **Open Access (arXiv)**: https://arxiv.org/abs/1406.1417
  - **Pipeline Context**: Graph-based social homophily and network neighbor features for age, gender, and socio-economic prediction.

### 4. Probability Calibration & Imbalanced Class Learning

- **Niculescu-Mizil, A., & Caruana, R. (2005).** "Predicting good probabilities with supervised learning."
  *Proceedings of the 22nd International Conference on Machine Learning (ICML '05)*, 625–632.
  - **Public Link (DOI)**: https://doi.org/10.1145/1102351.1102430
  - **Pipeline Context**: Platt scaling and isotonic regression methods implemented in `train.py` to yield true posterior probabilities.

- **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** "On calibration of modern neural networks."
  *Proceedings of the 34th International Conference on Machine Learning (ICML '17)*, PMLR 70, 1321–1330.
  - **Open Access (arXiv)**: https://arxiv.org/abs/1706.04599
  - **Pipeline Context**: Calibration metric evaluations and temperature scaling principles for multi-class probability outputs.

- **He, H., & Garcia, E. A. (2009).** "Learning from imbalanced data."
  *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
  - **Status**: Q1 (Computer Science & Artificial Intelligence)
  - **Public Link (DOI)**: https://doi.org/10.1109/TKDE.2008.239
  - **Pipeline Context**: Theoretical analysis of how synthetic oversampling (SMOTE) shifts posterior log-odds, motivating the holdout calibration implemented in `train.py`.

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
