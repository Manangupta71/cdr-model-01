# Developer Handbook

Technical architecture, module specifications, algorithmic implementations, and extension interfaces for the CDR socio-demographic prediction pipeline.

---

## Module Specifications

### `generate_bengaluru_data.py`

Generates telecommunication data streams simulating mobile operator exports for an urban population:

- `generate_population()`: Multi-attribute demographic sampling with conditional dependencies (age $\rightarrow$ education $\rightarrow$ occupation $\rightarrow$ income/work status $\rightarrow$ socio-economic class), preserving empirical joint correlations.
  - **Extension Point**: Calibrating against NSS or Census microdata requires updating only this sampling function; all downstream modules consume `population.csv` generically.
- `generate_antenna_layout()` / `assign_antennas()`: Simulates cell tower distributions weighted by composite residential and employment activity, resolving antenna associations via a spatial KD-tree (`scipy.spatial.cKDTree`).
- `generate_mobility_traces()`: Diurnal activity schedules (home $\rightarrow$ commute $\rightarrow$ work $\rightarrow$ home) with Gaussian coordinate dispersion.
  - **Spatial Calibration**: Ping jitter standard deviation (~90m) is configured well within the place-clustering threshold (`CLUSTER_DIST_M = 250m` in `label_places.py`) to prevent home/work locations from fragmenting across observation days.
- `generate_communication_events()`: Simulates call and SMS records structured around household, workplace, and social network ties.
- `generate_recharge_events()`: Simulates prepaid airtime transactions calibrated to income brackets per Steele et al. (2017), capturing higher transaction frequency and smaller top-up amounts among lower-income brackets.
- **Parameter Resolution**: `generate_population(n_agents=None)` dynamically resolves default sizes inside the function body, allowing programmatic overrides without binding issues.

---

### `noise_reduction.py`

Applies cellular trajectory smoothing to raw mobility coordinates prior to spatial analysis:

- `filter_ping_pong_handovers`: Suppresses high-frequency cellular oscillation artifacts where subscribers bounce rapidly between adjacent towers ($A \to B \to A$) within a short temporal window (Jiang et al., 2017, IEEE T-ITS; Caceres et al., 2012, IET-ITS).
- Constant-velocity 2D kinematic model implemented in planar coordinates (meters).
- `kalman_filter_trace`: Closed-form linear-Gaussian Kalman filter (default). Fast and optimal for Gaussian ping jitter (Zheng, 2015).
- `particle_filter_trace`: Sequential Importance Resampling (SIR) bootstrap particle filter, providing robustness for non-Gaussian or multi-modal observation noise.
- Trajectory smoothing is isolated strictly per subscriber (`agent_id`).
- **Parameter Tuning**: `process_noise_std` (state dynamics) and `measurement_noise_std` (spatial observation noise) correspond directly to simulation physical dimensions.

---

### `stay_points.py`

Implements spatial-temporal stay-point detection based on Toole et al. (2015), Algorithms 2–5:

1. `_find_candidate_stays`: Iteratively evaluates consecutive pings within distance threshold $\Delta$ (`DELTA_M = 200m`). Candidate sets whose temporal duration exceeds $\tau$ (`TAU_SEC = 1200s` / 20 min) form candidate stays.
2. `_agglomerative_cluster`: Spatial grid-snapping (`GRID_SIZE_M = 200m`) agglomerates nearby candidate stays occurring at different times into unified stay locations.
3. `_final_pass`: Assigns unclustered intermediate pings to nearest established stay centroids within distance $\Delta$.

**Execution Flow**: `stay_points.py` prioritizes `data/mobility_traces_denoised.csv` when available (falling back to raw traces with a warning). Prior filtering prevents artificial stay fragmentation from telemetry noise.

---

### `label_places.py`

Implements semantic place categorization (Toole et al., 2015, Algorithm 6):

- Home Anchor: Identified as the stay location with maximum cumulative dwell time during weekday nighttime hours (20:00–07:00).
- Work Anchor: Identified as the non-home location with maximum cumulative dwell time during weekday daytime hours (07:00–20:00), constrained by minimum visit frequency ($\ge 1$ visit per week).
- Other: All remaining stay locations.
- **Diurnal Partitioning**: The 20:00–07:00 / 07:00–20:00 weekday boundary follows travel-demand origin-destination modeling literature, remaining distinct from bandicoot's 07:00–19:00 indicator bins.
- **Output Anchors**: Generates `output/home_work_anchors.csv` and `output/stay_points_labeled.csv`.

---

### `bandicoot_features.py`

Extracts the full behavioral indicator schema modeled after the `bandicoot` standard (de Montjoye et al., 2016) augmented with inferred spatial stay-point indicators (Alexander et al., 2015; Pappalardo et al., 2015):

- **Data Lineage Closure**: Ingests `output/home_work_anchors.csv` and `output/stay_points_labeled.csv` rather than ground-truth population coordinates, eliminating ground-truth leakage.
- **Inferred Activity Indicators**: Adds commute distance between inferred home and work centroids (`commute_distance_m`), work anchor presence (`has_inferred_work`), place count (`number_of_places`), place dwell-time entropy (`entropy_of_places`), and total/mean stay duration.
- Temporal Partitions: Split across 3 week parts (`allweek`, `weekday`, `weekend`) $\times$ 3 day parts (`allday`, `day`, `night`).
- Channels: Interaction indicators computed across `call`, `text`, and combined `callandtext`.
- Statistical Moments: Distributional metrics expanded into 7 descriptive statistics (mean, std, median, skewness, kurtosis, min, max).
- Key Formatting: Internal indicator calculations are formatted into canonical column names (`metric__weekpart__daypart__channel[__stat]`).

---

### `graph_features.py`

Extracts relational communication network topology and physical co-location encounter metrics on anonymized subscriber identifiers:

- **Academic Grounding**:
  - Eagle, N., Macy, M., & Claxton, R. (2010). "Network diversity and economic development." *Science*, 328(5981), 1029–1031. https://doi.org/10.1126/science.1186605
  - Onnela, J.-P. et al. (2007). "Structure and tie strengths in mobile communication networks." *PNAS*, 104(18), 7332–7336. https://doi.org/10.1073/pnas.0610245104
  - Dong, Y. et al. (2014). "Inferring social ties across heterogenous networks." *ACM SIGKDD*, 771–780. https://doi.org/10.1145/2623330.2623703
- **Network Topologies**:
  - **Directed Social Graph**: Built from aggregated communication records (`caller_phone` $\to$ `callee_phone`). Computes directed in-degree, out-degree, degree ratio, PageRank centrality, local clustering coefficient, and mutual reciprocity.
  - **Spatial Co-Location Graph**: Bipartite projection linking anonymized subscribers who simultaneously occupy the same stay points within overlapping temporal intervals ($\le 1$ hour). Computes encounter degree, total physical encounters, and co-location clustering coefficient.
- **Anonymization Assurance**: All graph projections preserve pseudonymous subscriber IDs without requiring any unmasked PII.
- **Output**: Generates `output/graph_features.csv` (11 graph indicators per subscriber).

---

### `motifs.py`

Extracts daily individual mobility motifs, tour entropy, and commute regularity:

- **Academic Grounding**:
  - Schneider, C. M. et al. (2013). "Unravelling daily human mobility motifs." *J. R. Soc. Interface*, 10(84), 20130246. https://doi.org/10.1098/rsif.2013.0246
  - Jiang, S. et al. (2016). "TimeGeo: a geospatially grounded framework for representing human mobility with daily activity patterns." *PNAS*, 113(37), E5378–E5387. https://doi.org/10.1073/pnas.1524261113
- **Motif Classification**:
  - Classifies daily stay sequences into canonical motif topologies:
    - **Motif 1 (H-W-H)**: Standard simple commute between Home and Work.
    - **Motif 2 (H-W-O-H)**: Work commute with intermediate secondary errand/activity.
    - **Motif 3 (H-O-H)**: Home to non-work activity and return.
    - **Motif 4 (Complex)**: Multi-stop tours with $\ge 4$ activities.
- **Indicators**:
  - `primary_motif_id`: Dominant daily tour topology.
  - `primary_motif_fraction`: Frequency of the dominant motif across observation days.
  - `motif_entropy`: Shannon entropy of daily motif distributions.
  - `commute_regularity`: Fraction of observation days with an inferred work visit.
  - `mean_daily_stops` & `distinct_motifs_count`: Daily tour complexity metrics.
- **Output**: Generates `output/mobility_motifs.csv` (6 motif indicators per subscriber).

---

### `train.py`

Implements multi-target **Conditional Classifier Chains** along the demographic dependency DAG with Platt probability calibration:

- **Academic Grounding**:
  - Read, J. et al. (2011). "Classifier chains for multi-label classification." *Machine Learning*, 85(3), 333–359. https://doi.org/10.1007/s10994-011-5256-5
  - Sun, L., & Erath, A. (2015). "A Bayesian network approach for population synthesis." *Transportation Research Part C*, 61, 49–62. https://doi.org/10.1016/j.trc.2015.10.010
  - Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good probabilities with supervised learning." *ICML '05*. https://doi.org/10.1145/1102351.1102430
  - He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." *IEEE TKDE*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
- **Unified Feature Space**: Merges baseline `bandicoot_features.csv`, relational `graph_features.csv`, and `mobility_motifs.csv`.
- **Bayesian Dependency DAG**:
  - Targets are chained sequentially: $\text{Age} \to \text{Gender} \to \text{Education} \to \text{Occupation} \to \text{Work Status} \to \text{Income} \to \text{Socio-Demographic Class}$.
  - During 5-fold cross-validation, out-of-fold predicted class probability vectors from parent models are injected as conditioning features for downstream targets, preventing target leakage while enforcing inter-attribute joint consistency.
- **SMOTE & Sigmoid Calibration**:
  - SMOTE balances training fold distributions.
  - Platt scaling (`CalibratedClassifierCV(method="sigmoid", cv="prefit")`) fit on un-resampled holdouts corrects distorted posteriors to true empirical priors.
- **Collinearity Pruning**: `VarianceThreshold(threshold=1e-5)` removes zero-variance features before tree construction.
- **Outputs**: Generates `output/results_summary.csv`, `output/classification_reports.txt`, and `output/synthetic_population_seed.csv`.

---

### `ipu.py`

Downstream population synthesis module implementing Iterative Proportional Updating (IPU) (Ye et al., 2009; Sun & Erath, 2015):

- Ingests `output/synthetic_population_seed.csv`.
- Aligns individual multi-attribute probability vectors against zonal marginal control totals (e.g. ward-level census distributions for age, gender, education, and income).
- Iteratively adjusts agent weights to convergence.
- Evaluates goodness of fit using Standardized Root Mean Square Error (SRMSE) and Total Absolute Difference (TAD).
- Exports weighted expanded population seeds (`output/synthetic_population_final.csv`) and validation tables (`output/ipu_validation_metrics.csv`).

---

### `matsim_plans.py`

Synthesizes agent-based 24-hour activity-travel diaries and standard MATSim `plans.xml` for microscopic traffic simulation:

- **Academic Grounding**:
  - Bassolas, A. et al. (2019). "Mobile phone records to feed activity-based travel demand models: MATSim for studying a cordon toll policy in Barcelona." *Transportation Research Part A*, 121, 56–74. https://doi.org/10.1016/j.tra.2019.01.007
  - Hörl, S., & Balać, M. (2021). "Synthetic population and travel demand for Paris and Île-de-France based on open and public data." *Transportation Research Part C*, 130, 103291. https://doi.org/10.1016/j.trc.2021.103291
  - Axhausen, K. W., & Horni, A. (2016). *The Multi-Agent Transport Simulation MATSim*. Ubiquity Press. https://doi.org/10.5334/baw
- **Activity Generation**:
  - Merges expanded synthetic population with inferred spatial home/work anchors.
  - Models temporal activity schedules (departure times, work/education durations, evening return) with individual stochasticity.
  - Selects primary transportation mode (`car`, `pt`, `walk`) calibrated to predicted income bracket and vehicle access.
- **Outputs**:
  - `output/activity_travel_diaries.csv`: Relational table of scheduled activities per agent.
  - `output/plans.xml`: Fully valid W3C standard XML conforming to the MATSim DTD specification (`http://www.matsim.org/files/dtd/plans_v4.dtd`).

---

## Anonymized CDR Data Pipeline Compliance

The pipeline is explicitly engineered to operate in enterprise telecommunications environments under strict data privacy regulations (e.g. GDPR, DPDP Act):

1. **Pseudonymous Hashing**: Primary subscriber keys (`phone_number`, `subscriber_id`) are treated strictly as arbitrary string identifiers. The codebase makes no assumptions regarding phone number formats, sequential integer IDs, or unmasked subscriber PII.
2. **Aggregated Interaction Graphing**: Social ties and co-location matrices link pseudonymous hashes directly without joining identity registries.
3. **Differential Anchor Resolution**: Inferred home and work centroids represent spatial activity clusters, which can be aggregated or clipped to administrative census wards to eliminate pinpoint localization risks before downstream model ingestion.

---

## Methodological Considerations and Future Work

1. **Decoupled Simulation Testing**: In synthetic environments, coupling between targets (e.g. `work_status` and `occupation`) should be validated against independently noisy distributions to benchmark sensitivity.
2. **Sample Stratification**: Real-world CDR data exhibits demographic representation biases (e.g., smartphone ownership disparities); poststratification weighting or multilevel regression can be integrated prior to model training.
3. **Cross-Seed Generalization**: Evaluating models trained on one synthetic population seed against distinct population realizations to verify parameter robustness.
4. **Empirical Linkage**: Integrating Fellegi-Sunter probabilistic record linkage for matching operator telemetry with ground-truth travel surveys.
