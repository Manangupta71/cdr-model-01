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

Constructs three distinct user graphs from anonymized Call Detail Records and trajectory telemetry, extracting topological metrics and neighbor behavioral feature aggregations:

- **Academic Grounding**:
  - Eagle, N., Macy, M., & Claxton, R. (2010). "Network diversity and economic development." *Science*, 328(5981), 1029–1031. https://doi.org/10.1126/science.1186605
  - Onnela, J.-P. et al. (2007). "Structure and tie strengths in mobile communication networks." *PNAS*, 104(18), 7332–7336. https://doi.org/10.1073/pnas.0610245104
  - Dong, Y. et al. (2014). "Inferring social ties across heterogenous networks." *ACM SIGKDD*, 771–780. https://doi.org/10.1145/2623330.2623703
  - Blumenstock, J., Cadamuro, G., & On, R. (2015). "Predicting poverty and wealth from mobile phone metadata." *Science*, 350(6264), 1073–1077. https://doi.org/10.1126/science.aac4420
- **The Three User Graphs**:
  1. **Social Interaction Graph ($G_{\text{social}}$)**: Built from aggregated telecommunication logs (`caller_id` $\to$ `callee_id`). Extracts in/out degree, weighted strength, PageRank centrality, local clustering coefficient, and mutual reciprocity.
  2. **Spatial Co-Location Graph ($G_{\text{colocation}}$)**: Links subscribers who simultaneously occupy the same spatial grid cells ($\le 300\text{ m}$) within overlapping temporal windows ($\le 1\text{ hour}$) using stay points. Computes encounter degree, encounter strength, and clustering coefficient.
  3. **Mobility Similarity Graph ($G_{\text{mobility}}$)**: Built via $k$-Nearest Neighbors ($k=5$) over normalized spatial and mobility indicators (commute distance, radius of gyration, location entropy). Computes degree, similarity strength, and local clustering coefficient.
- **Neighbor Behavioral Feature Aggregation (Zero-Leakage Protocol)**:
  - For each graph, computes the mean and standard deviation of peer neighbors across 5 continuous behavioral indicators:
    - `commute_distance_m`
    - `entropy_of_places`
    - `radius_of_gyration__allweek__allday`
    - `number_of_interactions__allweek__allday__call`
    - `amount_recharges__allweek__allday__mean`
  - **Strict Zero-Leakage Guarantee**: Under no circumstances are demographic labels used as features. During cross-validation, the mobility $k$-NN index is fit strictly on training fold nodes via `MobilityKNNGraphBuilder`.
- **Output**: Generates `output/graph_features.csv` (45 graph indicators per subscriber).

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

Trains CatBoost classifiers across seven socio-demographic targets, evaluates systematic feature ablations, and performs multi-seed validation:

- **Academic Grounding**:
  - Read, J. et al. (2011). "Classifier chains for multi-label classification." *Machine Learning*, 85(3), 333–359. https://doi.org/10.1007/s10994-011-5256-5
  - Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good probabilities with supervised learning." *ICML '05*. https://doi.org/10.1145/1102351.1102430
  - He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." *IEEE TKDE*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
- **Ablation Architecture**:
  Evaluates 5 feature configurations on identical 5-fold stratified cross-validation splits across multiple random seeds (default: seeds 42, 123, 999):
  1. `baseline`: Tabular Bandicoot indicators + commute anchors + mobility motifs
  2. `baseline+mobility`: Baseline + Mobility Similarity Graph ($k$-NN)
  3. `baseline+colocation`: Baseline + Co-location Graph
  4. `baseline+social`: Baseline + Social Interaction Graph
  5. `baseline+all_graphs`: Baseline + All Three User Graphs
- **Bayesian Dependency DAG & Out-of-Fold Chaining**:
  - Targets are chained sequentially: $\text{Age} \to \text{Gender} \to \text{Education} \to \text{Occupation} \to \text{Work Status} \to \text{Income} \to \text{Socio-Demographic Class}$.
  - Out-of-fold predicted class probability vectors from parent models are injected as conditioning features for downstream targets, preventing target leakage while enforcing inter-attribute joint consistency.
- **SMOTE & Platt Probability Calibration**:
  - SMOTE balances training fold distributions.
  - Platt scaling (`CalibratedClassifierCV(method="sigmoid", cv="prefit")`) fit on un-resampled holdouts corrects distorted posteriors to true empirical priors. Both uncalibrated and calibrated Macro-F1 and Brier scores are tracked.
- **Outputs**:
  - `output/predictions.csv`: Production predictions with point labels and full calibrated posterior probability distributions per subscriber.
  - `output/ablation_summary.csv`: Macro-F1 mean, fold std, seed spread, and Brier scores across all ablation variants.
  - `output/results_summary.csv`: Head-to-head comparison of baseline vs. all-graphs models.
  - `output/classification_reports.txt`: Detailed per-class precision, recall, and F1 reports.

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
