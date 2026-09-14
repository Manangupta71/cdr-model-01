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

Applies trajectory smoothing to raw mobility coordinates prior to spatial analysis:

- Constant-velocity 2D kinematic model implemented in planar coordinates (meters).
- `kalman_filter_trace`: Closed-form linear-Gaussian Kalman filter (default). Fast and optimal for Gaussian ping jitter.
- `particle_filter_trace`: Sequential Importance Resampling (SIR) bootstrap particle filter, providing robustness for non-Gaussian or multi-modal observation noise (Zheng, 2015).
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

Extracts the full 868-column behavioral indicator schema modeled after the `bandicoot` standard (de Montjoye et al., 2016):

- Temporal Partitions: Split across 3 week parts (`allweek`, `weekday`, `weekend`) $\times$ 3 day parts (`allday`, `day`, `night`).
- Channels: Interaction indicators computed across `call`, `text`, and combined `callandtext`.
- Statistical Moments: Distributional metrics expanded into 7 descriptive statistics (mean, std, median, skewness, kurtosis, min, max).
- Key Formatting: Internal indicator calculations are formatted into canonical column names (`metric__weekpart__daypart__channel[__stat]`).
- Operational Parameters: Inactivity threshold for conversation segmentation (>1hr), SMS response window (1hr), and Pareto cutoffs (80% concentration).

---

### `train.py`

Implements machine learning model training and inference across the 7 socio-demographic targets:

1. `train_and_evaluate()`:
   - 5-fold stratified cross-validation for out-of-fold generalization estimation.
   - SMOTE oversampling applied strictly within each training fold to address class imbalance without data leakage.
   - CatBoost classifier with early stopping against an internal validation split.
   - Outputs macro-F1, weighted-F1, and complete classification reports.

2. `train_final_model_and_predict()` / `export_for_population_synthesis()`:
   - Trains final production models on the complete dataset using CatBoost.
   - Predicts discrete classes (`predicted_<target>`) and full probability distributions (`prob_<target>__<class>`) for every subscriber.
   - Outputs `output/synthetic_population_seed.csv`.

---

## Graph-Based Feature Extensions

The pipeline's modular structure allows incorporating relational network features alongside `bandicoot_features.py`:

- **Spatial Proximity Network**: k-NN graph constructed across inferred home/work centroids (`scipy.spatial.cKDTree` or `sklearn.neighbors.NearestNeighbors`).
- **Co-location Network**: Bipartite projection linking subscribers who share temporal stay points.
- **Communication Social Network**: Weighted directed graph constructed from call/SMS logs.

Centrality measures (degree, PageRank, clustering coefficients) extracted from these graphs via `networkx` can be merged directly into the feature matrix using `phone_number`.

---

## Methodological Considerations and Future Work

1. **Decoupled Simulation Testing**: In synthetic environments, coupling between targets (e.g. `work_status` and `occupation`) should be validated against independently noisy distributions to benchmark sensitivity.
2. **Sample Stratification**: Real-world CDR data exhibits demographic representation biases (e.g., smartphone ownership disparities); poststratification weighting or multilevel regression can be integrated prior to model training.
3. **Cross-Seed Generalization**: Evaluating models trained on one synthetic population seed against distinct population realizations to verify parameter robustness.
4. **Empirical Linkage**: Integrating Fellegi-Sunter probabilistic record linkage for matching operator telemetry with ground-truth travel surveys.
