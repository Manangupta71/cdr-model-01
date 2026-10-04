# Data Lineage — Synthetic Bengaluru Generator

Every distributional assumption modeled in `generate_bengaluru_data.py` is systematically classified below:

- **GROUNDED** — Backed by empirical literature values, official standards, or established statistics.
- **PLAUSIBLE** — Modeling estimates chosen to reflect representative urban conditions.
- **SIMULATION COUPLING** — Explicit generative rules where attributes are structurally linked for simulation coherence.

This audit details the data generation mechanics and distinguishes empirically calibrated distributions from generative simulation rules.

| Component | Classification | Technical Description |
|---|---|---|
| Age group distribution | PLAUSIBLE | Reflects Bengaluru's working-age demographic profile (technology sector migration); synthetic distribution. |
| Gender split | PLAUSIBLE | Approximate 52/48 ratio representing typical metropolitan census proportions. |
| Education conditioned on age | PLAUSIBLE | Conditioned distribution reflecting higher educational attainment in younger adult cohorts. |
| Occupation conditioned on age + education | PLAUSIBLE | Conditioned distribution reflecting Bengaluru's professional, IT, and service industry employment base. |
| **work_status derived from occupation** | **SIMULATION COUPLING** | Rule-based mapping from occupation category with retirement condition for age 60+. Tracked separately in evaluation. |
| Income bracket conditioned on occupation + education | GROUNDED (direction) / PLAUSIBLE (magnitude) | Conditional distribution linking income level to occupational category and educational attainment. |
| `socio_demographic_class` (NCCS-style band) | GROUNDED (methodology) | Derived from household income bracket and education level following the New Consumer Classification System (NCCS) framework. |
| Home/work zone assignment | PLAUSIBLE | Spatial allocation weighted by residential density and employment density per urban zone (e.g., Whitefield, Electronic City, Koramangala). |
| Zone coordinates | GROUNDED (centroids) | Centroids representing major Bengaluru localities, geocoded within city bounds. |
| Antenna (cell tower) layout | PLAUSIBLE | Spatial tower distribution weighted by local activity density, reflecting urban cellular deployment density. |
| Daily mobility schedule | PLAUSIBLE | Diurnal home–work–home commute and daytime activity schedule with stochastic deviation. |
| Ping (tower-fix) jitter | GROUNDED (calibrated) | Set to ~90m Gaussian standard deviation, ensuring coordinate dispersion remains within the spatial place-clustering threshold (`CLUSTER_DIST_M = 250m`) to reliably resolve home and work anchors. |
| Social network edge structure | PLAUSIBLE | Household, workplace, and weak-tie community graph structures generating realistic call/SMS interactions. |
| Communication event timing & durations | PLAUSIBLE | Poisson call generation and exponential call duration distributions. |
| **Recharge frequency and amount by income** | **GROUNDED (direction)** | Calibrated following Steele et al. (2017), *J. R. Soc. Interface*: lower-income users top up more frequently in smaller amounts. |
| **Social network centrality & reciprocity** | **GROUNDED (methodology)** | Ingests communication graph topology computing directed in/out degree, PageRank, clustering, and reciprocity (Eagle et al., 2010; Onnela et al., 2007). |
| **Daily human mobility motifs** | **GROUNDED (classification)** | Daily activity chains mapped to canonical motifs (H-W-H, H-W-O-H, H-O-H, complex) with Shannon entropy and commute regularity (Schneider et al., 2013; Jiang et al., 2016). |
| **Microscopic activity plans & mode choice** | **GROUNDED (framework)** | 24-hour activity-travel schedules and MATSim plans.xml generated from predicted demographics and inferred anchors (Bassolas et al., 2019; Axhausen & Horni, 2016). |

---

## Evaluation Implications

- **`work_status`**: Performance reflects the conditional rule applied during synthetic generation; evaluated and reported with this context noted.
- **Demographic Targets (`age_group`, `gender`, `education_level`, `occupation_category`, `income_bracket`, `socio_demographic_class`)**: Governed by multi-dimensional probabilistic sampling with stochastic noise, providing a realistic test of feature extraction and classification efficacy.

---

## Methodological Implementation Notes

1. **Stay-Point Extraction Algorithm**: Implements the four-step procedure from Toole et al. (2015), *Transportation Research Part C*, Algorithms 2–5 (candidate stays, grid-based agglomerative clustering, and final snapping pass).
2. **Elimination of Coordinate Leakage**: In real CDR deployments, ground-truth home and work coordinates are unobserved. `bandicoot_features.py` strictly consumes inferred home/work centroids from `label_places.py` (`output/home_work_anchors.csv`) rather than querying `population.csv`, closing the pipeline loop.
3. **Cellular Ping-Pong Handover Suppression**: Pre-processes raw antenna pings to eliminate high-frequency ping-pong bouncing between adjacent cellular towers prior to Kalman filtering (Jiang et al., 2017, *IEEE T-ITS*; Caceres et al., 2012, *IET-ITS*).
4. **Probability Calibration Post-SMOTE**: SMOTE balances training frequencies, which artificially distorts predicted posterior log-odds. `train.py` applies Platt scaling (sigmoid calibration) on an un-resampled holdout split (Niculescu-Mizil & Caruana, 2005; He & Garcia, 2009) to recover well-calibrated class probability vectors.
5. **Feature Schema Alignment**: Follows bandicoot's standard `metric__weekpart__daypart__channel[__stat]` naming convention across behavioral indicators, augmented by stay-point activity indicators (Alexander et al., 2015; Pappalardo et al., 2015).
6. **Downstream Population Synthesis (IPU)**: Seeds produced by `train.py` are expanded to match zonal census marginal distributions using Iterative Proportional Updating (`ipu.py`), validated with SRMSE and Total Absolute Difference metrics (Ye et al., 2009; Sun & Erath, 2015).
7. **Social & Spatial Co-Location Graphs**: `graph_features.py` extracts directed communication topology (PageRank, degree ratio, reciprocity) and spatial encounter degrees across pseudonymized subscriber identifiers (*Eagle et al., 2010, Science*; *Dong et al., 2014, KDD*).
8. **Daily Human Mobility Motifs**: `motifs.py` classifies daily travel chains into topological motifs (H-W-H, H-W-O-H, H-O-H, complex) and computes Shannon motif entropy and commute regularity (*Schneider et al., 2013, J. R. Soc. Interface*; *Jiang et al., 2016, PNAS TimeGeo*).
9. **Bayesian Dependency DAG & Classifier Chains**: `train.py` structures prediction along the demographic DAG ($\text{Age} \to \text{Gender} \to \text{Education} \to \text{Occupation} \to \text{Work Status} \to \text{Income} \to \text{Class}$) using out-of-fold probability vectors to ensure attribute consistency (*Sun & Erath, 2015, TR-C*; *Read et al., 2011, Machine Learning*).
10. **Agent-Based Traffic Micro-Simulation Inputs**: `matsim_plans.py` synthesizes 24-hour activity-travel diaries and standard MATSim `plans.xml` from predicted demographics and spatial anchors (*Bassolas et al., 2019, TR-A*; *Axhausen & Horni, 2016*).
11. **Anonymized Data Compliance**: All pipeline components operate strictly on hashed/pseudonymized strings (`phone_number`), guaranteeing zero dependency on sequential integer IDs or plaintext PII.

---

## Path to Empirical Deployment

Transitioning from synthetic simulation to operator-grade Call Detail Records requires:

1. **Probabilistic Record Linkage**: Employing Fellegi-Sunter record linkage (Fellegi & Sunter, 1969) to align pseudo-anonymized subscriber IDs with ground-truth survey samples.
2. **Ground-Truth Calibration**: Aligning behavioral indicators against representative household travel survey datasets for external validation.
3. **Zonal Marginal Constraint Matching**: Applying `ipu.py` against ward-level census tables (e.g. Census of India 2011 Primary Census Abstract or BBMP administrative wards) for travel demand model calibration.

