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

---

## Evaluation Implications

- **`work_status`**: Performance reflects the conditional rule applied during synthetic generation; evaluated and reported with this context noted.
- **Demographic Targets (`age_group`, `gender`, `education_level`, `occupation_category`, `income_bracket`, `socio_demographic_class`)**: Governed by multi-dimensional probabilistic sampling with stochastic noise, providing a realistic test of feature extraction and classification efficacy.

---

## Methodological Implementation Notes

1. **Stay-Point Extraction Algorithm**: Implements the four-step procedure from Toole et al. (2015), *Transportation Research Part C*, Algorithms 2–5 (candidate stays, grid-based agglomerative clustering, and final snapping pass).
2. **Feature Schema Alignment**: Follows bandicoot's standard `metric__weekpart__daypart__channel[__stat]` naming convention across all 868 behavioral indicators.
3. **Numerical Stability in Moment Calculations**: Zero-variance and near-constant samples are handled explicitly to prevent NaN values in skewness and kurtosis calculations prior to SMOTE oversampling.

---

## Path to Empirical Deployment

Transitioning from synthetic simulation to operator-grade Call Detail Records requires:

1. **Probabilistic Record Linkage**: Employing Fellegi-Sunter record linkage (Fellegi & Sunter, 1969) to align pseudo-anonymized subscriber IDs with ground-truth survey samples.
2. **Ground-Truth Calibration**: Aligning behavioral indicators against representative household travel survey datasets for external validation.
