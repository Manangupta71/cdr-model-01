# CDR-Based Socio-Demographic Prediction Pipeline

A computational pipeline that simulates realistic Call Detail Record (CDR) data for an urban population (Bengaluru), extracts the standard `bandicoot` behavioral-indicator feature set, engineers graph-derived relational features, and trains machine learning models to predict individual socio-demographic attributes.

> **Project Context**: Developed as part of the Summer Research Fellowship Programme (SRFP) at the Indian Institute of Technology Bombay (IIT Bombay), Department of Civil Engineering (Transportation Systems Engineering), under the guidance of Dr. Archak Mittal.

---

## Pipeline Architecture

The pipeline models the complete path from raw telecommunications telemetry to calibrated socio-demographic predictions:

```
generate_bengaluru_data.py  → Raw CDR telemetry (voice calls, SMS, spatial pings,
                              cell towers, prepaid airtime recharges)
noise_reduction.py          → Trajectory noise filtering via Ping-Pong Handover suppression
                              (Jiang et al., 2017) and kinematic Kalman / Particle filters
stay_points.py              → Spatial-temporal stay point extraction (Toole et al., 2015)
label_places.py             → Meaningful place inference (Home / Work / Other anchors)
bandicoot_features.py       → Behavioral indicator extraction consuming inferred anchors
                              (Alexander et al., 2015; de Montjoye et al., 2016)
graph_features.py           → Relational network topology & neighbor feature aggregates
                              (Eagle et al., 2010; Onnela et al., 2007; Dong et al., 2014)
motifs.py                   → Daily human mobility motifs, entropy, & commute regularity
                              (Schneider et al., 2013; Jiang et al., 2016)
spatial_context.py          → Zone-level residential and employment density context
                              derived strictly from inferred anchors
streetview_probe.py         → (Optional) In-memory Google Street View greenery (ExG)
                              and edge density feature extraction
train.py                    → 5-fold Stratified CV (random or grouped by inferred zone),
                              SMOTE, CatBoost classifiers, and Platt probability calibration
```

---

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Pipeline Execution

Run the pipeline sequentially from raw data simulation to model evaluation and prediction:

```bash
# 1. Generate synthetic CDR telemetry (default n=500, configurable)
python generate_bengaluru_data.py --n_agents 500  # -> data/population.csv, mobility_traces.csv,
                                                   #    communication_events.csv, recharge_events.csv,
                                                   #    antennas.csv

# 2. Apply trajectory noise reduction & ping-pong handover suppression
python noise_reduction.py                         # -> data/mobility_traces_denoised.csv

# 3. Detect stay points
python stay_points.py                             # -> output/stay_points.csv

# 4. Infer home and work anchors
python label_places.py                            # -> output/stay_points_labeled.csv,
                                                   #    output/home_work_anchors.csv

# 5. Extract behavioral indicators & commute features (using inferred anchors)
python bandicoot_features.py                      # -> output/bandicoot_features.csv

# 6. Extract relational communication network & co-location features
python graph_features.py                          # -> output/graph_features.csv

# 7. Extract daily mobility motifs & tour regularity
python motifs.py                                  # -> output/mobility_motifs.csv

# 8. (Optional) Probe Google Street View indices for inferred home anchors
python streetview_probe.py --dry_run --fetch      # -> output/streetview_features.csv

# 9. Train models, evaluate feature ablations & export predictions
python train.py                                   # Random 5-fold CV (default)
python train.py --cv_scheme grouped               # Grouped CV by inferred home zone
python train.py --quick                           # Single-seed verification
```

### Spatial Context & Leakage Evaluation

- **Spatial Context**: `spatial_context.py` extracts zone-level density and employment weights from inferred anchors, providing baseline comparison against spatial context features and negative controls (`baseline+zone_ctx_shuffled`).
- **Grouped CV Scheme**: Run `python train.py --cv_scheme grouped` to partition folds strictly by nearest home zone, testing against spatial leakage.
- **Positive Control**: Run `python generate_bengaluru_data.py --zone_income_coupling 1.0` to generate synthetic data where income depends on zone affluence, confirming detectability power.
- Detailed experimental methodology: see [SPATIAL_CONTEXT_EXPERIMENT.md](docs/SPATIAL_CONTEXT_EXPERIMENT.md).

Each stage automatically loads its corresponding input from `data/` and `output/`. The pipeline strictly preserves **anonymized subscriber IDs** across every module without assuming unmasked PII.

> **Execution Note**: `train.py` executes 5-fold stratified cross-validation with per-fold SMOTE oversampling and Platt probability calibration across all 7 socio-demographic targets. Runtime scales with population size (`N_AGENTS` in `generate_bengaluru_data.py`).

---

## Target Variables

The pipeline predicts seven socio-demographic dimensions:

1. `age_group` (18–24, 25–34, 35–44, 45–59, 60+)
2. `gender` (male, female)
3. `education_level` (primary, secondary, graduate, postgraduate)
4. `occupation_category` (informal labor, retail/service, clerical, professional, unemployed, student)
5. `work_status` (employed, unemployed, student, retired)
6. `income_bracket` (low, lower_mid, upper_mid, high)
7. `socio_demographic_class` (NCCS-style socio-economic grades E through A)

---

## Empirical Evaluation & Graph Feature Ablation

All models are evaluated using 5-fold stratified cross-validation across three distinct random seeds (`42`, `123`, `999`). Results compare the tabular baseline (`bandicoot` indicators + inferred commute anchors + mobility motifs) against models augmented with graph-derived features under a strict zero-leakage protocol:

| Target Variable | Baseline Macro-F1 | Baseline + All Graphs Macro-F1 | $\Delta$ Macro-F1 | Finding / Status |
|---|---|---|---|---|
| `age_group` | 0.3093 ± 0.0085 | **0.3522 ± 0.0070** | **+0.0430 (+13.9%)** | Substantial improvement from peer network homophily |
| `education_level` | 0.3048 ± 0.0052 | **0.3186 ± 0.0103** | **+0.0139 (+4.6%)** | Positive gain across random seeds |
| `income_bracket` | 0.8451 ± 0.0065 | **0.8462 ± 0.0050** | **+0.0011** | High baseline driven by recharge expenditure profiles |
| `occupation_category` | 0.4421 ± 0.0134 | 0.4297 ± 0.0099 | -0.0124 | Best single graph: Mobility k-NN (0.4465 ± 0.0135) |
| `socio_demographic_class` | 0.3937 ± 0.0022 | 0.3837 ± 0.0182 | -0.0100 | Best single graph: Social interaction (0.3846 ± 0.0023) |
| `gender` | 0.4887 ± 0.0174 | 0.4785 ± 0.0255 | -0.0102 | Stable near-parity |
| `work_status`* | 0.6704 ± 0.0162 | 0.6643 ± 0.0181 | -0.0061 | *Simulation coupling artifact (see note below) |

> **Methodological Note on `work_status`**: In the synthetic data generator, `work_status` is conditionally coupled with occupation category; cross-validation performance reflects this structural alignment. It is tracked separately and not presented as an algorithmic win. See `docs/DATA_LINEAGE.md`.

---

## Output Data Products

### 1. Subscriber Predictions: `output/predictions.csv`
Contains one record per subscriber (`phone_number`) with:
- `predicted_<target>`: Point prediction (most probable category).
- `prob_<target>__<class>`: Calibrated predicted probability distribution over all classes for that target.

### 2. Evaluation & Ablation Summaries: `output/results_summary.csv` & `output/ablation_summary.csv`
- `output/results_summary.csv`: Target-level performance metrics (Macro-F1, Micro-F1, per-fold standard deviation).
- `output/ablation_summary.csv`: Comparative evaluation of feature subsets across seeds:
  - Baseline (Bandicoot + Commute features)
  - Baseline + Mobility Similarity Graph
  - Baseline + Co-location Graph
  - Baseline + Social Interaction Graph
  - Baseline + All Three Graphs

### 3. Detailed Reports: `output/classification_reports.txt`
Full classification metrics (precision, recall, F1-score, support) for each target and demographic class.

### 4. Relational & Topological Indicator Tables
- `output/graph_features.csv`: Communication network centrality (PageRank, degree, reciprocity) and physical co-location metrics.
- `output/mobility_motifs.csv`: Daily motif indicators (primary motif ID, frequency, Shannon entropy, commute regularity).

---

## Documentation

- **[`docs/DEVELOPER_HANDBOOK.md`](docs/DEVELOPER_HANDBOOK.md)** — Architectural design, module implementations, algorithmic trade-offs, and extension interfaces.
- **[`docs/DATA_LINEAGE.md`](docs/DATA_LINEAGE.md)** — Classification of distributional assumptions in the synthetic generator (empirically grounded vs. heuristic).
- **[`docs/REFERENCES.md`](docs/REFERENCES.md)** — Methodological bibliography, academic literature citations, and related IIT Bombay transportation systems engineering research.
