# CDR-Based Socio-Demographic Prediction for Synthetic Population Generation

A computational pipeline that simulates realistic Call Detail Record (CDR) data for an urban population (Bengaluru), extracts the standard `bandicoot` behavioral-indicator feature set, and trains machine learning models to predict individual socio-demographic attributes — producing structured seed profiles for **synthetic population generation** in travel demand modeling.

> **Project Context**: Developed as part of the Summer Research Fellowship Programme (SRFP) at the Indian Institute of Technology Bombay (IIT Bombay), Department of Civil Engineering (Transportation Systems Engineering), under the guidance of Dr. Archak Mittal.

---

## Pipeline Architecture

The pipeline models the complete path from raw telecommunications telemetry to calibrated population synthesis seeds:

```
generate_bengaluru_data.py  → Raw CDR telemetry (voice calls, SMS, spatial pings,
                               cell towers, prepaid airtime recharges)
noise_reduction.py          → Trajectory noise filtering via Kalman / Particle filters
stay_points.py              → Spatial-temporal stay point extraction (Toole et al., 2015)
label_places.py             → Meaningful place inference (Home / Work / Other anchors)
bandicoot_features.py       → 868-column behavioral indicator extraction per subscriber
train.py                    → 5-fold cross-validation & final model training across 7
                              socio-demographic targets; exports synthetic population seeds
```

---

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Pipeline Execution

Run the pipeline sequentially from raw data simulation to final inference:

```bash
# 1. Generate synthetic CDR telemetry
python generate_bengaluru_data.py   # -> data/population.csv, mobility_traces.csv,
                                     #    communication_events.csv, recharge_events.csv,
                                     #    antennas.csv

# 2. Apply trajectory noise reduction
python noise_reduction.py           # -> data/mobility_traces_denoised.csv

# 3. Detect stay points
python stay_points.py               # -> output/stay_points.csv

# 4. Infer home and work anchors
python label_places.py              # -> output/stay_points_labeled.csv,
                                     #    output/home_work_anchors.csv

# 5. Extract 868 bandicoot behavioral indicators
python bandicoot_features.py        # -> output/bandicoot_features.csv

# 6. Train classifiers and export synthetic population seed
python train.py                     # -> output/results_summary.csv,
                                     #    output/classification_reports.txt,
                                     #    output/synthetic_population_seed.csv
```

Each stage automatically loads its corresponding input from `data/` and `output/`. Both `stay_points.py` and `bandicoot_features.py` detect and prioritize the denoised mobility traces if available.

> **Execution Note**: `train.py` executes 5-fold stratified cross-validation with per-fold SMOTE oversampling, followed by full-dataset training across all 7 socio-demographic targets. Runtime scales with population size (`N_AGENTS` in `generate_bengaluru_data.py`).

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

> **Methodological Note on `work_status`**: In the synthetic data generator, `work_status` is conditionally coupled with occupation category; cross-validation performance reflects this structural alignment. See `docs/DATA_LINEAGE.md` for full discussion.

---

## Output Data Product: `synthetic_population_seed.csv`

The primary output file, `output/synthetic_population_seed.csv`, contains one record per `phone_number` with:

- `predicted_<target>`: Point prediction (most probable category).
- `prob_<target>__<class>`: Complete predicted probability distribution over all classes for that target.

This dual formulation supports standard population synthesis workflows:
- **Direct Assignment**: Discrete sampling based on hard predicted labels.
- **Reweighted Optimization (IPU / IPF)**: Fitting full probability vectors against zonal marginal control totals (e.g. ward-level census distributions) for agent-based transport simulations.

---

## Documentation

- **[`docs/DEVELOPER_HANDBOOK.md`](docs/DEVELOPER_HANDBOOK.md)** — Architectural design, module implementations, algorithmic trade-offs, and extension interfaces.
- **[`docs/DATA_LINEAGE.md`](docs/DATA_LINEAGE.md)** — Classification of distributional assumptions in the synthetic generator (empirically grounded vs. heuristic).
- **[`docs/REFERENCES.md`](docs/REFERENCES.md)** — Methodological bibliography, academic literature citations, and related IIT Bombay transportation systems engineering research.
