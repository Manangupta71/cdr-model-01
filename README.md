# CDR-Based Socio-Demographic Prediction for Synthetic Population Generation

A pipeline that simulates realistic Call Detail Record (CDR) data for a
synthetic Bengaluru population, extracts the standard `bandicoot`
behavioral-indicator feature set, and trains a model to predict
socio-demographic attributes per phone number — producing an output file
formatted for use as seed data in **synthetic population generation**
(the step that feeds trip generation in the classical four-step travel
demand model).

> **Reconstruction notice**: this repository is a rebuild of an SRFP
> fellowship project (IIT Bombay, guided by Dr. Archak Mittal), from
> documented architecture, methods, and the papers supplied as the
> project's base references — not a recovered copy of the original code.
> See `docs/DEVELOPER_HANDBOOK.md` for what changed and why.

## What this actually delivers

The one thing this project has to produce is **a model that predicts
socio-demographics from CDR behavioral features, with output files ready
for synthetic population generation**. Everything else in this repo is
scaffolding that produces that model's input:

```
generate_bengaluru_data.py  → raw CDR-like records (calls, texts, GPS-like
                               pings, antennas, prepaid recharges)
noise_reduction.py          → Kalman/particle-filtered pings
stay_points.py              → discrete stay points per person
label_places.py             → home / work / other labels per stay
bandicoot_features.py       → the standard 868-column behavioral
                               indicator table (one row per phone_number)
train.py                    → THE DELIVERABLE: trains + cross-validates
                               7 socio-demographic classifiers, then fits
                               final models and exports
                               output/synthetic_population_seed.csv
```

## Quick start

```bash
pip install -r requirements.txt

python generate_bengaluru_data.py   # -> data/population.csv, mobility_traces.csv,
                                     #    communication_events.csv, recharge_events.csv,
                                     #    antennas.csv
python noise_reduction.py           # -> data/mobility_traces_denoised.csv
python stay_points.py               # -> output/stay_points.csv
python label_places.py              # -> output/stay_points_labeled.csv,
                                     #    output/home_work_anchors.csv
python bandicoot_features.py        # -> output/bandicoot_features.csv (868 columns)
python train.py                     # -> output/results_summary.csv,
                                     #    output/classification_reports.txt,
                                     #    output/synthetic_population_seed.csv  <- the deliverable
```

Each script picks up its input from `data/`/`output/` automatically, and
`stay_points.py` / `bandicoot_features.py` will use the denoised mobility
trace if present, falling back to the raw one otherwise.

**Runtime note**: `train.py` fits CatBoost (with per-fold SMOTE) 5 times
per target for cross-validation, plus one final full-data fit per target,
across 7 targets — expect several minutes at the default population size
(5,000 agents), scaling with `N_AGENTS` in `generate_bengaluru_data.py`.

## The output that matters: `synthetic_population_seed.csv`

One row per `phone_number`. For each of the 7 targets, two kinds of
columns:

- `predicted_<target>` — the model's single best-guess category
  (e.g. `predicted_income_bracket = "upper_mid"`)
- `prob_<target>__<class>` — the full predicted probability distribution
  over that target's categories (e.g. `prob_income_bracket__low`,
  `prob_income_bracket__lower_mid`, ...)

Both forms are provided because population-synthesis tools differ in
what they expect: some (e.g. simple sample-and-assign approaches) just
need the hard label; others (e.g. IPU/IPF-style reweighting against
zonal marginal control totals) want the full probability vector so a
synthetic individual can be drawn from it rather than assigned
deterministically.

## Targets

`age_group`, `gender`, `education_level`, `occupation_category`,
`work_status`, `income_bracket`, `socio_demographic_class`

> `work_status` is a **known generator artifact** — it's derived
> near-deterministically from occupation in the synthetic generator, so
> its high F1 reflects that shortcut, not genuine predictive difficulty.
> See `docs/DATA_LINEAGE.md`.

## Documentation map

- **This file** — what the pipeline is, how to run it, what it outputs.
- **`docs/DEVELOPER_HANDBOOK.md`** — architecture internals, the
  algorithm choices behind each module, known limitations, and where to
  extend the code.
- **`docs/DATA_LINEAGE.md`** — every distributional assumption in the
  synthetic generator, classified grounded / plausible / artifact.
- **`docs/REFERENCES.md`** — every citation, with public links, mapped to
  the specific module it backs — including the IIT Bombay Civil
  Engineering department papers relevant to this project.

## Status

This SRFP project's own research report (Mumbai-based) was submitted and
accepted. This repository is a from-scratch rebuild, retargeted to
Bengaluru, scoped specifically to the socio-demographic prediction model
and its synthetic-population-ready output.
