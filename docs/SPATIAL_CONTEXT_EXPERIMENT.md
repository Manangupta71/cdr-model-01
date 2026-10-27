# Spatial Context & Spatial Leakage Experiment

## Purpose

This experiment evaluates:
1. Whether location-context features—specifically zone-level spatial attributes derived from inferred home/work anchors and street-level environmental indices (greenery via Excess Green index, edge density) from Google Street View—provide additional predictive signal over baseline behavioral indicators and user graph topologies.
2. How much spatial leakage exists in standard cross-validation schemes by comparing standard random stratified cross-validation against spatially grouped cross-validation (grouped by nearest zone of the inferred home anchor).

## Experimental Design & Variants

The ablation suite incorporates systematic spatial context variants on top of the baseline tabular feature set:

- **`baseline`**: Standard behavioral indicators from `bandicoot_features.py` + commute features from inferred anchors + mobility motifs from `motifs.py`.
- **`baseline+zone_ctx`**: Baseline augmented with zone-level density and employment weights, distance to zone center, and work-zone employment weight, computed exclusively from inferred anchors in `output/home_work_anchors.csv`.
- **`baseline+zone_ctx_shuffled`**: Negative control permuting zone context columns across agents while preserving exact marginal distributions and column means.
- **`baseline+streetview`** *(optional)*: Baseline augmented with in-memory Street View indices (mean Excess Green vegetation index, mean edge density) from `output/streetview_features.csv`.
- **`baseline+streetview_shuffled`** *(optional)*: Negative control permuting Street View features across agents.

### Cross-Validation Schemes

- **`random`** (`StratifiedKFold`): Default 5-fold stratified cross-validation. Agents from the same neighbourhood can appear in both training and test splits.
- **`grouped`** (`StratifiedGroupKFold`): Cross-validation grouped by the nearest zone of each subscriber's inferred home anchor. All subscribers residing in a given zone are strictly isolated to either the training split or the test split, preventing geographic autocorrelation and spatial leakage across splits.

## Why a Null Result is Expected on Default Data

In the default synthetic generator (`generate_bengaluru_data.py`), an agent's `home_zone` is sampled from residential density weights alone, completely independent of all socio-demographic attributes (age, gender, education, occupation, income bracket, and socioeconomic class).

Because there is zero underlying statistical dependency between residential locality and subscriber demographics by construction, location-derived spatial context features are expected to yield no predictive gain over baseline behavioral indicators on the default dataset.

## Synthetic Positive Control (`--zone_income_coupling`)

To confirm that the pipeline and models have the statistical power to detect genuine geographic sorting when it exists, `generate_bengaluru_data.py` provides an optional coupling parameter:

```bash
python generate_bengaluru_data.py --zone_income_coupling 1.0
```

When `zone_income_coupling` ($\beta > 0$) is enabled, income probabilities are tilted based on an arbitrary, fixed zone affluence value:
$$\tilde{P}(\text{income} \mid \text{occ}, \text{edu}, \text{zone}) \propto P(\text{income} \mid \text{occ}, \text{edu}) \cdot \exp(\beta \cdot \text{affluence}_{\text{zone}} \cdot \mathbf{m})$$
where $\mathbf{m} = [-1.5, -0.5, 0.5, 1.5]$ spans low to high income tiers.

> **Important Disclaimer**: Zone affluence scores in this synthetic generator are seeded pseudorandom values instantiated strictly as an algorithmic positive control. They do not represent real-world neighbourhood income data or demographic claims about Bengaluru.

## How to Run

### 1. Default Pipeline (Null Hypothesis Benchmark)

Run the standard pipeline and evaluate under random or grouped cross-validation:

```bash
# Standard random cross-validation
python train.py --quick

# Grouped cross-validation (evaluating spatial leakage)
python train.py --cv_scheme grouped --quick
```

### 2. Optional Street View Probe

Run the Street View probe using an authorized Google Maps API key (or `--dry_run` for local simulation):

```bash
# Dry-run simulation (in-memory image generation, zero network calls, no API key needed)
python streetview_probe.py --dry_run --fetch

# Live metadata and in-memory image feature extraction (requires GOOGLE_MAPS_API_KEY)
export GOOGLE_MAPS_API_KEY="your-api-key"
python streetview_probe.py --fetch --max_requests 200 --rate_limit 0.1
```

Once `output/streetview_features.csv` is produced, `train.py` automatically incorporates the `baseline+streetview` and `baseline+streetview_shuffled` ablation variants.

### 3. Synthetic Coupled Benchmark (Power & Detectability Test)

Generate a coupled population to verify spatial context feature sensitivity:

```bash
python generate_bengaluru_data.py --zone_income_coupling 1.0
python noise_reduction.py
python stay_points.py
python label_places.py
python bandicoot_features.py
python graph_features.py
python motifs.py
python train.py --cv_scheme grouped
```
