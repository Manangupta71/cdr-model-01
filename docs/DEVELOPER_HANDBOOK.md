# Developer Handbook

This is the internals document — read `README.md` first for what the
pipeline does and how to run it. This file is for anyone who needs to
modify, extend, or debug the code.

## Module-by-module

### `generate_bengaluru_data.py`

Produces every raw record stream a real mobile network operator export
would contain, for a synthetic Bengaluru population:

- `generate_population()` — joint (not independent) demographic sampling.
  Each attribute is conditioned on the ones sampled before it (age →
  education → occupation → work_status/income → socioeconomic class), so
  correlations between attributes are realistic rather than accidental.
  **Extension point**: if you calibrate this against real NSS/census
  microdata (see `docs/DATA_LINEAGE.md`'s "path to real deployment"
  section), this is the only function that needs to change — everything
  downstream consumes `population.csv` generically.
- `generate_antenna_layout()` / `assign_antennas()` — scatters cell
  towers with density weighted by residential + employment activity,
  and assigns nearest-antenna IDs via a KD-tree (`scipy.spatial.cKDTree`)
  for speed.
- `generate_mobility_traces()` — a daily activity schedule (home → work
  → home on weekdays) with Gaussian ping jitter. **Known gotcha**: the
  jitter magnitude must stay well under the place-clustering threshold
  used in `label_places.py` (`CLUSTER_DIST_M`), or a single home
  fragments into multiple false "places" across nights — this happened
  during development (see `docs/DATA_LINEAGE.md`) and was fixed by
  tightening jitter from ~300m to ~90m.
- `generate_communication_events()` — builds a call/SMS stream with real
  social structure (household/coworker/weak-tie edges) rather than
  uniform random pairs, so downstream social-graph-style features (if
  reinstated — see "Dropped scope" below) have real community structure
  to find.
- `generate_recharge_events()` — prepaid top-up events, with frequency
  and amount tied to income bracket per Steele et al. (2017): poorer
  agents recharge more often, in smaller amounts.
- **Config-mutation gotcha**: `generate_population(n_agents=None)` (and
  similarly for mobility/comms/recharges) resolves `None` to the module
  global `N_AGENTS` INSIDE the function body, not as a default-argument
  value. This is deliberate — Python binds default argument values once,
  at function-definition time, so `def f(n=N_AGENTS)` silently ignores
  later changes to the module-level `N_AGENTS`. This bit us once during
  development; don't reintroduce it.

### `noise_reduction.py`

Kalman filter (default) and bootstrap particle filter (alternative),
both using a constant-velocity 2D motion model, operating in a local
planar meters projection (fine at city scale) rather than raw
lat/lon degrees. Runs per-agent — state must never be shared across
different people's traces.

**When to use which**: Kalman is faster and matches this project's own
ping-noise model (Gaussian jitter), so it's the default. Switch to
`method="particle"` if you change the noise model to something
heavy-tailed (e.g. occasional large tower-handoff errors) — see Zheng
(2015) in `docs/REFERENCES.md` for the tradeoff.

**Tuning**: `process_noise_std` and `measurement_noise_std` should
roughly track the actual jitter magnitude in `generate_bengaluru_data.py`
(`ping()`'s `jitter` parameter, converted from degrees to meters). If you
change the generator's jitter, revisit these.

### `stay_points.py`

Implements Toole et al. (2015)'s Algorithms 2-5 exactly (see module
docstring for the correspondence). Three internal steps:

1. `_find_candidate_stays` — grows a candidate set of consecutive pings
   while each stays within `DELTA_M` of the *previous ping in the set*
   (not the set's centroid — this matches the paper's specification).
   A candidate set becomes a candidate stay if its time span exceeds
   `TAU_SEC`.
2. `_agglomerative_cluster` — the paper leaves "agglomerative clustering"
   generic; this implementation uses grid-cell snapping (`GRID_SIZE_M`)
   as an efficient, deterministic stand-in — candidate stays landing in
   the same grid cell collapse into one stay point.
3. `_final_pass` — sweeps every ping not yet assigned to a stay and
   snaps it to the nearest stay within `DELTA_M`, if any.

**Run order matters**: this expects `data/mobility_traces_denoised.csv`
(falls back to the raw trace with a printed warning if absent). Running
it on noisy raw pings without the Kalman filter first will fragment
stays — this is exactly the failure mode described in
`docs/DATA_LINEAGE.md`.

### `label_places.py`

Implements Toole et al. (2015)'s OD Creation Algorithm, Step 1
(Home/Work Expansion), exactly: home = the stay visited most (by dwell
time) between 8pm–7am on weekdays; work = the highest-dwell *non-home*
stay between 7am–8pm on weekdays, nulled out if visited less than once
per week.

**Deliberately different time convention from `bandicoot_features.py`**:
this module's home/work windows (20:00–07:00 / 07:00–20:00, weekdays
only) are NOT the same as bandicoot's day/night split (07:00–19:00,
applied across all days). They serve different purposes — one labels
physical places, the other buckets interaction indicators — and forcing
them to share a boundary would misrepresent one paper's method as the
other's.

**Known heuristic noise**: on the 150-agent test population used during
development, work anchors were correctly inferred for ~85% of `employed`
agents and 100% of `student` agents, but also incorrectly inferred for
some `retired`/`unemployed` agents (who occasionally have a
high-weekday-daytime-dwell "other" location — e.g. a regular volunteering
spot — that the heuristic can't distinguish from work). This is an
honest limitation of the dwell-time heuristic, not a bug; if it matters
for your use case, cross-check `has_inferred_work` against `work_status`
before trusting an agent's work anchor.

### `bandicoot_features.py`

Reproduces the full 868-column bandicoot indicator schema (verified via
an exact sorted-diff against the target schema during development — see
`docs/DATA_LINEAGE.md`'s "corrections made" section for the column-order
bug that check caught).

**Key internal convention**: every per-indicator helper function builds
keys as `metric__channel` or `metric__channel__stat` (no week/day
tokens). The `_reorder_key()` helper then inserts `week_part`/`day_part`
right after the metric name to match the target schema's
`metric__weekpart__daypart__channel[__stat]` convention. This works
uniformly across every indicator because no metric name contains a
double underscore — if you add a new indicator, keep that invariant or
`_reorder_key` will misparse it.

**Approximated definitions** (bandicoot's own definitions for these
aren't fully public): conversation grouping (>1hr gap starts a new
conversation), text response window (reply must follow within 1hr to
count as "responded"), pareto thresholds (min. fraction of contacts/
antennas needed to reach 80% of the total), and `churn_rate` (week-over-
week relative change in interaction volume — a volatility measure, not
literal telecom churn). Each is flagged inline with a `# CHOICE:` comment
where it's genuinely ambiguous from the literature.

### `train.py`

Two distinct code paths, deliberately kept separate:

1. `train_and_evaluate()` — 5-fold stratified CV, per-fold SMOTE, for
   **reporting** (macro/weighted F1, full classification report). Never
   used to produce the final predictions — its only job is honest
   evaluation.
2. `train_final_model_and_predict()` /
   `export_for_population_synthesis()` — fits one model per target on
   **all** available data (SMOTE applied to the full training set, not
   per-fold), then predicts a label + full probability vector for every
   agent. This is the actual deliverable.

Keeping these separate matters: if you're tempted to "save time" by
reusing one of the 5 CV fold-models as the final model, don't — it would
have been trained on only 80% of the data and evaluated a different 20%.

**CatBoost settings** were reduced from the literature-typical
`iterations=500` to `iterations=200` purely for runtime — at 868
features × ~5,000 agents × 7 targets × 6 fits per target (5 CV folds + 1
final), the original setting pushed wall-clock time into the
15-20-minute range. Raise it back if you have the compute budget and
want marginally better-calibrated probabilities.

## Dropped scope (explicitly, not silently)

An earlier draft of this rebuild (targeting Mumbai) included a
three-graph feature-engineering module (`pipeline.py`): a mobility
k-NN similarity graph, a co-location graph, and a social interaction
graph, with structural features (degree, PageRank, clustering
coefficient) extracted from each. That module is **not included** in
this Bengaluru rebuild, per the explicit scope narrowing to "the model
that predicts socio-demographics for synthetic population generation."
If you want to reinstate it as an additional feature set alongside
`bandicoot_features.py`, the graph-construction logic (using `networkx`
and `scikit-learn`'s `NearestNeighbors`) is a natural fit to reintroduce
as a sibling module producing `output/graph_features.csv`, merged into
`train.py`'s feature table on `phone_number`/`agent_id`.

## Known limitations worth fixing next

1. **`work_status` artifact** — fix at the generator level (decouple it
   from occupation with its own independent noisy sampling), not just by
   flagging it, so it becomes a genuine test rather than a tautology.
2. **CDR sampling bias** — the synthetic population is a uniform random
   sample; real operator data isn't (skews by device ownership,
   socioeconomic group, etc.). A multilevel-regression-and-
   poststratification correction step belongs between
   `bandicoot_features.py` and `train.py` if this ever touches real data.
3. **Synthetic-to-synthetic generalization test** — train on one RNG
   seed's population, test on a population generated from a different
   seed, to catch generator-artifact targets systematically rather than
   by manual inspection (this is how `work_status` was caught, and it
   should have been automatic).
4. **Identity resolution and external validation** — see
   `docs/DATA_LINEAGE.md`'s "path to real deployment" section.
