# Data Lineage — Synthetic Bengaluru Generator

Every distributional assumption baked into `generate_bengaluru_data.py` is
classified below as one of:

- **GROUNDED** — backed by a cited literature value or well-known statistic.
- **PLAUSIBLE** — a reasonable modeling choice, not directly sourced.
- **UNGROUNDED / ARTIFACT** — acknowledged as a convenience of the
  generator that a downstream model could exploit in a way that would
  not transfer to real CDR data.

This document exists because an earlier iteration of the synthetic data
(for a different city) was characterized as "smoke and mirrors" during
review — the point of this audit is to make every shortcut visible
rather than hidden inside plausible-looking code, so results can be
reported honestly.

| Component | Classification | Note |
|---|---|---|
| Age group distribution | PLAUSIBLE | Skews slightly younger than the earlier Mumbai version, reflecting Bengaluru's IT-migration demographic; not census-sourced. |
| Gender split | PLAUSIBLE | Near 52/48, not drawn from a specific survey. |
| Education conditioned on age | PLAUSIBLE | Younger-skews-more-educated direction is well established; exact conditional probabilities are estimates. |
| Occupation conditioned on age + education | PLAUSIBLE | Skewed further toward "professional" than the Mumbai version, reflecting Bengaluru's IT-sector employment base; not calibrated to labor-force survey data. |
| **work_status derived from occupation** | **ARTIFACT** | Deterministic mapping (student->student, unemployed->unemployed, else->employed, with a retirement override). Flagged in `train.py` as a known-artifact target — do not report its F1 as a genuine result. |
| Income bracket conditioned on occupation + education | GROUNDED (direction) / PLAUSIBLE (magnitude) | Income rising with occupational prestige and education is well-supported directionally. |
| `socio_demographic_class` (NCCS-style band) | GROUNDED (direction) | Derived directly and only from income bracket + education, matching how NCCS classification is actually constructed (this was a leakage fix carried over from the original Mumbai version, where the class had been derived from a behavioral propensity score instead). |
| Home/work zone assignment | PLAUSIBLE | Weighted by illustrative residential/employment density values per zone (Whitefield, Electronic City, Marathahalli, and BKC-equivalent commercial cores get higher employment weight); not actual BBMP/BDA land-use data. |
| Zone coordinates | GROUNDED (place names) / PLAUSIBLE (exact centroid) | Real named Bengaluru localities (Koramangala, Indiranagar, Whitefield, Electronic City, etc.); approximate, not survey-grade geocoding. |
| Antenna (cell tower) layout | PLAUSIBLE | Scattered with density proportional to residential + employment weight, matching general urban tower-siting practice; not a real BBNL/operator tower inventory. |
| Daily mobility schedule (commute timing, dwell noise) | PLAUSIBLE | Standard home->work->home template with jitter; does not model real trip-chaining, multi-stop commutes, or seasonal variation. |
| Ping (tower-fix) jitter | GROUNDED (tightened after testing) | An earlier version used ~300m jitter for home/work pings, which exceeded the place-clustering threshold and fragmented a single home into multiple false "places" across nights. Tightened to ~90m and re-verified: home/work anchors now resolve correctly for the large majority of employed/student agents. |
| Household/coworker/weak-tie social graph structure | PLAUSIBLE | Chosen to give the social graph genuine community structure (vs. uniform random edges); edge-density parameters are not calibrated to a real telecom dataset. |
| Communication event timing/duration distributions | PLAUSIBLE | Poisson event counts and exponential call durations are standard modeling choices, not fit to a specific real dataset. |
| **Recharge (top-up) frequency and amount by income bracket** | **GROUNDED (direction)** | Steele et al. (2017), *J. R. Soc. Interface* — lower-income users top up more frequently, in smaller amounts, rather than less often in larger ones. Encoded directly in `RECHARGE_PROFILE`. |

## Known downstream consequences

- **`work_status`** results should always be reported with the artifact
  caveat attached.
- All other targets (`age_group`, `gender`, `education_level`,
  `occupation_category`, `income_bracket`, `socio_demographic_class`)
  are driven by probabilistic sampling with genuine noise, so their F1
  scores are informative about how well the feature-extraction approach
  can pick up the signal — though still on synthetic, not real, data.

## Corrections made during development (kept visible, not hidden)

1. **Stay-point algorithm mismatch**: an earlier draft of `stay_points.py`
   paraphrased a generic distance/time-threshold algorithm loosely
   attributed to Li et al. (2008). The actual base paper supplied for
   this project — Toole et al. (2015), *Transportation Research Part C*
   — specifies a more complete four-step procedure (candidate stays,
   grid-based agglomerative clustering, and a final snapping pass). The
   module has been rewritten to match that paper's Algorithms 2-5
   exactly; see the module docstring for the correction note.
2. **Column ordering bug**: a bandicoot-style feature-schema check
   (comparing generated columns against the target schema) caught that
   week-part/day-part tokens were being appended at the end of each
   column name instead of right after the metric name. Fixed and
   re-verified as an exact match against the full 868-column target
   schema.
3. **NaN skew/kurtosis on near-constant samples**: scipy's skewness and
   kurtosis can return NaN for degenerate (near-zero-variance) samples,
   which broke SMOTE downstream. Now caught and zeroed explicitly.

## Path to real CDR deployment

Two open items still block moving this pipeline onto real operator CDR
data:

1. **Identity resolution**: replacing exact phone-number matching with a
   **probabilistic record linkage** approach (Fellegi & Sunter, 1969)
   suited to noisy real-world identifiers.
2. **External validation**: establishing a way to check inferred
   socio-demographics against ground truth without a synthetic label to
   fall back on (e.g. a small labeled survey sample).
