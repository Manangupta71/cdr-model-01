"""
ipu.py

Downstream Population Synthesis via Iterative Proportional Updating (IPU).

Expands the predicted individual socio-demographic seed profiles
(output/synthetic_population_seed.csv) to match aggregate zonal marginal
control totals (e.g., ward-level census distributions), generating calibrated
synthetic population weights for agent-based travel demand models (e.g. MATSim).

Academic References:
  - Ye, X., Konduri, K., Pendyala, R. M., Sana, B., & Waddell, P. (2009).
    "A methodology to match distributions of both household and person
    attributes in the generation of synthetic populations."
    Transportation Research Board 88th Annual Meeting, Washington, D.C.
  - Sun, L., & Erath, A. (2015).
    "A Bayesian network approach for population synthesis."
    Transportation Research Part C: Emerging Technologies, 61, 49–62.
    https://doi.org/10.1016/j.trc.2015.10.010
  - Müller, K., & Axhausen, K. W. (2011).
    "Hierarchical IPM: a new approach to population synthesis."
    Transportation Research Record, 2255(1), 10–18.
    https://doi.org/10.3141/2255-02
  - Hörl, S., & Balać, M. (2021).
    "Synthetic population and travel demand for Paris and Île-de-France
    based on open and public data."
    Transportation Research Part C: Emerging Technologies, 130, 103291.
    https://doi.org/10.1016/j.trc.2021.103291
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_synthetic_census_controls(target_population_total=50000):
    """
    Generates representative metropolitan marginal control totals
    (e.g., Bengaluru ward / zone level) for multi-attribute population synthesis.
    """
    controls = {
        # Age group marginal distribution
        "age_group__18-24": int(0.24 * target_population_total),
        "age_group__25-34": int(0.30 * target_population_total),
        "age_group__35-44": int(0.21 * target_population_total),
        "age_group__45-59": int(0.16 * target_population_total),
        "age_group__60+":   target_population_total - sum(
            [int(p * target_population_total) for p in [0.24, 0.30, 0.21, 0.16]]
        ),

        # Gender marginal distribution
        "gender__male":   int(0.52 * target_population_total),
        "gender__female": target_population_total - int(0.52 * target_population_total),

        # Education level marginal distribution
        "education_level__primary":       int(0.12 * target_population_total),
        "education_level__secondary":     int(0.28 * target_population_total),
        "education_level__graduate":      int(0.38 * target_population_total),
        "education_level__postgraduate": target_population_total - sum(
            [int(p * target_population_total) for p in [0.12, 0.28, 0.38]]
        ),

        # Income bracket marginal distribution
        "income_bracket__low":       int(0.25 * target_population_total),
        "income_bracket__lower_mid": int(0.35 * target_population_total),
        "income_bracket__upper_mid": int(0.25 * target_population_total),
        "income_bracket__high":      target_population_total - sum(
            [int(p * target_population_total) for p in [0.25, 0.35, 0.25]]
        ),
    }
    return controls


def iterative_proportional_updating(seed_df, controls_dict, max_iterations=100, tolerance=1e-4):
    """
    Runs Iterative Proportional Updating (Ye et al., 2009; Sun & Erath, 2015)
    to calculate expansion weights for each seed agent.

    Parameters:
        seed_df: DataFrame with predicted class probabilities or discrete predictions.
        controls_dict: Dict of {constraint_column_or_key: target_total}.
        max_iterations: Maximum IPU iterations.
        tolerance: Convergence tolerance (max weight shift).

    Returns:
        weights: 1D array of non-negative expansion weights.
        metrics_df: DataFrame with convergence and constraint matching summary.
        meta: Summary dictionary with convergence stats, SRMSE, and TAD.
    """
    n_agents = len(seed_df)
    weights = np.ones(n_agents, dtype=float)

    # Build attribute incidence matrix X (n_agents x n_constraints)
    constraints = list(controls_dict.keys())
    target_totals = np.array([controls_dict[k] for k in constraints], dtype=float)
    X = np.zeros((n_agents, len(constraints)), dtype=float)

    for col_idx, key in enumerate(constraints):
        target, cls = key.split("__")
        prob_col = f"prob_{target}__{cls}"
        pred_col = f"predicted_{target}"

        if prob_col in seed_df.columns:
            X[:, col_idx] = seed_df[prob_col].to_numpy()
        elif pred_col in seed_df.columns:
            X[:, col_idx] = (seed_df[pred_col].astype(str) == str(cls)).astype(float).to_numpy()
        else:
            raise KeyError(f"Neither '{prob_col}' nor '{pred_col}' found in seed dataframe.")

    # IPU Iterations
    converged = False
    for iteration in range(1, max_iterations + 1):
        prev_weights = weights.copy()

        for j in range(len(constraints)):
            # Weighted total for constraint j
            contrib = X[:, j]
            weighted_sum = np.sum(weights * contrib)

            if weighted_sum > 0:
                adjustment = target_totals[j] / weighted_sum
                # Update weights for agents with positive contribution
                weights = weights * ((1.0 - contrib) + contrib * adjustment)

        # Check convergence
        max_rel_change = np.max(np.abs(weights - prev_weights) / np.maximum(prev_weights, 1e-8))
        if max_rel_change < tolerance:
            converged = True
            break

    # Compute final fitted totals and goodness-of-fit metrics
    fitted_totals = np.sum(weights[:, None] * X, axis=0)
    abs_errors = np.abs(fitted_totals - target_totals)
    pct_errors = abs_errors / np.maximum(target_totals, 1.0) * 100.0

    # Standardized Root Mean Square Error (SRMSE) & Total Absolute Difference (TAD)
    srmse = float(np.sqrt(np.mean((fitted_totals - target_totals) ** 2)) / np.mean(target_totals))
    tad = float(np.sum(abs_errors))

    metrics_df = pd.DataFrame({
        "constraint": constraints,
        "target_control": target_totals,
        "fitted_total": np.round(fitted_totals, 2),
        "absolute_error": np.round(abs_errors, 2),
        "percent_error": np.round(pct_errors, 2),
    })

    summary_meta = {
        "iterations": iteration,
        "converged": converged,
        "SRMSE": round(srmse, 5),
        "TAD": round(tad, 2),
    }

    return weights, metrics_df, summary_meta


def main():
    parser = argparse.ArgumentParser(description="Iterative Proportional Updating (IPU) for population synthesis.")
    parser.add_argument("--seed_file", type=str, default="output/synthetic_population_seed.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--target_pop", type=int, default=10000, help="Target total population for expansion")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.seed_file):
        raise FileNotFoundError(f"Seed file not found: {args.seed_file}. Run train.py first.")

    print(f"Loading synthetic population seeds from {args.seed_file}...")
    seed_df = pd.read_csv(args.seed_file)
    print(f"Loaded {len(seed_df)} agent seeds.")

    print(f"Generating zonal census marginal controls (target N = {args.target_pop})...")
    controls = generate_synthetic_census_controls(target_population_total=args.target_pop)

    print("Running Iterative Proportional Updating (Ye et al., 2009; Sun & Erath, 2015)...")
    weights, metrics_df, meta = iterative_proportional_updating(seed_df, controls)

    print("\n" + "=" * 65)
    print(f"IPU Optimization Summary (Iterations: {meta['iterations']}, Converged: {meta['converged']})")
    print(f"Goodness of Fit: SRMSE = {meta['SRMSE']:.4f}, TAD = {meta['TAD']:.1f}")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print("=" * 65)

    # Save metrics
    metrics_file = os.path.join(args.out_dir, "ipu_validation_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)

    # Save expanded synthetic population with weights
    out_df = seed_df.copy()
    out_df["sample_weight"] = np.round(weights, 4)
    # Integer stochastic replication for direct agent-based simulation
    out_df["integer_weight"] = np.round(weights).astype(int)

    final_file = os.path.join(args.out_dir, "synthetic_population_final.csv")
    out_df.to_csv(final_file, index=False)
    print(f"\nSaved final expanded population: {final_file}")
    print(f"Saved validation metrics: {metrics_file}")


if __name__ == "__main__":
    main()
