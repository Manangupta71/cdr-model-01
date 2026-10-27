"""
train.py

Trains and evaluates CatBoost classifiers for seven socio-demographic targets
using behavioral indicators, graph-derived relational features, and spatial context.
Evaluates systematic feature ablations across multiple seeds to test whether
graph features or spatial context features improve prediction over baseline
tabular bandicoot indicators.

Targets:
  age_group, gender, education_level, occupation_category, work_status,
  income_bracket, socio_demographic_class

Ablation Variants:
  1. baseline: Bandicoot indicators + commute features + mobility motifs
  2. baseline+mobility: Baseline + Mobility Similarity Graph (k-NN)
  3. baseline+colocation: Baseline + Co-location Graph
  4. baseline+social: Baseline + Social Interaction Graph
  5. baseline+all_graphs: Baseline + All Three User Graphs
  6. baseline+zone_ctx: Baseline + Zone Spatial Context
  7. baseline+zone_ctx_shuffled: Baseline + Shuffled Zone Context (Negative Control)
  [Optional: baseline+streetview, baseline+streetview_shuffled if streetview_features.csv exists]

Leakage Prevention Protocol:
  - 5-fold stratified cross-validation per target (random or grouped by inferred home zone).
  - SMOTE oversampling applied per fold, fit exclusively on the training split.
  - Mobility k-NN graph is fitted dynamically within each CV fold on the training split.
  - Platt probability calibration fit on un-resampled training holdouts.
  - Conditional classifier chains along demographic DAG pass out-of-fold probabilities.
  - Zero neighbor target labels are used as features; only behavioral feature aggregates.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

from graph_features import MobilityKNNGraphBuilder
from spatial_context import (
    build_zone_context,
    get_home_zone_groups,
    load_streetview_features,
    make_shuffled_control,
)

TARGETS = [
    "age_group",
    "gender",
    "education_level",
    "occupation_category",
    "work_status",
    "income_bracket",
    "socio_demographic_class",
]

# Simulation-coupled targets evaluated with explicit data lineage context.
ARTIFACT_TARGETS = {"work_status"}

DEPENDENCY_GRAPH = {
    "age_group": [],
    "gender": [],
    "education_level": ["age_group"],
    "occupation_category": ["age_group", "education_level"],
    "work_status": ["age_group", "occupation_category"],
    "income_bracket": ["education_level", "occupation_category"],
    "socio_demographic_class": ["education_level", "income_bracket"],
}


def _min_class_count(y):
    _, counts = np.unique(y, return_counts=True)
    return counts.min()


def _smote_resample(X_train, y_train, random_state=42):
    min_count = _min_class_count(y_train)
    if min_count <= 1:
        return X_train, y_train
    k_neighbors = max(1, min(5, min_count - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


def _compute_brier_score(y_true, proba):
    """Computes multi-class Brier score: mean squared difference from one-hot target."""
    n_classes = proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def _fit_catboost_models(X_train, y_train, n_classes, random_state=42):
    """
    Fits CatBoostClassifier with SMOTE oversampling on training split.
    Returns both uncalibrated model and Platt-calibrated model fit on
    an un-resampled training holdout split.
    """
    n_total = len(X_train)
    n_cal = max(10, int(0.15 * n_total))

    if n_total - n_cal >= 10:
        X_tr_raw, y_tr_raw = X_train[:-n_cal], y_train[:-n_cal]
        X_cal, y_cal = X_train[-n_cal:], y_train[-n_cal:]
    else:
        X_tr_raw, y_tr_raw = X_train, y_train
        X_cal, y_cal = X_train, y_train

    X_tr_res, y_tr_res = _smote_resample(X_tr_raw, y_tr_raw, random_state=random_state)

    base_model = CatBoostClassifier(
        iterations=80,
        depth=6,
        learning_rate=0.08,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=random_state,
        verbose=False,
        early_stopping_rounds=15,
        thread_count=-1,
    )
    base_model.fit(X_tr_res, y_tr_res)

    # Platt calibration (sigmoid scaling on un-resampled validation holdout)
    try:
        calibrator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
        calibrator.fit(X_cal, y_cal)
    except Exception:
        calibrator = base_model

    return base_model, calibrator


def _get_chain_features(features_df, prior_preds, target_col):
    """
    Appends out-of-fold probability vectors from parent demographic variables
    along the demographic dependency DAG.
    """
    parents = DEPENDENCY_GRAPH.get(target_col, [])
    if not parents:
        return features_df

    extra_dfs = []
    for p in parents:
        if p in prior_preds:
            p_df = prior_preds[p]
            prob_cols = [c for c in p_df.columns if c.startswith(f"prob_{p}__")]
            extra_dfs.append(p_df[["phone_number"] + prob_cols])

    if not extra_dfs:
        return features_df

    chained = features_df.copy()
    for extra in extra_dfs:
        chained = chained.merge(extra, on="phone_number")
    return chained


def evaluate_target_cv(features_df, population_df, target_col, prior_preds=None,
                       n_folds=5, random_state=42, use_dynamic_knn=False,
                       cv_scheme="random", groups=None):
    """
    Evaluates 5-fold cross-validation for a given target.
    Supports random StratifiedKFold or grouped StratifiedGroupKFold on inferred home zone.
    Returns metrics both with and without Platt calibration, plus OOF predictions.
    """
    prior_preds = prior_preds or {}
    chained = _get_chain_features(features_df, prior_preds, target_col)

    df = chained.merge(population_df[["phone_number", target_col]], on="phone_number")
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].to_numpy())
    n_classes = len(le.classes_)

    if cv_scheme == "grouped":
        if groups is None:
            raise ValueError("groups mapping must be provided when cv_scheme='grouped'")
        grp_labels = df["phone_number"].map(groups).to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(sgkf.split(df, y, grp_labels))
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(skf.split(df, y))

    oof_proba_uncal = np.zeros((len(df), n_classes))
    oof_proba_cal = np.zeros((len(df), n_classes))

    macro_f1_uncal, macro_f1_cal = [], []
    weighted_f1_uncal, weighted_f1_cal = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # If dynamic mobility k-NN is enabled, fit k-NN strictly on training fold
        if use_dynamic_knn:
            knn_builder = MobilityKNNGraphBuilder(k=5)
            knn_builder.fit(train_df)
            mob_tr = knn_builder.transform(train_df, is_train=True)
            mob_te = knn_builder.transform(test_df, is_train=False)

            train_df = train_df.merge(mob_tr, on="phone_number")
            test_df = test_df.merge(mob_te, on="phone_number")

        feature_cols = [c for c in train_df.columns if c not in ("phone_number", target_col)]
        X_train_raw = train_df[feature_cols].fillna(0.0).to_numpy()
        X_test_raw = test_df[feature_cols].fillna(0.0).to_numpy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        selector = VarianceThreshold(threshold=1e-5)
        X_train = selector.fit_transform(X_train_raw)
        X_test = selector.transform(X_test_raw)

        base_model, calibrator = _fit_catboost_models(X_train, y_train, n_classes, random_state=random_state)

        # Uncalibrated evaluation (map via classes_ if classes are missing in training fold)
        p_uncal = base_model.predict_proba(X_test)
        classes_uncal = np.asarray(getattr(base_model, "classes_", np.arange(p_uncal.shape[1])), dtype=int)
        full_p_uncal = np.zeros((len(test_idx), n_classes))
        full_p_uncal[:, classes_uncal] = p_uncal
        oof_proba_uncal[test_idx] = full_p_uncal
        pred_uncal = np.argmax(full_p_uncal, axis=1)
        macro_f1_uncal.append(f1_score(y_test, pred_uncal, average="macro", zero_division=0))
        weighted_f1_uncal.append(f1_score(y_test, pred_uncal, average="weighted", zero_division=0))

        # Calibrated evaluation (map via classes_ if classes are missing in training fold)
        p_cal = calibrator.predict_proba(X_test)
        classes_cal = np.asarray(getattr(calibrator, "classes_", np.arange(p_cal.shape[1])), dtype=int)
        full_p_cal = np.zeros((len(test_idx), n_classes))
        full_p_cal[:, classes_cal] = p_cal
        oof_proba_cal[test_idx] = full_p_cal
        pred_cal = np.argmax(full_p_cal, axis=1)
        macro_f1_cal.append(f1_score(y_test, pred_cal, average="macro", zero_division=0))
        weighted_f1_cal.append(f1_score(y_test, pred_cal, average="weighted", zero_division=0))

    # Compute Brier scores across OOF predictions
    uncal_brier = _compute_brier_score(y, oof_proba_uncal)
    cal_brier = _compute_brier_score(y, oof_proba_cal)

    # Build OOF predictions dataframe using calibrated probabilities for chaining
    oof_df = pd.DataFrame({"phone_number": df["phone_number"].to_numpy()})
    for i, cls in enumerate(le.classes_):
        oof_df[f"prob_{target_col}__{cls}"] = oof_proba_cal[:, i]

    # Classification report on calibrated OOF predictions
    y_pred_cal_all = np.argmax(oof_proba_cal, axis=1)
    report = classification_report(y, y_pred_cal_all, target_names=le.classes_.astype(str), zero_division=0)

    return {
        "target": target_col,
        "uncal_macro_f1_mean": float(np.mean(macro_f1_uncal)),
        "uncal_macro_f1_std": float(np.std(macro_f1_uncal)),
        "cal_macro_f1_mean": float(np.mean(macro_f1_cal)),
        "cal_macro_f1_std": float(np.std(macro_f1_cal)),
        "uncal_weighted_f1_mean": float(np.mean(weighted_f1_uncal)),
        "cal_weighted_f1_mean": float(np.mean(weighted_f1_cal)),
        "uncal_brier_score": uncal_brier,
        "cal_brier_score": cal_brier,
        "is_known_artifact": target_col in ARTIFACT_TARGETS,
        "classification_report": report,
        "oof_df": oof_df,
    }


def train_final_full_model_and_predict(features_df, population_df, target_col,
                                       prior_final_preds=None, random_state=42,
                                       use_dynamic_knn=False):
    """
    Fits one final model per target on full dataset for final production predictions.
    """
    prior_final_preds = prior_final_preds or {}
    chained = _get_chain_features(features_df, prior_final_preds, target_col)

    df = chained.merge(population_df[["phone_number", target_col]], on="phone_number")

    if use_dynamic_knn:
        knn_builder = MobilityKNNGraphBuilder(k=5)
        knn_builder.fit(df)
        mob_full = knn_builder.transform(df, is_train=True)
        df = df.merge(mob_full, on="phone_number")
        chained = chained.merge(mob_full, on="phone_number")

    feature_cols = [c for c in df.columns if c not in ("phone_number", target_col)]
    X_train_raw = df[feature_cols].fillna(0.0).to_numpy()
    X_all_raw = chained[feature_cols].fillna(0.0).to_numpy()

    selector = VarianceThreshold(threshold=1e-5)
    X_train = selector.fit_transform(X_train_raw)
    X_all = selector.transform(X_all_raw)

    le = LabelEncoder()
    y_train = le.fit_transform(df[target_col].to_numpy())
    n_classes = len(le.classes_)

    _, calibrator = _fit_catboost_models(X_train, y_train, n_classes, random_state=random_state)

    proba = calibrator.predict_proba(X_all)
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"phone_number": chained["phone_number"].to_numpy()})
    out[f"predicted_{target_col}"] = pred_labels
    for i, cls in enumerate(le.classes_):
        out[f"prob_{target_col}__{cls}"] = proba[:, i]
    return out


def load_dataset_features(bandicoot_path="output/bandicoot_features.csv",
                          graph_path="output/graph_features.csv",
                          motifs_path="output/mobility_motifs.csv"):
    """
    Loads baseline bandicoot table and auxiliary feature sets.
    """
    base_df = pd.read_csv(bandicoot_path)
    if os.path.exists(motifs_path):
        motifs_df = pd.read_csv(motifs_path)
        base_df = base_df.merge(motifs_df, on="phone_number", how="left")

    graph_df = pd.read_csv(graph_path) if os.path.exists(graph_path) else None

    return base_df, graph_df


def assemble_variant_features(base_df, graph_df, variant_name,
                              zone_ctx_df=None, streetview_df=None,
                              seed=42):
    """
    Assembles feature dataframe corresponding to the ablation variant.
    """
    if variant_name == "baseline":
        return base_df.copy(), False

    df = base_df.copy()
    use_dynamic_knn = False

    if variant_name == "baseline+social":
        if graph_df is not None:
            social_cols = ["phone_number"] + [c for c in graph_df.columns if c.startswith("graph_social__")]
            df = df.merge(graph_df[social_cols], on="phone_number", how="left")
    elif variant_name == "baseline+colocation":
        if graph_df is not None:
            colo_cols = ["phone_number"] + [c for c in graph_df.columns if c.startswith("graph_colocation__")]
            df = df.merge(graph_df[colo_cols], on="phone_number", how="left")
    elif variant_name == "baseline+mobility":
        # Use dynamic fold-isolated k-NN
        use_dynamic_knn = True
    elif variant_name == "baseline+all_graphs":
        if graph_df is not None:
            social_and_colo = ["phone_number"] + [c for c in graph_df.columns if c.startswith("graph_social__") or c.startswith("graph_colocation__")]
            df = df.merge(graph_df[social_and_colo], on="phone_number", how="left")
        use_dynamic_knn = True
    elif variant_name == "baseline+zone_ctx":
        if zone_ctx_df is not None:
            df = df.merge(zone_ctx_df, on="phone_number", how="left")
    elif variant_name == "baseline+zone_ctx_shuffled":
        if zone_ctx_df is not None:
            shuf = make_shuffled_control(zone_ctx_df, random_state=seed)
            df = df.merge(shuf, on="phone_number", how="left")
    elif variant_name == "baseline+streetview":
        if streetview_df is not None:
            df = df.merge(streetview_df, on="phone_number", how="left")
    elif variant_name == "baseline+streetview_shuffled":
        if streetview_df is not None:
            shuf = make_shuffled_control(streetview_df, random_state=seed)
            df = df.merge(shuf, on="phone_number", how="left")

    return df, use_dynamic_knn


def run_experiments(seeds=(42, 123, 999), n_folds=5, run_ablation=True, cv_scheme="random"):
    """
    Executes cross-validation over the ablation suite across multiple seeds,
    records fold-level std and seed spread, and exports predictions.
    """
    base_df, graph_df = load_dataset_features()
    population = pd.read_csv("data/population.csv")

    try:
        zone_ctx_df = build_zone_context()
        home_zone_groups = get_home_zone_groups()
    except Exception:
        zone_ctx_df = None
        home_zone_groups = None

    streetview_df = load_streetview_features()

    if run_ablation:
        variants = [
            "baseline",
            "baseline+mobility",
            "baseline+colocation",
            "baseline+social",
            "baseline+all_graphs",
            "baseline+zone_ctx",
            "baseline+zone_ctx_shuffled",
        ]
        if streetview_df is not None:
            variants.extend(["baseline+streetview", "baseline+streetview_shuffled"])
    else:
        variants = ["baseline", "baseline+all_graphs"]

    print("=" * 75)
    print(f"Running Socio-Demographic Prediction Pipeline Across {len(seeds)} Random Seeds: {seeds}")
    print(f"CV Scheme: {cv_scheme.upper()} (Splits: {n_folds})")
    print(f"Ablation Variants ({len(variants)}): {variants}")
    print("=" * 75)

    experiment_records = []
    final_best_reports = {}

    for var in variants:
        print(f"\n>>> Evaluating Variant: {var} <<<")
        for seed in seeds:
            print(f"--- [Seed {seed}] ---")
            feat_df, use_dyn_knn = assemble_variant_features(
                base_df, graph_df, var,
                zone_ctx_df=zone_ctx_df,
                streetview_df=streetview_df,
                seed=seed,
            )
            prior_oof = {}
            for target in TARGETS:
                res = evaluate_target_cv(
                    feat_df, population, target,
                    prior_preds=prior_oof,
                    n_folds=n_folds,
                    random_state=seed,
                    use_dynamic_knn=use_dyn_knn,
                    cv_scheme=cv_scheme,
                    groups=home_zone_groups,
                )
                prior_oof[target] = res["oof_df"]

                experiment_records.append({
                    "variant": var,
                    "seed": seed,
                    "target": target,
                    "uncal_macro_f1": res["uncal_macro_f1_mean"],
                    "cal_macro_f1": res["cal_macro_f1_mean"],
                    "uncal_weighted_f1": res["uncal_weighted_f1_mean"],
                    "cal_weighted_f1": res["cal_weighted_f1_mean"],
                    "uncal_brier": res["uncal_brier_score"],
                    "cal_brier": res["cal_brier_score"],
                    "is_known_artifact": res["is_known_artifact"],
                })

                if var == "baseline+all_graphs" and seed == seeds[0]:
                    final_best_reports[target] = res["classification_report"]

    exp_df = pd.DataFrame(experiment_records)

    # 1. Ablation Summary Aggregated Across Seeds
    ablation_summary = (
        exp_df.groupby(["variant", "target", "is_known_artifact"])
        .agg(
            cal_macro_f1_mean=("cal_macro_f1", "mean"),
            cal_macro_f1_std=("cal_macro_f1", "std"),
            cal_macro_f1_spread=("cal_macro_f1", lambda s: float(s.max() - s.min())),
            uncal_macro_f1_mean=("uncal_macro_f1", "mean"),
            uncal_macro_f1_std=("uncal_macro_f1", "std"),
            cal_brier_mean=("cal_brier", "mean"),
            uncal_brier_mean=("uncal_brier", "mean"),
            cal_weighted_f1_mean=("cal_weighted_f1", "mean"),
        )
        .reset_index()
    )
    for c in ["cal_macro_f1_mean", "cal_macro_f1_std", "cal_macro_f1_spread",
              "uncal_macro_f1_mean", "uncal_macro_f1_std", "cal_brier_mean",
              "uncal_brier_mean", "cal_weighted_f1_mean"]:
        ablation_summary[c] = ablation_summary[c].fillna(0.0).round(4)

    ablation_summary.to_csv("output/ablation_summary.csv", index=False)
    print("\nSaved output/ablation_summary.csv")

    # 2. Results Summary comparing Baseline vs Best Graph Variant
    base_sub = exp_df[exp_df["variant"] == "baseline"].groupby("target").agg(
        baseline_cal_macro_f1=("cal_macro_f1", "mean"),
        baseline_cal_macro_f1_std=("cal_macro_f1", "std"),
        baseline_uncal_brier=("uncal_brier", "mean"),
        baseline_cal_brier=("cal_brier", "mean"),
    )
    all_graph_sub = exp_df[exp_df["variant"] == "baseline+all_graphs"].groupby("target").agg(
        all_graphs_cal_macro_f1=("cal_macro_f1", "mean"),
        all_graphs_cal_macro_f1_std=("cal_macro_f1", "std"),
        all_graphs_uncal_brier=("uncal_brier", "mean"),
        all_graphs_cal_brier=("cal_brier", "mean"),
    )
    results_summary = base_sub.join(all_graph_sub).reset_index()
    results_summary["delta_f1_from_graphs"] = (
        results_summary["all_graphs_cal_macro_f1"] - results_summary["baseline_cal_macro_f1"]
    ).round(4)
    results_summary["brier_reduction_from_cal"] = (
        results_summary["all_graphs_uncal_brier"] - results_summary["all_graphs_cal_brier"]
    ).round(4)
    results_summary["is_known_artifact"] = results_summary["target"].isin(ARTIFACT_TARGETS)

    for col in results_summary.columns:
        if any(k in col for k in ["f1", "brier", "delta"]):
            results_summary[col] = results_summary[col].fillna(0.0).round(4)

    results_summary.to_csv("output/results_summary.csv", index=False)
    print("Saved output/results_summary.csv")

    # 3. Classification Reports
    with open("output/classification_reports.txt", "w") as f:
        f.write("======================================================================\n")
        f.write("CLASSIFICATION REPORTS PER TARGET (baseline+all_graphs, calibrated)\n")
        f.write("======================================================================\n\n")
        for target, rep in final_best_reports.items():
            f.write(f"\n=== Target: {target} ===\n")
            f.write(rep)
            f.write("\n")
    print("Saved output/classification_reports.txt")

    # 4. Generate Final Production Predictions (predictions.csv)
    print("\nFitting final production models on full unified dataset...")
    full_feat_df, use_dyn_knn = assemble_variant_features(
        base_df, graph_df, "baseline+all_graphs",
        zone_ctx_df=zone_ctx_df,
        streetview_df=streetview_df,
        seed=seeds[0],
    )
    prior_final_preds = {}
    merged_preds = pd.DataFrame({"phone_number": full_feat_df["phone_number"].to_numpy()})

    for target in TARGETS:
        target_pred_df = train_final_full_model_and_predict(
            full_feat_df, population, target,
            prior_final_preds=prior_final_preds,
            random_state=seeds[0],
            use_dynamic_knn=use_dyn_knn
        )
        prior_final_preds[target] = target_pred_df
        merged_preds = merged_preds.merge(target_pred_df, on="phone_number")

    pred_path = "output/predictions.csv"
    merged_preds.to_csv(pred_path, index=False)
    print(f"Exported {pred_path} ({len(merged_preds)} subscribers x {merged_preds.shape[1] - 1} prediction/probability columns).")

    print("\n" + "=" * 75)
    print("FINAL RESULTS SUMMARY (Baseline vs. Baseline + All Graphs):")
    print("=" * 75)
    print(results_summary.to_string(index=False))
    print("\nNOTE: work_status is conditionally coupled with occupation category in the synthetic")
    print("generator (see docs/DATA_LINEAGE.md); its high performance reflects this coupling.")


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost models and evaluate graph feature ablations.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999],
                        help="Random seeds for CV stability verification.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--quick", action="store_true", help="Run quick single-seed check.")
    parser.add_argument("--cv_scheme", choices=["random", "grouped"], default="random",
                        help="Cross-validation scheme: 'random' (StratifiedKFold) or "
                             "'grouped' (StratifiedGroupKFold on nearest inferred home zone).")
    args = parser.parse_args()

    seeds = [42] if args.quick else args.seeds
    run_experiments(seeds=seeds, n_folds=args.n_folds, run_ablation=True, cv_scheme=args.cv_scheme)


if __name__ == "__main__":
    main()
