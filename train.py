"""
train.py

Trains and evaluates CatBoost classifiers for seven socio-demographic
targets from the bandicoot-style feature table (bandicoot_features.csv),
executing 5-fold stratified cross-validation with SMOTE oversampling,
followed by full-dataset model training and generation of synthetic population
seed profiles (predicted categories and class probabilities) for travel demand
modeling (e.g., iterative proportional fitting / IPU).

Targets:
  age_group, gender, education_level, occupation_category, work_status,
  income_bracket, socio_demographic_class

Protocol:
  - 5-fold stratified cross-validation per target for performance estimation.
  - SMOTE oversampling applied per fold, fit exclusively on the training split.
  - CatBoostClassifier with early stopping on an internal validation split.
  - Final model per target fit on all available data to predict the class
    label and probability vector for each subscriber.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

RANDOM_STATE = 42
N_FOLDS = 5

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


from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold

def _min_class_count(y):
    _, counts = np.unique(y, return_counts=True)
    return counts.min()


def _smote_resample(X_train, y_train):
    min_count = _min_class_count(y_train)
    if min_count <= 1:
        return X_train, y_train
    k_neighbors = max(1, min(5, min_count - 1))
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


def _fit_calibrated_catboost(X_train, y_train, n_classes):
    """
    Fits a CatBoostClassifier with SMOTE oversampling and post-hoc Platt scaling
    (sigmoid calibration) on an un-resampled holdout split.

    References:
      - Niculescu-Mizil, A., & Caruana, R. (2005).
        "Predicting good probabilities with supervised learning." ICML '05.
        https://doi.org/10.1145/1102351.1102430
      - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
        "On calibration of modern neural networks." ICML '17.
        https://arxiv.org/abs/1706.04599
      - He, H., & Garcia, E. A. (2009).
        "Learning from imbalanced data." IEEE Trans. Knowl. Data Eng., 21(9), 1263-1284.
        https://doi.org/10.1109/TKDE.2008.239
        (Addresses empirical prior distortion caused by synthetic oversampling).
    """
    n_total = len(X_train)
    n_cal = max(10, int(0.15 * n_total))

    # Keep an un-resampled calibration validation set to restore natural empirical priors
    if n_total - n_cal >= 10:
        X_tr_raw, y_tr_raw = X_train[:-n_cal], y_train[:-n_cal]
        X_cal, y_cal = X_train[-n_cal:], y_train[-n_cal:]
    else:
        X_tr_raw, y_tr_raw = X_train, y_train
        X_cal, y_cal = X_train, y_train

    X_tr_res, y_tr_res = _smote_resample(X_tr_raw, y_tr_raw)

    base_model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.08,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=20,
        thread_count=-1,
    )
    base_model.fit(X_tr_res, y_tr_res)

    # Calibrate probabilities against the un-resampled calibration set
    try:
        calibrator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
        calibrator.fit(X_cal, y_cal)
        return calibrator
    except Exception:
        return base_model


def train_and_evaluate(features_df, population_df, target_col, n_folds=N_FOLDS):
    df = features_df.merge(population_df[["phone_number", target_col]], on="phone_number")
    feature_cols = [c for c in features_df.columns if c != "phone_number"]

    X_raw = df[feature_cols].fillna(0.0).to_numpy()
    y_raw = df[target_col].to_numpy()

    # Prune near-zero-variance redundant columns
    selector = VarianceThreshold(threshold=1e-5)
    X = selector.fit_transform(X_raw)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_macro_f1, fold_weighted_f1 = [], []
    all_true, all_pred = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = _fit_calibrated_catboost(X_train, y_train, n_classes)

        proba = model.predict_proba(X_test)
        y_pred = np.argmax(proba, axis=1)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_macro_f1.append(macro_f1)
        fold_weighted_f1.append(weighted_f1)
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"  [{target_col}] fold {fold_idx + 1}/{n_folds}: "
              f"macro-F1={macro_f1:.3f}, weighted-F1={weighted_f1:.3f}")

    report = classification_report(
        all_true, all_pred, target_names=le.classes_.astype(str), zero_division=0
    )

    return {
        "target": target_col,
        "macro_f1_mean": float(np.mean(fold_macro_f1)),
        "macro_f1_std": float(np.std(fold_macro_f1)),
        "weighted_f1_mean": float(np.mean(fold_weighted_f1)),
        "weighted_f1_std": float(np.std(fold_weighted_f1)),
        "is_known_artifact": target_col in ARTIFACT_TARGETS,
        "classification_report": report,
    }


def train_final_model_and_predict(features_df, population_df, target_col):
    """
    Fits one final calibrated model per target on all agents and returns
    predicted classes and class probabilities formatted for synthetic
    population generation.
    """
    df = features_df.merge(population_df[["phone_number", target_col]], on="phone_number")
    feature_cols = [c for c in features_df.columns if c != "phone_number"]

    X_all_raw = features_df[feature_cols].fillna(0.0).to_numpy()
    X_train_raw = df[feature_cols].fillna(0.0).to_numpy()
    y_raw = df[target_col].to_numpy()

    selector = VarianceThreshold(threshold=1e-5)
    X_train = selector.fit_transform(X_train_raw)
    X_all = selector.transform(X_all_raw)

    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    model = _fit_calibrated_catboost(X_train, y_train, n_classes)

    proba = model.predict_proba(X_all)
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"phone_number": features_df["phone_number"].to_numpy()})
    out[f"predicted_{target_col}"] = pred_labels
    for i, cls in enumerate(le.classes_):
        out[f"prob_{target_col}__{cls}"] = proba[:, i]
    return out


def export_for_population_synthesis(features_df, population_df, targets=TARGETS):
    """
    Fits final models for all targets and combines predictions and class
    probabilities into a unified synthetic population seed matrix.
    """
    merged = pd.DataFrame({"phone_number": features_df["phone_number"].to_numpy()})
    for target in targets:
        print(f"Fitting final model for '{target}' (full-data, for population synthesis export)...")
        target_preds = train_final_model_and_predict(features_df, population_df, target)
        merged = merged.merge(target_preds, on="phone_number")
    return merged


def main():
    features = pd.read_csv("output/bandicoot_features.csv")
    population = pd.read_csv("data/population.csv")

    results = []
    for target in TARGETS:
        print(f"\n=== Cross-validating target: {target} ===")
        result = train_and_evaluate(features, population, target)
        results.append(result)

    summary = pd.DataFrame([
        {
            "target": r["target"],
            "macro_f1_mean": round(r["macro_f1_mean"], 4),
            "macro_f1_std": round(r["macro_f1_std"], 4),
            "weighted_f1_mean": round(r["weighted_f1_mean"], 4),
            "known_artifact": r["is_known_artifact"],
        }
        for r in results
    ])
    summary.to_csv("output/results_summary.csv", index=False)

    print("\n" + "=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)
    print("\nNOTE: work_status is conditionally coupled with occupation in the "
          "synthetic generator. See docs/DATA_LINEAGE.md for details.")

    with open("output/classification_reports.txt", "w") as f:
        for r in results:
            f.write(f"\n=== {r['target']} ===\n")
            f.write(r["classification_report"])
            f.write("\n")

    print("\n=== Fitting final models and exporting synthetic population seed data ===")
    synth_seed = export_for_population_synthesis(features, population)
    synth_seed.to_csv("output/synthetic_population_seed.csv", index=False)
    print(f"Exported output/synthetic_population_seed.csv "
          f"({synth_seed.shape[0]} agents x {synth_seed.shape[1] - 1} prediction/probability columns).")


if __name__ == "__main__":
    main()
