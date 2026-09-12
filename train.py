"""
train.py

Trains and evaluates CatBoost classifiers for seven socio-demographic
targets from the bandicoot-style feature table (bandicoot_features.csv),
then fits a final model per target on all available data and exports a
prediction file formatted for use as SEED DATA in synthetic population
generation (e.g. iterative proportional fitting / IPU-style population
synthesis, as used downstream of the trip-generation stage in the
classical four-step model — see docs/REFERENCES.md, Mathew's Travel
Demand Modeling notes, Ch. 2-4).

Scope note: this module is the actual project deliverable — a model that
predicts socio-demographics per phone_number from CDR-derived behavioral
features. Everything upstream (data generation, noise reduction,
stay-point extraction, place labeling, bandicoot feature extraction) is
scaffolding that produces this module's input; pipeline.py's three-graph
features are an optional additional feature set, not required to run this.

Targets:
  age_group, gender, education_level, occupation_category, work_status,
  income_bracket, socio_demographic_class

Protocol:
  - 5-fold STRATIFIED cross-validation per target, for reporting F1.
  - SMOTE oversampling applied PER FOLD, fit only on the training split
    of that fold (never on the held-out fold).
  - CatBoostClassifier with early stopping on a validation slice carved
    out of the training fold.
  - After CV reporting, one FINAL model per target is fit on ALL agents
    (with SMOTE applied to the full training set) and used to predict a
    class + full class-probability vector for every agent — this is the
    file that should be fed to a population synthesis tool.

KNOWN CAVEAT: `work_status` is flagged as a generator artifact — see
`# ARTIFACT` in generate_bengaluru_data.py. Its near-perfect F1 reflects
how cleanly the synthetic generator's own rules determine it, not a
genuine pipeline capability on real CDR data.
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

# Targets known to be generator artifacts — reported separately with a caveat.
ARTIFACT_TARGETS = {"work_status"}


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


def _fit_catboost(X_train, y_train, n_classes):
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.08,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=20,
        thread_count=-1,
    )
    n_val = max(1, int(0.1 * len(X_train)))
    if len(X_train) - n_val < 2:
        model.fit(X_train, y_train)
    else:
        model.fit(
            X_train[:-n_val], y_train[:-n_val],
            eval_set=(X_train[-n_val:], y_train[-n_val:]),
            use_best_model=True,
        )
    return model


def train_and_evaluate(features_df, population_df, target_col, n_folds=N_FOLDS):
    df = features_df.merge(population_df[["phone_number", target_col]], on="phone_number")
    feature_cols = [c for c in features_df.columns if c != "phone_number"]

    X = df[feature_cols].fillna(0.0).to_numpy()
    y_raw = df[target_col].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_macro_f1, fold_weighted_f1 = [], []
    all_true, all_pred = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_res, y_train_res = _smote_resample(X_train, y_train)
        model = _fit_catboost(X_train_res, y_train_res, n_classes)

        y_pred = model.predict(X_test).flatten().astype(int)
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
    Fits one final model per target on ALL agents (post-SMOTE) and
    returns a DataFrame [phone_number, predicted_<target>,
    prob_<target>__<class>, ...] for every agent in features_df — this
    is the population-synthesis-ready output for this target.
    """
    df = features_df.merge(population_df[["phone_number", target_col]], on="phone_number")
    feature_cols = [c for c in features_df.columns if c != "phone_number"]

    X_all = features_df[feature_cols].fillna(0.0).to_numpy()  # predict for every agent, not just labeled ones
    X_train = df[feature_cols].fillna(0.0).to_numpy()
    y_raw = df[target_col].to_numpy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    X_train_res, y_train_res = _smote_resample(X_train, y_train)
    model = _fit_catboost(X_train_res, y_train_res, n_classes)

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
    Runs train_final_model_and_predict for every target and assembles one
    wide table: phone_number + predicted class & class-probability columns
    for all seven targets. This is the file to hand to a population
    synthesis tool (e.g. as IPU/IPF seed data, matching predicted category
    probabilities against zonal marginal control totals).
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
    print("\nNOTE: rows with known_artifact=True (work_status) reflect a "
          "generator-side shortcut, not genuine predictive signal on real "
          "CDR data. See docs/DATA_LINEAGE.md.")

    with open("output/classification_reports.txt", "w") as f:
        for r in results:
            f.write(f"\n=== {r['target']} ===\n")
            f.write(r["classification_report"])
            f.write("\n")

    print("\n=== Fitting final models and exporting synthetic population seed data ===")
    synth_seed = export_for_population_synthesis(features, population)
    synth_seed.to_csv("output/synthetic_population_seed.csv", index=False)
    print(f"Wrote output/synthetic_population_seed.csv "
          f"({synth_seed.shape[0]} agents x {synth_seed.shape[1] - 1} predicted/probability columns) "
          f"— this is the file to feed into a synthetic population generation tool.")


if __name__ == "__main__":
    main()
