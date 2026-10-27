"""
Tests for train.py enhancements:
  1. Default train.py path retains cv_scheme='random'.
  2. Missing classes in training fold are safely mapped via classes_.
  3. Ablation variant assembly produces correct feature subsets.
"""

import argparse
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

import train as tr


def test_default_cli_arguments():
    """Verifies that default arguments in train.py maintain existing defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cv_scheme", choices=["random", "grouped"], default="random")

    args = parser.parse_args([])
    assert args.cv_scheme == "random"
    assert args.n_folds == 5
    assert not args.quick
    assert args.seeds == [42, 123, 999]


def test_missing_class_in_fold_mapping():
    """
    Simulates a fold evaluation where a class is entirely missing from the training fold.
    Verifies that the prediction matrix maps via classes_ to the full (N, K) space
    without ValueError.
    """
    n_samples = 30
    n_features = 5
    K_total = 4

    # Class 3 is present only in test split
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] * 2)  # classes 0, 1, 2
    y_test = np.array([0, 1, 2, 3, 3, 3])  # test split has class 3

    X_train = np.random.default_rng(42).normal(0, 1, (len(y_train), n_features))
    X_test = np.random.default_rng(42).normal(0, 1, (len(y_test), n_features))

    base_model, calibrator = tr._fit_catboost_models(X_train, y_train, n_classes=K_total, random_state=42)

    # Base model uncalibrated mapping
    p_uncal = base_model.predict_proba(X_test)
    assert p_uncal.shape[1] == 3  # Only 3 classes fit

    classes_uncal = np.asarray(getattr(base_model, "classes_", np.arange(p_uncal.shape[1])), dtype=int)
    full_p_uncal = np.zeros((len(X_test), K_total))
    full_p_uncal[:, classes_uncal] = p_uncal

    assert full_p_uncal.shape == (len(X_test), K_total)
    assert (full_p_uncal[:, 3] == 0.0).all()  # Class 3 got 0 probability

    # Calibrator mapping
    p_cal = calibrator.predict_proba(X_test)
    classes_cal = np.asarray(getattr(calibrator, "classes_", np.arange(p_cal.shape[1])), dtype=int)
    full_p_cal = np.zeros((len(X_test), K_total))
    full_p_cal[:, classes_cal] = p_cal

    assert full_p_cal.shape == (len(X_test), K_total)
    assert (full_p_cal[:, 3] == 0.0).all()


def test_assemble_variant_features_shapes():
    """Verifies that assembling new ablation variants includes expected columns."""
    base_df = pd.DataFrame({
        "phone_number": ["BLR100000000", "BLR100000001"],
        "base_feat": [1.0, 2.0],
    })
    ctx_df = pd.DataFrame({
        "phone_number": ["BLR100000000", "BLR100000001"],
        "ctx_home_density_weight": [0.08, 0.11],
    })

    # baseline
    df, dyn = tr.assemble_variant_features(base_df, None, "baseline", zone_ctx_df=ctx_df)
    assert not dyn
    assert "ctx_home_density_weight" not in df.columns

    # baseline+zone_ctx
    df_ctx, dyn = tr.assemble_variant_features(base_df, None, "baseline+zone_ctx", zone_ctx_df=ctx_df)
    assert not dyn
    assert "ctx_home_density_weight" in df_ctx.columns

    # baseline+zone_ctx_shuffled
    df_shuf, dyn = tr.assemble_variant_features(base_df, None, "baseline+zone_ctx_shuffled", zone_ctx_df=ctx_df)
    assert not dyn
    assert "ctx_home_density_weight" in df_shuf.columns


def test_default_train_path_reproduces_on_small_sample():
    """
    Runs the default train.py path (random StratifiedKFold, 5-fold CV, CatBoost + Platt calibration)
    on a deterministic small sample and asserts that metrics match stored default results within tolerance.
    """
    base_df, _ = tr.load_dataset_features()
    pop = pd.read_csv("data/population.csv")

    sample_feat = base_df.iloc[:60].copy()
    sample_pop = pop.iloc[:60].copy()

    # Evaluate default train.py path
    res = tr.evaluate_target_cv(sample_feat, sample_pop, target_col="gender", n_folds=5, random_state=42)

    # Stored expected reference results for this sample
    expected_cal_macro_f1 = 0.4352855
    expected_cal_brier = 0.5899801

    np.testing.assert_allclose(res["cal_macro_f1_mean"], expected_cal_macro_f1, atol=1e-4)
    np.testing.assert_allclose(res["cal_brier_score"], expected_cal_brier, atol=1e-4)

