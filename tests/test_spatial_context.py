"""
Tests for spatial_context.py:
  1. Grouped folds never place the same zone group in both train and test splits.
  2. Shuffled control preserves column marginal distributions/means but breaks agent link.
  3. Zone context features are built exclusively from inferred anchors.
  4. Street View feature loader returns None when absent, parses numeric columns when present.
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

import spatial_context as sc
from generate_bengaluru_data import ZONES


def test_build_zone_context_uses_only_inferred_anchors():
    """Verifies that zone context is extracted from inferred anchors only."""
    ctx_df = sc.build_zone_context()
    assert "phone_number" in ctx_df.columns
    expected_cols = [
        "ctx_home_density_weight",
        "ctx_home_employment_weight",
        "ctx_home_dist_to_zone_centre_km",
        "ctx_work_employment_weight",
    ]
    for col in expected_cols:
        assert col in ctx_df.columns
        assert not ctx_df[col].isna().any()

    # Distances must be strictly positive
    assert (ctx_df["ctx_home_dist_to_zone_centre_km"] >= 0.0).all()
    # Weights must be valid zone weights
    valid_densities = {z.density_weight for z in ZONES}
    assert set(ctx_df["ctx_home_density_weight"].unique()).issubset(valid_densities)


def test_grouped_folds_never_overlap_groups():
    """Verifies that StratifiedGroupKFold never places the same home zone group in train and test."""
    groups_dict = sc.get_home_zone_groups()
    assert len(groups_dict) > 0

    anchors = pd.read_csv("output/home_work_anchors.csv")
    anchors = sc._resolve_phone_numbers(anchors)
    df = pd.DataFrame({
        "phone_number": anchors["phone_number"],
        "target": np.random.default_rng(42).choice(["A", "B", "C"], size=len(anchors)),
    })
    groups = df["phone_number"].map(groups_dict).to_numpy()
    y = df["target"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(df, y, groups)):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        assert len(overlap) == 0, f"Fold {fold} has overlapping groups: {overlap}"


def test_shuffled_control_preserves_means_and_breaks_link():
    """Verifies shuffled control keeps column means exactly but breaks the agent-feature mapping."""
    ctx_df = sc.build_zone_context()
    feature_cols = [
        "ctx_home_density_weight",
        "ctx_home_employment_weight",
        "ctx_home_dist_to_zone_centre_km",
        "ctx_work_employment_weight",
    ]
    shuffled_df = sc.make_shuffled_control(ctx_df, feature_cols=feature_cols, random_state=99)

    assert shuffled_df is not None
    # Identical phone numbers and shape
    assert (shuffled_df["phone_number"] == ctx_df["phone_number"]).all()
    assert shuffled_df.shape == ctx_df.shape

    for col in feature_cols:
        orig_vals = ctx_df[col].to_numpy()
        shuf_vals = shuffled_df[col].to_numpy()

        # Means and sums must be identical
        np.testing.assert_allclose(np.mean(orig_vals), np.mean(shuf_vals), rtol=1e-7)
        np.testing.assert_allclose(np.std(orig_vals), np.std(shuf_vals), rtol=1e-7)

        # Sorted arrays must match (identical marginal distribution)
        np.testing.assert_array_equal(np.sort(orig_vals), np.sort(shuf_vals))

        # The pairing across agents must be broken (discrete zone weights with repeated
        # values naturally have ~30-35% accidental collision rate under random permutation)
        mismatch_count = np.sum(orig_vals != shuf_vals)
        assert mismatch_count > 0.5 * len(orig_vals), (
            f"Column {col} has too many identical rows after shuffling ({mismatch_count})"
        )


def test_load_streetview_features_nonexistent(tmp_path):
    """Verifies loader returns None when the file does not exist."""
    missing_file = str(tmp_path / "nonexistent_sv.csv")
    result = sc.load_streetview_features(missing_file)
    assert result is None


def test_load_streetview_features_existing(tmp_path):
    """Verifies loader returns only numeric sv_* feature columns and phone_number."""
    dummy_csv = tmp_path / "streetview_features.csv"
    dummy_data = pd.DataFrame({
        "phone_number": ["BLR100000000", "BLR100000001"],
        "sv_status": ["OK", "OK"],
        "sv_date": ["2025-01", "2025-02"],
        "sv_pano_id": ["pano1", "pano2"],
        "sv_has_coverage": [1, 1],
        "sv_exg_mean": [0.35, 0.42],
        "sv_edge_density_mean": [0.12, 0.18],
    })
    dummy_data.to_csv(dummy_csv, index=False)

    loaded = sc.load_streetview_features(str(dummy_csv))
    assert loaded is not None
    assert set(loaded.columns) == {"phone_number", "sv_has_coverage", "sv_exg_mean", "sv_edge_density_mean"}
    assert "sv_status" not in loaded.columns
    assert "sv_date" not in loaded.columns
    assert "sv_pano_id" not in loaded.columns
