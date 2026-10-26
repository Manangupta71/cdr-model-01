"""
Tests for streetview_probe.py:
  1. Dry-run mode executes end-to-end without making any network calls.
  2. In-memory Excess Green and edge density metrics compute valid float indices.
"""

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import streetview_probe as sv


def test_dry_run_makes_no_network_calls(tmp_path):
    """Verifies that --dry_run makes zero HTTP network calls and requires no API key."""
    out_file = tmp_path / "sv_dry_run.csv"

    # Mock requests.get to fail immediately if called
    with patch("requests.get", side_effect=RuntimeError("Network request made during dry run!")):
        res_df = sv.run_probe(
            anchors_path="output/home_work_anchors.csv",
            output_path=str(out_file),
            fetch_images=True,
            max_requests=10,
            dry_run=True,
            resume=False,
        )

    assert out_file.exists()
    assert len(res_df) > 0
    assert "sv_has_coverage" in res_df.columns
    assert "sv_exg_mean" in res_df.columns
    assert "sv_edge_density_mean" in res_df.columns
    # Check that in dry-run, mock pano_ids and valid indices are populated
    assert (res_df["sv_status"] == "OK").all()
    assert not res_df["sv_exg_mean"].isna().all()


def test_excess_green_and_edge_density_in_memory():
    """Verifies Excess Green and edge density computations on synthetic in-memory images."""
    # Pure green image should have high ExG
    green_arr = np.zeros((50, 50, 3), dtype=np.uint8)
    green_arr[:, :, 1] = 255
    green_img = Image.fromarray(green_arr)
    exg_green = sv.compute_excess_green_index(green_img)
    assert exg_green > 1.5, f"Pure green should have high ExG, got {exg_green}"

    # Pure red image should have negative ExG
    red_arr = np.zeros((50, 50, 3), dtype=np.uint8)
    red_arr[:, :, 0] = 255
    red_img = Image.fromarray(red_arr)
    exg_red = sv.compute_excess_green_index(red_img)
    assert exg_red < -0.5, f"Pure red should have negative ExG, got {exg_red}"

    # Image with sharp checkerboard pattern should have positive edge density
    checker = np.zeros((50, 50), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker_img = Image.fromarray(checker)
    edge_density = sv.compute_edge_density(checker_img)
    assert 0.0 <= edge_density <= 1.0
