"""
spatial_context.py

Spatial context feature extraction and validation protocol for synthetic CDR
socio-demographic prediction pipeline.

Extracts zone-level spatial attributes for inferred home and work locations
derived from mobility traces (home_work_anchors.csv), and loads Street View-derived
environmental indices (streetview_features.csv) when available.

Ground-Truth Separation:
  Uses exclusively inferred anchor coordinates (inferred_home_lat, inferred_home_lon,
  inferred_work_lat, inferred_work_lon) from the pipeline's stay-point labeling stage.
  Never accesses ground-truth coordinates or zone assignments from population.csv.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from generate_bengaluru_data import ZONES, Zone


def nearest_zone(
    lat: Union[float, np.ndarray, pd.Series],
    lon: Union[float, np.ndarray, pd.Series],
    zones: List[Zone] = ZONES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the nearest zone index and Euclidean distance in kilometers
    to the zone centroid.

    Parameters
    ----------
    lat : float or array-like
        Latitude coordinate(s).
    lon : float or array-like
        Longitude coordinate(s).
    zones : list of Zone, default ZONES
        List of reference zones with centroid coordinates.

    Returns
    -------
    idx : np.ndarray
        Array of nearest zone indices (0-indexed).
    dist_km : np.ndarray
        Array of approximate distances to zone centroid in kilometers.
    """
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=float))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=float))

    zone_coords = np.array([[z.lat, z.lon] for z in zones])
    # Euclidean planar distance in degrees
    d = np.sqrt(
        (zone_coords[:, 0][None, :] - lat_arr[:, None]) ** 2
        + (zone_coords[:, 1][None, :] - lon_arr[:, None]) ** 2
    )
    idx = d.argmin(axis=1)
    # 1 degree ~ 111.0 km (sufficient approximation at urban Bengaluru scale)
    dist_km = d[np.arange(len(idx)), idx] * 111.0
    return idx, dist_km


def _resolve_phone_numbers(anchors_df: pd.DataFrame, root: str = ".") -> pd.DataFrame:
    """
    Ensures anchors DataFrame has a 'phone_number' identifier column.
    Uses population.csv identifier mapping if present, otherwise computes BLR phone format.
    Never uses any spatial or demographic fields from population.csv.
    """
    df = anchors_df.copy()
    if "phone_number" in df.columns:
        return df

    pop_path = os.path.join(root, "data", "population.csv")
    if os.path.exists(pop_path):
        pop_id_map = pd.read_csv(pop_path, usecols=["agent_id", "phone_number"])
        df = df.merge(pop_id_map, on="agent_id", how="left")
    else:
        df["phone_number"] = df["agent_id"].apply(lambda aid: f"BLR{100000000 + int(aid)}")
    return df


def build_zone_context(root: str = ".") -> pd.DataFrame:
    """
    Builds zone-level spatial context features from inferred home/work anchors.

    From output/home_work_anchors.csv, assigns each agent's inferred home
    and work anchors to the nearest ZONE in generate_bengaluru_data.py.
    Returns tabular features keyed by phone_number:
      - ctx_home_density_weight: Residential density weight of inferred home zone
      - ctx_home_employment_weight: Employment density weight of inferred home zone
      - ctx_home_dist_to_zone_centre_km: Distance from inferred home anchor to zone center (km)
      - ctx_work_employment_weight: Employment density weight of inferred work zone (0.0 if no work anchor)

    Parameters
    ----------
    root : str, default "."
        Root directory containing output/home_work_anchors.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with phone_number and ctx_* feature columns.
    """
    anchors_path = os.path.join(root, "output", "home_work_anchors.csv")
    if not os.path.exists(anchors_path):
        raise FileNotFoundError(f"Inferred anchors file not found at: {anchors_path}")

    anchors = pd.read_csv(anchors_path)
    anchors = _resolve_phone_numbers(anchors, root=root)

    # Nearest home zone from inferred home anchor
    home_idx, home_dist_km = nearest_zone(
        anchors["inferred_home_lat"], anchors["inferred_home_lon"], ZONES
    )

    ctx = pd.DataFrame({"phone_number": anchors["phone_number"]})
    ctx["ctx_home_density_weight"] = [ZONES[i].density_weight for i in home_idx]
    ctx["ctx_home_employment_weight"] = [ZONES[i].employment_weight for i in home_idx]
    ctx["ctx_home_dist_to_zone_centre_km"] = home_dist_km

    # Nearest work zone from inferred work anchor
    work_lat = anchors["inferred_work_lat"].fillna(0.0)
    work_lon = anchors["inferred_work_lon"].fillna(0.0)
    work_idx, _ = nearest_zone(work_lat, work_lon, ZONES)

    has_work = anchors["has_inferred_work"].to_numpy().astype(bool)
    work_emp_weights = np.array([ZONES[i].employment_weight for i in work_idx])
    ctx["ctx_work_employment_weight"] = np.where(has_work, work_emp_weights, 0.0)

    return ctx


def get_inferred_home_zone_groups(root: str = ".") -> Dict[str, int]:
    """
    Maps each agent's phone_number to the nearest zone index of their
    inferred home anchor. Used for grouped cross-validation splitting.

    Parameters
    ----------
    root : str, default "."
        Root directory of the repo.

    Returns
    -------
    dict
        Mapping of phone_number -> integer zone index.
    """
    anchors_path = os.path.join(root, "output", "home_work_anchors.csv")
    if not os.path.exists(anchors_path):
        raise FileNotFoundError(f"Inferred anchors file not found at: {anchors_path}")

    anchors = pd.read_csv(anchors_path)
    anchors = _resolve_phone_numbers(anchors, root=root)
    home_idx, _ = nearest_zone(
        anchors["inferred_home_lat"], anchors["inferred_home_lon"], ZONES
    )
    return dict(zip(anchors["phone_number"], home_idx))


# Convenience alias
get_home_zone_groups = get_inferred_home_zone_groups


def load_streetview_features(
    path: str = "output/streetview_features.csv",
) -> Optional[pd.DataFrame]:
    """
    Loads Street View-derived feature table if present on disk.

    Extracts numeric sv_* feature columns along with phone_number identifier,
    excluding metadata strings (sv_status, sv_date, sv_pano_id).
    Returns None if the file does not exist.

    Parameters
    ----------
    path : str, default "output/streetview_features.csv"
        File path to streetview_features.csv.

    Returns
    -------
    pd.DataFrame or None
        Numeric Street View features DataFrame or None if absent.
    """
    if not os.path.exists(path):
        return None

    try:
        sv_df = pd.read_csv(path)
    except Exception:
        return None

    if "phone_number" not in sv_df.columns:
        return None

    non_feature_cols = {"sv_status", "sv_date", "sv_pano_id"}
    feature_cols = [
        c for c in sv_df.columns
        if c.startswith("sv_") and c not in non_feature_cols
    ]

    if not feature_cols:
        return None

    return sv_df[["phone_number"] + feature_cols].copy()


def make_shuffled_control(
    df: Optional[pd.DataFrame],
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Negative control helper that permutes feature columns across agents.

    Maintains identical marginal distributions, feature means, and column variances,
    while completely breaking the association between the individual agent and
    their spatial context features.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame containing features and 'phone_number'.
    feature_cols : list of str, optional
        Specific columns to permute. If None, all columns except 'phone_number'
        are permuted.
    random_state : int, default 42
        Seed for the random permutation generator.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with permuted columns, or None if df is None.
    """
    if df is None:
        return None

    shuffled = df.copy()
    rng = np.random.default_rng(random_state)

    if feature_cols is None:
        cols_to_shuffle = [c for c in shuffled.columns if c != "phone_number"]
    else:
        cols_to_shuffle = [c for c in feature_cols if c in shuffled.columns and c != "phone_number"]

    for col in cols_to_shuffle:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())

    return shuffled
