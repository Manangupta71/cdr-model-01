"""
stay_points.py

Implements the stay-point detection algorithm specified in:

  Toole, J. L., Colak, S., Sturt, B., Alexander, L. P., Evsukoff, A., &
  González, M. C. (2015). "The path most traveled: Travel demand
  estimation using big data resources." Transportation Research Part C:
  Emerging Technologies, 58, 162-177. (Algorithms 2-5).

Four-step procedure:

  Step 1 (Initialize): Per-subscriber time-ordered pings, distance
    threshold delta (meters) between consecutive pings, temporal
    threshold tau (seconds) for candidate stay qualification, and
    spatial grid size ds (meters) for agglomerative clustering.

  Step 2 (Candidate Stays): Sequential ping scan. Consecutive pings within
    delta are grouped into candidate sets. When distance exceeds delta,
    the time duration is evaluated against tau. If duration >= tau,
    a candidate stay is recorded at the set's spatial centroid.

  Step 3 (Agglomerative Clustering): Spatial grid snapping (cell size ds)
    merges recurring candidate stays across different times/days into
    unified stay points.

  Step 4 (Final Pass): Sweeps unassigned pings and associates them with
    the nearest stay centroid within delta distance.

Defaults: delta=200m, tau=1200s (20 min), ds=200m, consistent with
cellular tower spatial resolution in Toole et al. (2015), Section 3.1.

Execution Note: Prior noise reduction via `noise_reduction.py` is
recommended to prevent telemetry jitter from fragmenting stay points.
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

EARTH_RADIUS_M = 6371000.0
DELTA_M = 200.0          # distance threshold between consecutive candidate-set pings
TAU_SEC = 1200.0         # time threshold (20 min) for a candidate set to count as a stay
GRID_SIZE_M = 200.0      # agglomerative clustering grid cell size


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_M * c


def _centroid(pings):
    lats = [p[1] for p in pings]
    lons = [p[2] for p in pings]
    return float(np.mean(lats)), float(np.mean(lons))


# ---------------------------------------------------------------------------
# Step 2: Candidate stays
# ---------------------------------------------------------------------------

def _find_candidate_stays(pings, delta_m=DELTA_M, tau_sec=TAU_SEC):
    """
    pings: list of (timestamp, lat, lon), time-ordered, for one agent.
    Returns: list of dicts {lat, lon, arrival_time, departure_time,
             ping_idxs} — one per candidate stay (Algorithm 3).
    """
    candidate_stays = []
    candidate_set = [pings[0]] if pings else []
    candidate_idxs = [0] if pings else []

    for i in range(len(pings) - 1):
        dist = haversine_m(pings[i][1], pings[i][2], pings[i + 1][1], pings[i + 1][2])
        if dist < delta_m:
            candidate_set.append(pings[i + 1])
            candidate_idxs.append(i + 1)
        else:
            time_span = (candidate_set[-1][0] - candidate_set[0][0]).total_seconds()
            if time_span > tau_sec:
                lat, lon = _centroid(candidate_set)
                candidate_stays.append({
                    "lat": lat, "lon": lon,
                    "arrival_time": candidate_set[0][0],
                    "departure_time": candidate_set[-1][0],
                    "ping_idxs": list(candidate_idxs),
                })
            candidate_set = [pings[i + 1]]
            candidate_idxs = [i + 1]

    # flush the final candidate set
    if len(candidate_set) > 0:
        time_span = (candidate_set[-1][0] - candidate_set[0][0]).total_seconds()
        if time_span > tau_sec:
            lat, lon = _centroid(candidate_set)
            candidate_stays.append({
                "lat": lat, "lon": lon,
                "arrival_time": candidate_set[0][0],
                "departure_time": candidate_set[-1][0],
                "ping_idxs": list(candidate_idxs),
            })

    return candidate_stays


# ---------------------------------------------------------------------------
# Step 3: Agglomerative (grid-based) clustering of candidate stays
# ---------------------------------------------------------------------------

def _agglomerative_cluster(candidate_stays, grid_size_m=GRID_SIZE_M):
    """
    Collapses candidate stays that fall in the same grid cell (a simple,
    efficient stand-in for the generic agglomerative clustering step in
    Algorithm 4) into single stay points, each labeled with a stay_index.
    Returns: list of dicts {lat, lon, arrival_time, departure_time,
             ping_idxs, stay_index}.
    """
    if not candidate_stays:
        return []

    # meters-per-degree approximation is fine at city scale for grid snapping
    lat0 = candidate_stays[0]["lat"]
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * cos(radians(lat0))

    cells = {}
    for cs in candidate_stays:
        cell_row = int((cs["lat"] * m_per_deg_lat) // grid_size_m)
        cell_col = int((cs["lon"] * m_per_deg_lon) // grid_size_m)
        cells.setdefault((cell_row, cell_col), []).append(cs)

    stay_points = []
    for stay_index, (_, members) in enumerate(cells.items()):
        lat = float(np.mean([m["lat"] for m in members]))
        lon = float(np.mean([m["lon"] for m in members]))
        ping_idxs = [i for m in members for i in m["ping_idxs"]]
        arrival = min(m["arrival_time"] for m in members)
        departure = max(m["departure_time"] for m in members)
        stay_points.append({
            "stay_index": stay_index, "lat": lat, "lon": lon,
            "arrival_time": arrival, "departure_time": departure,
            "ping_idxs": ping_idxs,
        })
    return stay_points


# ---------------------------------------------------------------------------
# Step 4: Final pass — snap remaining unassigned pings to the nearest stay
# ---------------------------------------------------------------------------

def _final_pass(pings, stay_points, delta_m=DELTA_M):
    assigned = set()
    for sp in stay_points:
        assigned.update(sp["ping_idxs"])

    for i, ping in enumerate(pings):
        if i in assigned:
            continue
        best_sp, best_dist = None, None
        for sp in stay_points:
            d = haversine_m(ping[1], ping[2], sp["lat"], sp["lon"])
            if d < delta_m and (best_dist is None or d < best_dist):
                best_sp, best_dist = sp, d
        if best_sp is not None:
            best_sp["ping_idxs"].append(i)
            assigned.add(i)
            best_sp["arrival_time"] = min(best_sp["arrival_time"], ping[0])
            best_sp["departure_time"] = max(best_sp["departure_time"], ping[0])

    return stay_points


# ---------------------------------------------------------------------------
# Full per-agent / per-population driver
# ---------------------------------------------------------------------------

def extract_stay_points(pings, delta_m=DELTA_M, tau_sec=TAU_SEC, grid_size_m=GRID_SIZE_M):
    """
    pings: list of (timestamp, lat, lon), time-ordered, for a SINGLE agent.
    Returns: list of dicts {arrival_time, departure_time, lat, lon, n_points}
             (Algorithms 2-5 run in sequence).
    """
    if len(pings) < 2:
        return []
    candidate_stays = _find_candidate_stays(pings, delta_m, tau_sec)
    stay_points = _agglomerative_cluster(candidate_stays, grid_size_m)
    stay_points = _final_pass(pings, stay_points, delta_m)

    return [{
        "arrival_time": sp["arrival_time"],
        "departure_time": sp["departure_time"],
        "lat": sp["lat"],
        "lon": sp["lon"],
        "n_points": len(sp["ping_idxs"]),
    } for sp in stay_points]


def extract_stay_points_for_population(mobility_df):
    """
    mobility_df: DataFrame with columns [agent_id, timestamp, lat, lon]
                 (ideally already passed through noise_reduction.py).
    Returns: DataFrame [agent_id, arrival_time, departure_time, lat, lon, n_points]
    """
    mobility_df = mobility_df.sort_values(["agent_id", "timestamp"])
    all_stay_points = []
    for agent_id, group in mobility_df.groupby("agent_id"):
        pings = list(zip(group["timestamp"], group["lat"], group["lon"]))
        sps = extract_stay_points(pings)
        for sp in sps:
            sp["agent_id"] = agent_id
            all_stay_points.append(sp)
    return pd.DataFrame(all_stay_points, columns=[
        "agent_id", "arrival_time", "departure_time", "lat", "lon", "n_points"
    ])


if __name__ == "__main__":
    import os
    src = "data/mobility_traces_denoised.csv" if os.path.exists("data/mobility_traces_denoised.csv") \
        else "data/mobility_traces.csv"
    print(f"Reading pings from {src}")
    mobility = pd.read_csv(src, parse_dates=["timestamp"])
    stay_points_df = extract_stay_points_for_population(mobility)
    stay_points_df.to_csv("output/stay_points.csv", index=False)
    print(f"Extracted {len(stay_points_df)} stay points across {mobility['agent_id'].nunique()} agents.")
