"""
label_places.py

Labels each agent's stay points as HOME, WORK, or OTHER, following the
home/work detection rule specified in:

  Toole, J. L. et al. (2015). "The path most traveled: Travel demand
  estimation using big data resources." Transportation Research Part C,
  58, 162-177. — OD Creation Algorithm, Step 1 (Home/Work Expansion),
  reproduced here exactly:

    user.home = index of stay point visited the most between 8pm and
                7am on weekdays
    user.work = index of non-home stay point visited the most between
                7am and 8pm on weekdays
    if user visits work less than once per week: user.work = null
    every stay is then labeled home, work, or other

Note: The home/work diurnal window (20:00-07:00 vs 07:00-20:00 weekdays)
follows origin-destination modeling conventions (Toole et al., 2015), distinct
from bandicoot's standard 07:00-19:00 indicator split.

Stay points are first grouped into physical "places" (a place may be
visited across many different stay-point instances on different days)
using the same grid-cell logic as the agglomerative clustering step in
stay_points.py, so that "visited most" counts dwell time per place,
not per isolated stay-point instance.
"""

import numpy as np
import pandas as pd
from stay_points import haversine_m

CLUSTER_DIST_M = 250.0          # place-clustering tolerance (matches stay_points.py's grid size)
HOME_START_HOUR = 20            # 8pm
HOME_END_HOUR = 7               # 7am (wraps past midnight)
WORK_START_HOUR = 7             # 7am
WORK_END_HOUR = 20              # 8pm
MIN_WORK_VISITS_PER_WEEK = 1.0  # "visits work less than once per week" -> work = null


def _cluster_places(stay_points_agent, cluster_dist_m=CLUSTER_DIST_M):
    """Greedy single-link clustering of an agent's stay points into 'places'."""
    places = []  # each: {"lat":..., "lon":..., "stay_point_idxs": [...]}
    for idx, sp in stay_points_agent.iterrows():
        assigned = False
        for place in places:
            d = haversine_m(sp["lat"], sp["lon"], place["lat"], place["lon"])
            if d <= cluster_dist_m:
                place["stay_point_idxs"].append(idx)
                idxs = place["stay_point_idxs"]
                place["lat"] = stay_points_agent.loc[idxs, "lat"].mean()
                place["lon"] = stay_points_agent.loc[idxs, "lon"].mean()
                assigned = True
                break
        if not assigned:
            places.append({"lat": sp["lat"], "lon": sp["lon"], "stay_point_idxs": [idx]})
    return places


def _place_visit_stats(stay_points_agent, idxs, n_weeks):
    """Per-place dwell time and visit count within the home/work weekday windows."""
    home_minutes = 0.0
    work_minutes = 0.0
    n_visits = 0
    for idx in idxs:
        row = stay_points_agent.loc[idx]
        arrival = pd.Timestamp(row["arrival_time"])
        departure = pd.Timestamp(row["departure_time"])
        duration_min = max((departure - arrival).total_seconds() / 60.0, 1.0)
        is_weekday = arrival.dayofweek < 5
        hour = arrival.hour

        if is_weekday:
            n_visits += 1
            is_home_hour = (hour >= HOME_START_HOUR) or (hour < HOME_END_HOUR)
            is_work_hour = WORK_START_HOUR <= hour < WORK_END_HOUR
            if is_home_hour:
                home_minutes += duration_min
            if is_work_hour:
                work_minutes += duration_min

    visits_per_week = n_visits / max(n_weeks, 1e-6)
    return {"home_minutes": home_minutes, "work_minutes": work_minutes,
            "visits_per_week": visits_per_week}


def label_places_for_agent(stay_points_agent, n_weeks=2.0):
    """
    stay_points_agent: DataFrame slice (single agent_id) of stay points,
                        columns [arrival_time, departure_time, lat, lon, n_points]
    n_weeks: length of the observation window in weeks, for the
             "less than once per week" work-visit test.
    Returns: DataFrame with an added 'place_id' column grouping stay
             points into places, and 'place_label' (home/work/other).
    """
    stay_points_agent = stay_points_agent.reset_index(drop=True)
    places = _cluster_places(stay_points_agent)

    scored = []
    for pid, place in enumerate(places):
        stats = _place_visit_stats(stay_points_agent, place["stay_point_idxs"], n_weeks)
        scored.append({"place_id": pid, **stats, **place})

    # home = stay point visited most (by dwell time, 8pm-7am weekdays)
    home_id = max(scored, key=lambda p: p["home_minutes"])["place_id"] if scored else None
    if scored and max(p["home_minutes"] for p in scored) <= 0:
        home_id = None  # no weekday-night observations at all -> no home inferable

    # work = highest-dwell non-home place (7am-8pm weekdays), nulled if
    # visited less than once per week
    work_candidates = [p for p in scored if p["place_id"] != home_id and p["work_minutes"] > 0]
    work_id = None
    if work_candidates:
        best_work = max(work_candidates, key=lambda p: p["work_minutes"])
        if best_work["visits_per_week"] >= MIN_WORK_VISITS_PER_WEEK:
            work_id = best_work["place_id"]

    label_map = {}
    for p in scored:
        if p["place_id"] == home_id:
            label_map[p["place_id"]] = "home"
        elif p["place_id"] == work_id:
            label_map[p["place_id"]] = "work"
        else:
            label_map[p["place_id"]] = "other"

    place_label = np.empty(len(stay_points_agent), dtype=object)
    place_id_col = np.empty(len(stay_points_agent), dtype=int)
    for place in scored:
        for idx in place["stay_point_idxs"]:
            place_label[idx] = label_map[place["place_id"]]
            place_id_col[idx] = place["place_id"]

    out = stay_points_agent.copy()
    out["place_id"] = place_id_col
    out["place_label"] = place_label
    return out


def label_places_for_population(stay_points_df, sim_days=14):
    n_weeks = max(sim_days / 7.0, 1.0)
    labeled = []
    for agent_id, group in stay_points_df.groupby("agent_id"):
        labeled_agent = label_places_for_agent(group, n_weeks=n_weeks)
        labeled_agent["agent_id"] = agent_id
        labeled.append(labeled_agent)
    return pd.concat(labeled, ignore_index=True) if labeled else pd.DataFrame(
        columns=["agent_id", "arrival_time", "departure_time", "lat", "lon",
                 "n_points", "place_id", "place_label"]
    )


def build_home_work_anchors(labeled_stay_points_df):
    """Produces one row per agent with their inferred home/work centroid."""
    rows = []
    for agent_id, group in labeled_stay_points_df.groupby("agent_id"):
        home = group[group["place_label"] == "home"]
        work = group[group["place_label"] == "work"]
        rows.append({
            "agent_id": agent_id,
            "inferred_home_lat": home["lat"].mean() if len(home) else np.nan,
            "inferred_home_lon": home["lon"].mean() if len(home) else np.nan,
            "inferred_work_lat": work["lat"].mean() if len(work) else np.nan,
            "inferred_work_lon": work["lon"].mean() if len(work) else np.nan,
            "has_inferred_work": len(work) > 0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import generate_bengaluru_data as gen

    stay_points_df = pd.read_csv(
        "output/stay_points.csv", parse_dates=["arrival_time", "departure_time"]
    )
    labeled = label_places_for_population(stay_points_df, sim_days=gen.SIM_DAYS)
    labeled.to_csv("output/stay_points_labeled.csv", index=False)

    anchors = build_home_work_anchors(labeled)
    anchors.to_csv("output/home_work_anchors.csv", index=False)

    print(f"Labeled {len(labeled)} stay points into places for {labeled['agent_id'].nunique()} agents.")
    print(f"Work anchor inferred for {anchors['has_inferred_work'].sum()}/{len(anchors)} agents.")
