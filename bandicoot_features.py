"""
bandicoot_features.py

Computes the standard mobile-phone-metadata behavioral indicator schema
popularized by the `bandicoot` toolbox:

  de Montjoye, Y.-A., Rocher, L., & Pentland, A. S. (2016). bandicoot: a
  Python toolbox for mobile phone metadata. Journal of Machine Learning
  Research, 17(175), 1-5.

from the raw synthetic CDR streams produced by generate_bengaluru_data.py
(communication_events.csv, mobility_traces.csv + antennas.csv,
recharge_events.csv). Every indicator is computed per phone_number, split
across the standard 3x3 grid of:

  week part:  allweek / weekday / weekend
  day part:   allday  / day     / night     (day := 07:00-19:00 local)

and, where the indicator concerns interactions, per channel (call / text /
callandtext). Distributional indicators (call_duration, response_delay_text,
balance_of_contacts, interactions_per_contact, interevent_time,
amount_recharges, interevent_time_recharges) are further expanded into
seven summary statistics: mean, std, median, skewness, kurtosis, min, max.

Implements standard operational definitions for bandicoot behavioral indicators
(conversation grouping, Pareto thresholds, response latency, and churn rate).
Operational parameters are documented inline.
"""

import warnings
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# scipy warns on near-constant tiny samples (e.g. two nearly-identical call
# durations) when computing skew/kurtosis; the result (~0) is harmless here.
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

WEEK_PARTS = ["allweek", "weekday", "weekend"]
DAY_PARTS = ["allday", "day", "night"]
DAY_START_HOUR = 7   # Default day/night boundary (07:00 local)
DAY_END_HOUR = 19
STATS = ["mean", "std", "median", "skewness", "kurtosis", "min", "max"]
CONVERSATION_GAP_SEC = 3600      # Interactions >1h apart delineate a new conversation
TEXT_REPLY_WINDOW_SEC = 3600     # Maximum latency to associate an outgoing reply with an incoming text
PARETO_FRACTION = 0.8


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _dist_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {s: 0.0 for s in STATS}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "std": 0.0, "median": v, "skewness": 0.0,
                "kurtosis": 0.0, "min": v, "max": v}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skew_val = skew(values)
        kurt_val = kurtosis(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "skewness": float(skew_val) if np.isfinite(skew_val) else 0.0,
        "kurtosis": float(kurt_val) if np.isfinite(kurt_val) else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _pareto_fraction_of_units(sorted_desc_values, fraction=PARETO_FRACTION):
    """Min number of units (as a fraction of total units) needed to reach
    `fraction` of the total sum, given values already sorted descending."""
    total = sorted_desc_values.sum()
    if total <= 0 or len(sorted_desc_values) == 0:
        return 0.0
    cum = np.cumsum(sorted_desc_values)
    n_needed = int(np.searchsorted(cum, fraction * total) + 1)
    return n_needed / len(sorted_desc_values)


def _week_mask(ts, week_part):
    if week_part == "allweek":
        return np.ones(len(ts), dtype=bool)
    wd = ts.dt.dayofweek.to_numpy()
    return (wd < 5) if week_part == "weekday" else (wd >= 5)


def _day_mask(ts, day_part):
    if day_part == "allday":
        return np.ones(len(ts), dtype=bool)
    hour = ts.dt.hour.to_numpy()
    is_day = (hour >= DAY_START_HOUR) & (hour < DAY_END_HOUR)
    return is_day if day_part == "day" else ~is_day


def _filter(df, week_part, day_part):
    mask = _week_mask(df["timestamp"], week_part) & _day_mask(df["timestamp"], day_part)
    return df[mask]


# ---------------------------------------------------------------------------
# Build a per-agent, per-perspective interaction table from raw comm events
# ---------------------------------------------------------------------------

def _build_interaction_table(comm_events_df):
    """
    Each raw event (caller -> callee) becomes two rows: one from the
    caller's perspective (direction='out') and one from the callee's
    ('in'). This lets every downstream indicator be computed per agent
    without re-deriving direction each time.
    """
    out_rows = comm_events_df.rename(columns={"caller_id": "agent_id", "callee_id": "contact_id"}).copy()
    out_rows["direction"] = "out"
    in_rows = comm_events_df.rename(columns={"callee_id": "agent_id", "caller_id": "contact_id"}).copy()
    in_rows["direction"] = "in"
    combined = pd.concat([out_rows, in_rows], ignore_index=True)
    return combined.sort_values(["agent_id", "timestamp"]).reset_index(drop=True)


def _conversation_ids(sub_df):
    """Groups a (single-agent, single-contact, time-sorted) event slice into
    conversations separated by gaps > CONVERSATION_GAP_SEC."""
    ts = sub_df["timestamp"].to_numpy()
    if len(ts) == 0:
        return np.array([], dtype=int)
    gaps = np.diff(ts) / np.timedelta64(1, "s")
    new_conv = np.concatenate([[True], gaps > CONVERSATION_GAP_SEC])
    return np.cumsum(new_conv) - 1


# ---------------------------------------------------------------------------
# Per-agent, per-(week_part, day_part) indicator block
# ---------------------------------------------------------------------------

def _interaction_indicators(agent_events, channel_label, event_types):
    """
    agent_events: interaction table already filtered to one agent and one
                  (week_part, day_part) window.
    channel_label: 'call', 'text', or 'callandtext' (used in output keys)
    event_types: which event_type values belong to this channel
    """
    feats = {}
    sub = agent_events[agent_events["event_type"].isin(event_types)]

    n_in = int((sub["direction"] == "in").sum())
    n_out = int((sub["direction"] == "out").sum())
    n_total = n_in + n_out

    feats[f"number_of_interactions__{channel_label}"] = n_total
    feats[f"number_of_interaction_in__{channel_label}"] = n_in
    feats[f"number_of_interaction_out__{channel_label}"] = n_out
    feats[f"number_of_contacts__{channel_label}"] = int(sub["contact_id"].nunique())

    # NOTE: the target schema only defines percent_initiated_interactions
    # for the 'call' channel (not 'text') — matched deliberately here.
    if channel_label == "call":
        feats[f"percent_initiated_interactions__{channel_label}"] = (
            n_out / n_total if n_total > 0 else 0.0
        )

    hour = sub["timestamp"].dt.hour
    is_night = (hour < DAY_START_HOUR) | (hour >= DAY_END_HOUR)
    feats[f"percent_nocturnal__{channel_label}"] = (
        float(is_night.mean()) if len(sub) > 0 else 0.0
    )

    # per-contact aggregates: count, balance, interactions-per-contact
    if len(sub) > 0:
        by_contact = sub.groupby("contact_id")
        counts = by_contact.size().to_numpy()
        in_counts = sub[sub["direction"] == "in"].groupby("contact_id").size()
        out_counts = sub[sub["direction"] == "out"].groupby("contact_id").size()
        all_contacts = sub["contact_id"].unique()
        balances = []
        for c in all_contacts:
            ci = int(in_counts.get(c, 0))
            co = int(out_counts.get(c, 0))
            denom = ci + co
            balances.append((co - ci) / denom if denom > 0 else 0.0)
        balances = np.array(balances)

        feats[f"entropy_of_contacts__{channel_label}"] = _shannon_entropy(counts)
        for stat, val in _dist_stats(counts).items():
            feats[f"interactions_per_contact__{channel_label}__{stat}"] = val
        for stat, val in _dist_stats(balances).items():
            feats[f"balance_of_contacts__{channel_label}__{stat}"] = val

        sorted_counts = np.sort(counts)[::-1]
        feats[f"percent_pareto_interactions__{channel_label}"] = _pareto_fraction_of_units(sorted_counts)

        # interevent time across ALL interactions in this channel (not per contact)
        ts_sorted = np.sort(sub["timestamp"].to_numpy())
        if len(ts_sorted) > 1:
            gaps = np.diff(ts_sorted) / np.timedelta64(1, "s")
        else:
            gaps = np.array([])
        for stat, val in _dist_stats(gaps).items():
            feats[f"interevent_time__{channel_label}__{stat}"] = val
    else:
        feats[f"entropy_of_contacts__{channel_label}"] = 0.0
        for stat in STATS:
            feats[f"interactions_per_contact__{channel_label}__{stat}"] = 0.0
            feats[f"balance_of_contacts__{channel_label}__{stat}"] = 0.0
            feats[f"interevent_time__{channel_label}__{stat}"] = 0.0
        feats[f"percent_pareto_interactions__{channel_label}"] = 0.0

    return feats, sub


def _call_specific_indicators(call_sub):
    feats = {}
    for stat, val in _dist_stats(call_sub["duration_sec"].to_numpy()).items():
        feats[f"call_duration__call__{stat}"] = val
    if len(call_sub) > 0:
        by_contact_dur = call_sub.groupby("contact_id")["duration_sec"].sum()
        sorted_durs = np.sort(by_contact_dur.to_numpy())[::-1]
        feats["percent_pareto_durations__call"] = _pareto_fraction_of_units(sorted_durs)
    else:
        feats["percent_pareto_durations__call"] = 0.0
    return feats


def _text_specific_indicators(text_sub):
    """response_delay_text and response_rate_text: for each incoming text,
    look for the next outgoing text to the same contact within
    TEXT_REPLY_WINDOW_SEC and treat that gap as the response delay."""
    feats = {}
    delays = []
    n_incoming = 0
    n_responded = 0
    for contact_id, grp in text_sub.groupby("contact_id"):
        grp = grp.sort_values("timestamp")
        rows = list(grp.itertuples())
        for i, row in enumerate(rows):
            if row.direction != "in":
                continue
            n_incoming += 1
            for row2 in rows[i + 1:]:
                if row2.direction == "out":
                    delay = (row2.timestamp - row.timestamp).total_seconds()
                    if delay <= TEXT_REPLY_WINDOW_SEC:
                        delays.append(delay)
                        n_responded += 1
                    break
                if (row2.timestamp - row.timestamp).total_seconds() > TEXT_REPLY_WINDOW_SEC:
                    break

    for stat, val in _dist_stats(np.array(delays)).items():
        feats[f"response_delay_text__callandtext__{stat}"] = val
    feats["response_rate_text__callandtext"] = (n_responded / n_incoming) if n_incoming > 0 else 0.0
    return feats


def _conversation_indicators(sub_all):
    """percent_initiated_conversations, combined call+text, grouped into
    conversations per contact separated by CONVERSATION_GAP_SEC."""
    if len(sub_all) == 0:
        return {"percent_initiated_conversations__callandtext": 0.0}
    initiated = []
    for contact_id, grp in sub_all.groupby("contact_id"):
        grp = grp.sort_values("timestamp")
        conv_ids = _conversation_ids(grp)
        grp = grp.assign(conv_id=conv_ids)
        first_per_conv = grp.groupby("conv_id").first()
        initiated.extend((first_per_conv["direction"] == "out").tolist())
    return {"percent_initiated_conversations__callandtext": float(np.mean(initiated)) if initiated else 0.0}


# ---------------------------------------------------------------------------
# Spatial indicators (antenna-based)
# ---------------------------------------------------------------------------

def _spatial_indicators(agent_pings, home_antenna_id):
    feats = {}
    n = len(agent_pings)
    if n == 0:
        feats["number_of_antennas"] = 0
        feats["entropy_of_antennas"] = 0.0
        feats["percent_at_home"] = 0.0
        feats["radius_of_gyration"] = 0.0
        feats["frequent_antennas"] = 0
        return feats

    antenna_counts = agent_pings["antenna_id"].value_counts()
    feats["number_of_antennas"] = int(agent_pings["antenna_id"].nunique())
    feats["entropy_of_antennas"] = _shannon_entropy(antenna_counts.to_numpy())
    feats["percent_at_home"] = float((agent_pings["antenna_id"] == home_antenna_id).mean())

    lat_c, lon_c = agent_pings["lat"].mean(), agent_pings["lon"].mean()
    d = np.sqrt((agent_pings["lat"] - lat_c) ** 2 + (agent_pings["lon"] - lon_c) ** 2) * 111000
    feats["radius_of_gyration"] = float(np.sqrt((d ** 2).mean()))

    sorted_counts = np.sort(antenna_counts.to_numpy())[::-1]
    n_needed = int(_pareto_fraction_of_units(sorted_counts) * len(sorted_counts))
    feats["frequent_antennas"] = max(n_needed, 1 if len(sorted_counts) > 0 else 0)
    return feats


# ---------------------------------------------------------------------------
# Recharge (financial) indicators
# ---------------------------------------------------------------------------

def _recharge_indicators(agent_recharges):
    feats = {}
    n = len(agent_recharges)
    feats["number_of_recharges"] = n
    for stat, val in _dist_stats(agent_recharges["amount"].to_numpy() if n else np.array([])).items():
        feats[f"amount_recharges__{stat}"] = val

    if n > 1:
        ts_sorted = np.sort(agent_recharges["timestamp"].to_numpy())
        gaps = np.diff(ts_sorted) / np.timedelta64(1, "D")  # days, matching recharge cadence
    else:
        gaps = np.array([])
    for stat, val in _dist_stats(gaps).items():
        feats[f"interevent_time_recharges__{stat}"] = val

    if n > 0:
        sorted_amounts = np.sort(agent_recharges["amount"].to_numpy())[::-1]
        feats["percent_pareto_recharges"] = _pareto_fraction_of_units(sorted_amounts)
    else:
        feats["percent_pareto_recharges"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# churn_rate: week-over-week volatility in total interaction volume
# ---------------------------------------------------------------------------

def _churn_rate(agent_events, sim_days):
    if len(agent_events) == 0:
        return {"churn_rate__mean": 0.0, "churn_rate__std": 0.0}
    agent_events = agent_events.copy()
    agent_events["week"] = (agent_events["timestamp"] - agent_events["timestamp"].min()).dt.days // 7
    weekly_counts = agent_events.groupby("week").size()
    if len(weekly_counts) < 2:
        return {"churn_rate__mean": 0.0, "churn_rate__std": 0.0}
    counts = weekly_counts.to_numpy().astype(float)
    prev = counts[:-1]
    curr = counts[1:]
    rel_change = np.where(prev > 0, (curr - prev) / prev, 0.0)
    return {"churn_rate__mean": float(np.mean(rel_change)), "churn_rate__std": float(np.std(rel_change, ddof=1) if len(rel_change) > 1 else 0.0)}


# ---------------------------------------------------------------------------
# Full per-agent feature row
# ---------------------------------------------------------------------------

def _reorder_key(key, week_part, day_part):
    """
    Internal indicator functions build keys as 'metric__channel' or
    'metric__channel__stat' (channel/stat may be absent). The target
    schema wants 'metric__weekpart__daypart__channel[__stat]'. Since no
    metric name contains a double underscore, splitting on '__' and
    re-inserting the week/day tokens right after the metric name works
    uniformly across every indicator.
    """
    parts = key.split("__")
    metric, rest = parts[0], parts[1:]
    return "__".join([metric, week_part, day_part] + rest)


def _place_and_anchor_indicators(agent_id, anchors_dict, stays_by_agent):
    """
    Extracts high-level spatial anchor and stay-point activity indicators:
      - Commute distance (meters) between inferred home and work centroids.
      - Number of distinct physical activity places visited.
      - Place dwell-time Shannon entropy.
      - Total and average stay duration.
      - Fraction of stay dwell time spent at home and work anchors.

    References:
      - Alexander, L., Jiang, S., Murga, M., & González, M. C. (2015).
        "Origin–destination trips by purpose and time of day inferred from mobile phone data."
        Transportation Research Part C: Emerging Technologies, 58, 240-250.
        https://doi.org/10.1016/j.trc.2015.02.018
      - Pappalardo, L., Pedreschi, D., Smoreda, Z., & Giannotti, F. (2015).
        "Using big data to study the link between human mobility and socio-economic status."
        Journal of The Royal Society Interface, 12(113), 20150597.
        https://doi.org/10.1098/rsif.2015.0597
    """
    from stay_points import haversine_m

    feats = {
        "commute_distance_m": 0.0,
        "has_inferred_work": 0,
        "number_of_places": 0,
        "entropy_of_places": 0.0,
        "total_stay_duration_min": 0.0,
        "mean_stay_duration_min": 0.0,
        "percent_dwell_at_home": 0.0,
        "percent_dwell_at_work": 0.0,
    }

    anchor = anchors_dict.get(agent_id, None)
    if anchor is not None:
        h_lat, h_lon = anchor.get("inferred_home_lat"), anchor.get("inferred_home_lon")
        w_lat, w_lon = anchor.get("inferred_work_lat"), anchor.get("inferred_work_lon")
        has_work = bool(anchor.get("has_inferred_work", False))
        feats["has_inferred_work"] = 1 if has_work else 0

        if has_work and np.isfinite(h_lat) and np.isfinite(w_lat):
            feats["commute_distance_m"] = float(haversine_m(h_lat, h_lon, w_lat, w_lon))

    stays = stays_by_agent.get(agent_id, None)
    if stays is not None and len(stays) > 0:
        feats["number_of_places"] = int(stays["place_id"].nunique()) if "place_id" in stays.columns else 0

        # Dwell time per stay
        arr = pd.to_datetime(stays["arrival_time"])
        dep = pd.to_datetime(stays["departure_time"])
        durations = np.maximum((dep - arr).dt.total_seconds().to_numpy() / 60.0, 1.0)
        total_dur = float(np.sum(durations))
        feats["total_stay_duration_min"] = total_dur
        feats["mean_stay_duration_min"] = float(np.mean(durations))

        if "place_id" in stays.columns and total_dur > 0:
            place_durations = pd.Series(durations).groupby(stays["place_id"].to_numpy()).sum().to_numpy()
            feats["entropy_of_places"] = _shannon_entropy(place_durations)

        if "place_label" in stays.columns and total_dur > 0:
            home_mask = (stays["place_label"] == "home").to_numpy()
            work_mask = (stays["place_label"] == "work").to_numpy()
            feats["percent_dwell_at_home"] = float(np.sum(durations[home_mask]) / total_dur)
            feats["percent_dwell_at_work"] = float(np.sum(durations[work_mask]) / total_dur)

    return feats


def compute_agent_features(agent_id, phone_number, interactions_all, mobility_all,
                            recharges_all, home_antenna_id, sim_days,
                            anchors_dict=None, stays_by_agent=None):
    row = {"phone_number": phone_number}

    agent_events = interactions_all[interactions_all["agent_id"] == agent_id]
    agent_pings = mobility_all[mobility_all["agent_id"] == agent_id]
    agent_recharges = recharges_all[recharges_all["agent_id"] == agent_id]

    row["reporting__number_of_records"] = int(len(agent_events) + len(agent_recharges))

    for week_part in WEEK_PARTS:
        for day_part in DAY_PARTS:
            sub_all = _filter(agent_events, week_part, day_part)

            active_days = int(sub_all["timestamp"].dt.date.nunique()) if len(sub_all) else 0
            row[f"active_days__{week_part}__{day_part}__callandtext"] = active_days

            feats_call, call_sub = _interaction_indicators(sub_all, "call", ["call"])
            feats_text, text_sub = _interaction_indicators(sub_all, "text", ["text"])
            row.update({_reorder_key(k, week_part, day_part): v for k, v in feats_call.items()})
            row.update({_reorder_key(k, week_part, day_part): v for k, v in feats_text.items()})

            call_feats = _call_specific_indicators(call_sub)
            row.update({_reorder_key(k, week_part, day_part): v for k, v in call_feats.items()})

            text_feats = _text_specific_indicators(text_sub)
            row.update({_reorder_key(k, week_part, day_part): v for k, v in text_feats.items()})

            conv_feats = _conversation_indicators(sub_all)
            row.update({_reorder_key(k, week_part, day_part): v for k, v in conv_feats.items()})

            sub_pings = _filter(agent_pings, week_part, day_part) if len(agent_pings) else agent_pings
            spatial_feats = _spatial_indicators(sub_pings, home_antenna_id)
            row.update({_reorder_key(k, week_part, day_part): v for k, v in spatial_feats.items()})

            sub_recharges = _filter(agent_recharges, week_part, day_part) if len(agent_recharges) else agent_recharges
            recharge_feats = _recharge_indicators(sub_recharges)
            row.update({_reorder_key(k, week_part, day_part): v for k, v in recharge_feats.items()})

    row.update(_churn_rate(agent_events, sim_days))
    row["average_balance_recharges"] = (
        float(agent_recharges["balance_after"].mean()) if len(agent_recharges) else 0.0
    )

    # Inferred spatial anchor and stay-point features (Alexander et al., 2015; Pappalardo et al., 2015)
    if anchors_dict is not None and stays_by_agent is not None:
        place_feats = _place_and_anchor_indicators(agent_id, anchors_dict, stays_by_agent)
        row.update(place_feats)

    return row


def build_bandicoot_feature_table(population_df, comm_events_df, mobility_df,
                                   antennas_df, recharge_events_df, sim_days,
                                   anchors_df=None, labeled_stays_df=None):
    interactions_all = _build_interaction_table(comm_events_df)

    # Map nearest antenna to inferred home anchor (eliminating ground-truth coordinate leakage)
    from scipy.spatial import cKDTree
    tree = cKDTree(antennas_df[["lat", "lon"]].to_numpy())

    anchors_dict = {}
    if anchors_df is not None and len(anchors_df) > 0:
        anchors_dict = {row["agent_id"]: row.to_dict() for _, row in anchors_df.iterrows()}

    stays_by_agent = {}
    if labeled_stays_df is not None and len(labeled_stays_df) > 0:
        for aid, grp in labeled_stays_df.groupby("agent_id"):
            stays_by_agent[aid] = grp

    # Determine home antenna per agent strictly from INFERRED anchors or nighttime mobility pings
    home_antenna = {}
    for agent in population_df.itertuples():
        aid = agent.agent_id
        anchor = anchors_dict.get(aid, None)
        if anchor and np.isfinite(anchor.get("inferred_home_lat", np.nan)):
            h_lat, h_lon = anchor["inferred_home_lat"], anchor["inferred_home_lon"]
        else:
            # Fallback: estimate home centroid from nighttime pings (20:00-07:00)
            agent_pings = mobility_df[mobility_df["agent_id"] == aid]
            night_pings = agent_pings[(agent_pings["timestamp"].dt.hour < 7) | (agent_pings["timestamp"].dt.hour >= 20)]
            if len(night_pings) > 0:
                h_lat, h_lon = night_pings["lat"].mean(), night_pings["lon"].mean()
            elif len(agent_pings) > 0:
                h_lat, h_lon = agent_pings["lat"].mean(), agent_pings["lon"].mean()
            else:
                h_lat, h_lon = antennas_df["lat"].mean(), antennas_df["lon"].mean()

        _, home_idx = tree.query([[h_lat, h_lon]])
        home_antenna[aid] = antennas_df["antenna_id"].iloc[home_idx[0]]

    rows = []
    for agent in population_df.itertuples():
        row = compute_agent_features(
            agent.agent_id, agent.phone_number, interactions_all, mobility_df,
            recharge_events_df, home_antenna[agent.agent_id], sim_days,
            anchors_dict=anchors_dict, stays_by_agent=stays_by_agent
        )
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    population = pd.read_csv("data/population.csv")
    comm_events = pd.read_csv("data/communication_events.csv", parse_dates=["timestamp"])
    antennas = pd.read_csv("data/antennas.csv")
    recharges = pd.read_csv("data/recharge_events.csv", parse_dates=["timestamp"])

    mobility_path = "data/mobility_traces_denoised.csv" if os.path.exists("data/mobility_traces_denoised.csv") \
        else "data/mobility_traces.csv"
    print(f"Reading mobility pings from {mobility_path}")
    mobility = pd.read_csv(mobility_path, parse_dates=["timestamp"])

    # Load inferred place anchors (Toole et al., 2015; Alexander et al., 2015)
    anchors = None
    labeled_stays = None
    if os.path.exists("output/home_work_anchors.csv"):
        print("Reading inferred home/work anchors from output/home_work_anchors.csv (no ground-truth coordinate leakage)...")
        anchors = pd.read_csv("output/home_work_anchors.csv")
    if os.path.exists("output/stay_points_labeled.csv"):
        print("Reading labeled stay points from output/stay_points_labeled.csv...")
        labeled_stays = pd.read_csv("output/stay_points_labeled.csv", parse_dates=["arrival_time", "departure_time"])

    import generate_bengaluru_data as gen
    features = build_bandicoot_feature_table(
        population, comm_events, mobility, antennas, recharges, gen.SIM_DAYS,
        anchors_df=anchors, labeled_stays_df=labeled_stays
    )
    features.to_csv("output/bandicoot_features.csv", index=False)
    print(f"Bandicoot-style feature table: {features.shape[0]} agents x {features.shape[1] - 1} indicators.")

