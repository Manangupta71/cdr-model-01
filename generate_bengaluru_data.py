"""
generate_bengaluru_data.py

Synthetic, city-agnostic CDR (Call Detail Record) data generator, instantiated
here with Bengaluru-shaped parameters (real named localities, illustrative
tower layout, IT-corridor employment geography). Produces the raw record
streams a real mobile network operator export would contain, which
bandicoot_features.py then turns into the standard behavioral-indicator
feature table:

  1. A synthetic population with socio-demographic attributes drawn from
     PROBABILISTIC JOINT distributions (age, gender, income bracket,
     education level, occupation category, work status, socioeconomic
     class), plus a home/work locality assignment.
  2. A cell-tower ("antenna") layout across the city, used to discretize
     raw GPS-like pings into the antenna IDs a real CDR would report.
  3. A mobility trace (antenna pings over time) per agent, generated from
     a daily activity schedule.
  4. A communication event stream (calls/SMS) with real social structure
     (household, coworker, weak-tie edges).
  5. A prepaid airtime recharge (top-up) event stream per agent. Recharge
     frequency and amount are modeled to correlate with income bracket,
     following the direction reported in Steele et al. (2017, J. R. Soc.
     Interface): lower-income users top up more frequently in smaller
     amounts, rather than less often in larger ones.

NOTE ON DATA LINEAGE: see docs/DATA_LINEAGE.md for the grounded /
plausible / artifact classification of every distributional choice below.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.spatial import cKDTree

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_AGENTS = 5000
SIM_DAYS = 14
CITY_BOUNDS = {"lat_min": 12.83, "lat_max": 13.14, "lon_min": 77.46, "lon_max": 77.78}
N_ANTENNAS = 250  # illustrative tower density across the metro area

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-59", "60+"]
GENDERS = ["male", "female"]
INCOME_BRACKETS = ["low", "lower_mid", "upper_mid", "high"]
EDUCATION_LEVELS = ["primary", "secondary", "graduate", "postgraduate"]
OCCUPATION_CATEGORIES = ["informal_labor", "retail_service", "clerical", "professional", "unemployed", "student"]
WORK_STATUS = ["employed", "unemployed", "student", "retired"]
SOCIOECONOMIC_CLASS = ["E", "D", "C", "B", "A"]  # NCCS-style bands, low to high


@dataclass
class Zone:
    name: str
    lat: float
    lon: float
    density_weight: float  # relative residential density
    employment_weight: float  # relative jobs density


# Real Bengaluru localities with illustrative (not survey-sourced) density
# weights. IT-corridor areas (Whitefield, Electronic City, Marathahalli,
# ORR/Bellandur) get high employment_weight; established residential areas
# (Jayanagar, Banashankari, Rajajinagar, Yelahanka) get high density_weight.
ZONES = [
    Zone("MG Road", 12.9757, 77.6086, 0.03, 0.11),
    Zone("Koramangala", 12.9352, 77.6245, 0.09, 0.10),
    Zone("Indiranagar", 12.9784, 77.6408, 0.08, 0.08),
    Zone("Whitefield", 12.9698, 77.7500, 0.10, 0.16),
    Zone("Electronic City", 12.8452, 77.6602, 0.07, 0.15),
    Zone("Jayanagar", 12.9308, 77.5838, 0.11, 0.05),
    Zone("HSR Layout", 12.9116, 77.6389, 0.09, 0.09),
    Zone("Marathahalli", 12.9569, 77.6974, 0.09, 0.12),
    Zone("Rajajinagar", 12.9911, 77.5555, 0.09, 0.04),
    Zone("Yelahanka", 13.1007, 77.5963, 0.08, 0.03),
    Zone("BTM Layout", 12.9166, 77.6101, 0.09, 0.04),
    Zone("Banashankari", 12.9250, 77.5468, 0.08, 0.03),
]


# ---------------------------------------------------------------------------
# Step 0: Antenna (cell tower) layout
# ---------------------------------------------------------------------------

def generate_antenna_layout(n_antennas=N_ANTENNAS):
    """
    Scatters antennas across the city with density proportional to the sum
    of residential + employment weight at nearby zones (denser towers in
    busier areas, matching real urban cell-tower deployment patterns).
    Returns a DataFrame [antenna_id, lat, lon].
    """
    zone_weights = np.array([z.density_weight + z.employment_weight for z in ZONES])
    zone_weights /= zone_weights.sum()
    chosen_zones = rng.choice(len(ZONES), size=n_antennas, p=zone_weights)
    rows = []
    for antenna_id, zi in enumerate(chosen_zones):
        z = ZONES[zi]
        rows.append({
            "antenna_id": antenna_id,
            "lat": z.lat + rng.normal(0, 0.018),
            "lon": z.lon + rng.normal(0, 0.018),
        })
    return pd.DataFrame(rows)


def assign_antennas(points_df, antennas_df):
    """Nearest-antenna assignment via a KD-tree (planar approx, fine at city scale)."""
    tree = cKDTree(antennas_df[["lat", "lon"]].to_numpy())
    _, idx = tree.query(points_df[["lat", "lon"]].to_numpy())
    return antennas_df["antenna_id"].to_numpy()[idx]


# ---------------------------------------------------------------------------
# Step 1: Joint demographic sampling (unchanged logic, Bengaluru geography)
# ---------------------------------------------------------------------------

def _sample_age_group():
    p = [0.24, 0.30, 0.21, 0.16, 0.09]  # PLAUSIBLE: skews slightly younger (IT-migration city)
    return rng.choice(AGE_GROUPS, p=p)


def _sample_education(age_group):
    if age_group in ("18-24",):
        p = [0.08, 0.30, 0.45, 0.17]
    elif age_group in ("25-34", "35-44"):
        p = [0.10, 0.25, 0.40, 0.25]
    else:
        p = [0.25, 0.35, 0.28, 0.12]
    return rng.choice(EDUCATION_LEVELS, p=p)


def _sample_occupation(age_group, education):
    if age_group == "18-24" and education in ("secondary", "graduate") and rng.random() < 0.35:
        return "student"
    base = {
        "primary": [0.50, 0.32, 0.06, 0.03, 0.07, 0.02],
        "secondary": [0.25, 0.35, 0.18, 0.12, 0.06, 0.04],
        "graduate": [0.06, 0.16, 0.23, 0.48, 0.03, 0.04],
        "postgraduate": [0.02, 0.06, 0.15, 0.72, 0.02, 0.03],
    }[education]
    p = np.array(base, dtype=float)
    p /= p.sum()
    return rng.choice(OCCUPATION_CATEGORIES, p=p)


def _work_status_from_occupation(occupation, age_group):
    # ARTIFACT: see docs/DATA_LINEAGE.md — work_status is near-deterministic
    # given occupation in this generator and should not be read as a
    # genuine test of predictive difficulty.
    if age_group == "60+" and rng.random() < 0.6:
        return "retired"
    if occupation == "student":
        return "student"
    if occupation == "unemployed":
        return "unemployed"
    return "employed"


def _sample_income(occupation, education):
    base = {
        "informal_labor": [0.50, 0.38, 0.10, 0.02],
        "retail_service": [0.25, 0.45, 0.24, 0.06],
        "clerical": [0.08, 0.32, 0.42, 0.18],
        "professional": [0.02, 0.10, 0.38, 0.50],
        "unemployed": [0.70, 0.25, 0.04, 0.01],
        "student": [0.55, 0.32, 0.10, 0.03],
    }[occupation]
    p = np.array(base, dtype=float)
    if education in ("graduate", "postgraduate"):
        p = p * np.array([0.6, 0.9, 1.15, 1.5])
    p /= p.sum()
    return rng.choice(INCOME_BRACKETS, p=p)


def _sample_socioeconomic_class(income_bracket, education):
    # GROUNDED (direction, post-fix): derived only from income + education
    # directly, matching NCCS-style construction — not from any behavioral
    # propensity score (see docs/DATA_LINEAGE.md leakage-fix note).
    income_score = {"low": 0, "lower_mid": 1, "upper_mid": 2, "high": 3}[income_bracket]
    edu_score = {"primary": 0, "secondary": 1, "graduate": 2, "postgraduate": 3}[education]
    combined = income_score + edu_score + rng.normal(0, 0.6)
    idx = int(np.clip(round(combined / 6 * 4), 0, 4))
    return SOCIOECONOMIC_CLASS[idx]


def _normalized(weights):
    w = np.array(weights, dtype=float)
    return w / w.sum()


def generate_population(n_agents=None):
    if n_agents is None:
        n_agents = N_AGENTS
    records = []
    for agent_id in range(n_agents):
        age_group = _sample_age_group()
        gender = rng.choice(GENDERS, p=[0.52, 0.48])
        education = _sample_education(age_group)
        occupation = _sample_occupation(age_group, education)
        work_status = _work_status_from_occupation(occupation, age_group)
        income_bracket = _sample_income(occupation, education)
        socio_class = _sample_socioeconomic_class(income_bracket, education)

        home_zone = rng.choice(ZONES, p=_normalized([z.density_weight for z in ZONES]))
        if work_status == "employed":
            work_zone = rng.choice(ZONES, p=_normalized([z.employment_weight for z in ZONES]))
        else:
            work_zone = home_zone

        records.append({
            "agent_id": agent_id,
            "phone_number": f"BLR{100000000 + agent_id}",
            "age_group": age_group,
            "gender": gender,
            "education_level": education,
            "occupation_category": occupation,
            "work_status": work_status,
            "income_bracket": income_bracket,
            "socio_demographic_class": socio_class,
            "home_zone": home_zone.name,
            "home_lat": home_zone.lat + rng.normal(0, 0.01),
            "home_lon": home_zone.lon + rng.normal(0, 0.01),
            "work_zone": work_zone.name,
            "work_lat": work_zone.lat + rng.normal(0, 0.01),
            "work_lon": work_zone.lon + rng.normal(0, 0.01),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 2: Mobility trace generation (tower pings)
# ---------------------------------------------------------------------------

def _daily_schedule(agent, day_offset, is_weekend):
    pings = []
    base_date = datetime(2026, 1, 5) + timedelta(days=day_offset)

    def ping(hour, minute, lat, lon, jitter=0.0008):
        ts = base_date + timedelta(hours=hour, minutes=minute)
        pings.append((ts, lat + rng.normal(0, jitter), lon + rng.normal(0, jitter)))

    for h in (0, 2, 4, 6):
        ping(h, int(rng.integers(0, 59)), agent.home_lat, agent.home_lon)

    if not is_weekend and agent.work_status in ("employed", "student"):
        commute_hour = int(rng.integers(7, 10))
        ping(commute_hour, int(rng.integers(0, 59)),
             (agent.home_lat + agent.work_lat) / 2, (agent.home_lon + agent.work_lon) / 2, jitter=0.01)
        for h in range(commute_hour + 1, 18):
            if rng.random() < 0.7:
                ping(h, int(rng.integers(0, 59)), agent.work_lat, agent.work_lon)
        return_hour = int(rng.integers(18, 21))
        ping(return_hour, int(rng.integers(0, 59)),
             (agent.home_lat + agent.work_lat) / 2, (agent.home_lon + agent.work_lon) / 2, jitter=0.01)
        for h in range(return_hour + 1, 24):
            ping(h, int(rng.integers(0, 59)), agent.home_lat, agent.home_lon)
    else:
        for h in range(7, 24, 2):
            if rng.random() < 0.35:
                leisure = rng.choice(ZONES)
                ping(h, int(rng.integers(0, 59)), leisure.lat, leisure.lon, jitter=0.008)
            else:
                ping(h, int(rng.integers(0, 59)), agent.home_lat, agent.home_lon)

    return pings


def generate_mobility_traces(population_df, sim_days=None):
    if sim_days is None:
        sim_days = SIM_DAYS
    rows = []
    for agent in population_df.itertuples():
        for day in range(sim_days):
            is_weekend = (day % 7) in (5, 6)
            for ts, lat, lon in _daily_schedule(agent, day, is_weekend):
                rows.append({"agent_id": agent.agent_id, "timestamp": ts, "lat": lat, "lon": lon})
    df = pd.DataFrame(rows).sort_values(["agent_id", "timestamp"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 3: Communication event stream
# ---------------------------------------------------------------------------

def generate_communication_events(population_df, sim_days=None, avg_household_size=3.2):
    if sim_days is None:
        sim_days = SIM_DAYS
    events = []
    by_home = population_df.groupby("home_zone")["agent_id"].apply(list).to_dict()
    by_work = population_df.groupby("work_zone")["agent_id"].apply(list).to_dict()
    n = len(population_df)
    agent_ids = population_df["agent_id"].to_numpy()

    edges = set()
    for zone, ids in by_home.items():
        ids = list(ids)
        rng.shuffle(ids)
        group_size = max(1, int(avg_household_size))
        for i in range(0, len(ids), group_size):
            group = ids[i:i + group_size]
            for a in group:
                for b in group:
                    if a != b:
                        edges.add((min(a, b), max(a, b)))

    for zone, ids in by_work.items():
        ids = list(ids)
        if len(ids) < 2:
            continue
        n_edges = int(len(ids) * 1.5)
        for _ in range(n_edges):
            a, b = rng.choice(ids, size=2, replace=False)
            edges.add((min(a, b), max(a, b)))

    n_weak = int(n * 0.8)
    for _ in range(n_weak):
        a, b = rng.choice(agent_ids, size=2, replace=False)
        edges.add((min(a, b), max(a, b)))

    for (a, b) in edges:
        n_events = rng.poisson(lam=2.0 * sim_days / 7) + 1
        for _ in range(n_events):
            day = int(rng.integers(0, sim_days))
            hour = int(np.clip(rng.normal(14, 5), 0, 23))
            minute = int(rng.integers(0, 59))
            ts = datetime(2026, 1, 5) + timedelta(days=day, hours=hour, minutes=minute)
            direction = rng.choice([1, -1])
            caller, callee = (a, b) if direction == 1 else (b, a)
            event_type = rng.choice(["call", "text"], p=[0.6, 0.4])
            duration = int(rng.exponential(120)) if event_type == "call" else 0
            events.append({
                "caller_id": caller, "callee_id": callee, "timestamp": ts,
                "event_type": event_type, "duration_sec": duration,
            })

    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Prepaid airtime recharge events
# ---------------------------------------------------------------------------

# GROUNDED (direction): Steele et al. (2017), J. R. Soc. Interface —
# lower-income users top up more frequently in SMALLER amounts; this is
# encoded directly below rather than left to chance.
RECHARGE_PROFILE = {
    "low":       {"mean_days_between": 3.0,  "amount_mean": 40,  "amount_sd": 15},
    "lower_mid": {"mean_days_between": 5.0,  "amount_mean": 80,  "amount_sd": 25},
    "upper_mid": {"mean_days_between": 9.0,  "amount_mean": 180, "amount_sd": 60},
    "high":      {"mean_days_between": 16.0, "amount_mean": 400, "amount_sd": 150},
}


def generate_recharge_events(population_df, sim_days=None):
    if sim_days is None:
        sim_days = SIM_DAYS
    rows = []
    start = datetime(2026, 1, 5)
    for agent in population_df.itertuples():
        profile = RECHARGE_PROFILE[agent.income_bracket]
        t = rng.exponential(profile["mean_days_between"])
        balance = 0.0
        while t < sim_days:
            amount = max(10.0, rng.normal(profile["amount_mean"], profile["amount_sd"]))
            balance += amount
            # balance depletes at a rate roughly tied to daily usage; simple decay model
            balance = max(0.0, balance - rng.uniform(0.3, 0.6) * amount)
            ts = start + timedelta(days=t)
            rows.append({
                "agent_id": agent.agent_id, "timestamp": ts,
                "amount": round(amount, 2), "balance_after": round(balance, 2),
            })
            t += rng.exponential(profile["mean_days_between"])
    return pd.DataFrame(rows).sort_values(["agent_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating antenna layout (n={N_ANTENNAS})...")
    antennas = generate_antenna_layout()
    antennas.to_csv(f"{out_dir}/antennas.csv", index=False)

    print(f"Generating population (n={N_AGENTS})...")
    population = generate_population()
    population.to_csv(f"{out_dir}/population.csv", index=False)

    print(f"Generating mobility traces ({SIM_DAYS} days)...")
    mobility = generate_mobility_traces(population)
    mobility["antenna_id"] = assign_antennas(mobility, antennas)
    mobility.to_csv(f"{out_dir}/mobility_traces.csv", index=False)

    print("Generating communication event stream...")
    comms = generate_communication_events(population)
    comms.to_csv(f"{out_dir}/communication_events.csv", index=False)

    print("Generating recharge event stream...")
    recharges = generate_recharge_events(population)
    recharges.to_csv(f"{out_dir}/recharge_events.csv", index=False)

    print(f"Done. population={len(population)}, antennas={len(antennas)}, "
          f"pings={len(mobility)}, comm_events={len(comms)}, recharges={len(recharges)}")
    print("See docs/DATA_LINEAGE.md for the grounded/plausible/ungrounded audit.")


if __name__ == "__main__":
    main()
