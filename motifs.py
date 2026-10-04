"""
motifs.py

Extracts daily human mobility motifs, tour patterns, and behavioral regularity
from labeled stay points, grounding individual activity chains in statistical
physics and urban computing.

Academic References:
  - Schneider, C. M., Belik, V., Couronné, T., Smoreda, Z., & González, M. C. (2013).
    "Unravelling daily human mobility motifs."
    Journal of The Royal Society Interface, 10(84), 20130246.
    https://doi.org/10.1098/rsif.2013.0246
  - Jiang, S., Yang, Y., Gupta, S., Veneziano, D., Athavale, S., & González, M. C. (2016).
    "The TimeGeo modeling framework for urban mobility without travel surveys."
    Proceedings of the National Academy of Sciences (PNAS), 113(37), E5370–E5378.
    https://doi.org/10.1073/pnas.1524261113
"""

import os
import argparse
import numpy as np
import pandas as pd


def _classify_daily_sequence(place_labels):
    """
    Classifies a sequence of visited place labels in a single 24h day into
    one of the standard daily mobility motifs (Schneider et al., 2013):
      - 0: Stationary at Home (H-H)
      - 1: Simple Commute (H-W-H)
      - 2: Commute with intermediate stop (H-W-O-H or H-O-W-H)
      - 3: Simple Non-work Outing (H-O-H)
      - 4: Multi-stop complex tour (>= 3 distinct destinations)
    """
    if len(place_labels) == 0:
        return 0

    # Collapse consecutive identical places
    collapsed = [place_labels[0]]
    for lbl in place_labels[1:]:
        if lbl != collapsed[-1]:
            collapsed.append(lbl)

    has_work = "work" in collapsed
    has_other = "other" in collapsed
    has_home = "home" in collapsed

    if not has_work and not has_other:
        return 0  # Stationary at home
    elif has_work and not has_other:
        return 1  # Simple commute
    elif has_work and has_other:
        return 2  # Commute + secondary stop
    elif not has_work and has_other:
        return 3  # Non-work outing
    else:
        return 4  # Complex tour


def extract_daily_mobility_motifs(labeled_stays_df, population_df):
    """
    Groups stay points per subscriber by calendar date, identifies daily
    motifs, and computes behavioral regularity and entropy.
    """
    id_to_phone = dict(zip(population_df["agent_id"], population_df["phone_number"]))
    all_phones = population_df["phone_number"].tolist()

    df = labeled_stays_df.copy()
    df["arr"] = pd.to_datetime(df["arrival_time"])
    df["date"] = df["arr"].dt.date

    # Group stays by agent and day
    agent_motifs = {aid: [] for aid in population_df["agent_id"]}
    agent_trip_counts = {aid: [] for aid in population_df["agent_id"]}

    for (aid, dt), grp in df.groupby(["agent_id", "date"]):
        if aid not in agent_motifs:
            continue
        grp_sorted = grp.sort_values("arr")
        labels = grp_sorted["place_label"].tolist()
        motif_id = _classify_daily_sequence(labels)
        agent_motifs[aid].append(motif_id)
        agent_trip_counts[aid].append(max(0, len(labels) - 1))

    rows = []
    for aid in population_df["agent_id"]:
        phone = id_to_phone[aid]
        motifs = agent_motifs.get(aid, [])
        trips = agent_trip_counts.get(aid, [0])

        if len(motifs) == 0:
            rows.append({
                "phone_number": phone,
                "motif__primary_id": 0,
                "motif__entropy": 0.0,
                "motif__fraction_commute_tours": 0.0,
                "motif__fraction_stay_home_days": 1.0,
                "motif__distinct_motifs_count": 0,
                "motif__mean_daily_trips": 0.0,
            })
            continue

        motifs_arr = np.array(motifs)
        unique_m, counts = np.unique(motifs_arr, return_counts=True)
        primary_id = int(unique_m[np.argmax(counts)])

        # Shannon entropy across daily motifs
        p = counts / counts.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))

        frac_commute = float(np.mean(np.isin(motifs_arr, [1, 2])))
        frac_home = float(np.mean(motifs_arr == 0))
        n_distinct = len(unique_m)
        mean_trips = float(np.mean(trips)) if trips else 0.0

        rows.append({
            "phone_number": phone,
            "motif__primary_id": primary_id,
            "motif__entropy": float(entropy),
            "motif__fraction_commute_tours": float(frac_commute),
            "motif__fraction_stay_home_days": float(frac_home),
            "motif__distinct_motifs_count": int(n_distinct),
            "motif__mean_daily_trips": float(mean_trips),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract daily human mobility motifs from stay points.")
    parser.add_argument("--stays_file", type=str, default="output/stay_points_labeled.csv")
    parser.add_argument("--pop_file", type=str, default="data/population.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.stays_file):
        raise FileNotFoundError(f"Stay points file {args.stays_file} not found. Run label_places.py first.")

    print(f"Loading {args.stays_file} and {args.pop_file}...")
    stays = pd.read_csv(args.stays_file)
    pop = pd.read_csv(args.pop_file)

    print("Extracting daily mobility motifs and tour regularity (Schneider et al., 2013; Jiang et al., 2016)...")
    motifs_df = extract_daily_mobility_motifs(stays, pop)

    out_file = os.path.join(args.out_dir, "mobility_motifs.csv")
    motifs_df.to_csv(out_file, index=False)
    print(f"Extracted {motifs_df.shape[1] - 1} motif indicators for {len(motifs_df)} anonymized subscribers -> {out_file}")


if __name__ == "__main__":
    main()
