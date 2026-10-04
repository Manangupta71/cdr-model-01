"""
graph_features.py

Extracts topological and relational graph features from anonymized Call Detail
Records (communication logs and co-location encounters), capturing social
capital, tie strength, and co-presence dynamics. Operates purely on
pseudonymous/anonymized subscriber identifiers.

Academic References:
  - Eagle, N., Macy, M., & Claxton, R. (2010).
    "Network diversity and economic development."
    Science, 328(5981), 1029–1031.
    https://doi.org/10.1126/science.1186605
  - Onnela, J.-P., Saramäki, J., Hyvönen, J., et al. (2007).
    "Structure and tie strengths in mobile communication networks."
    Proceedings of the National Academy of Sciences (PNAS), 104(18), 7332–7336.
    https://doi.org/10.1073/pnas.0610245104
  - Dong, Y., Yang, Y., Tang, J., Yang, Y., & Chawla, N. V. (2014).
    "Inferring user demographics and social strategies in mobile social networks."
    Proceedings of the 20th ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining (KDD '14), 771–780.
    https://doi.org/10.1145/2623330.2623703
"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx


def extract_communication_graph_features(comm_events_df, population_df):
    """
    Constructs a weighted directed communication network among anonymized
    subscribers and calculates centrality and topological metrics.
    """
    # Create mapping between agent_id and anonymized phone_number / subscriber_id
    id_to_phone = dict(zip(population_df["agent_id"], population_df["phone_number"]))
    all_phones = population_df["phone_number"].tolist()

    # Build directed graph
    G = nx.DiGraph()
    G.add_nodes_from(all_phones)

    # Aggregate call/SMS events between pairs
    edge_counts = {}
    for row in comm_events_df.itertuples():
        c1 = id_to_phone.get(row.caller_id, str(row.caller_id))
        c2 = id_to_phone.get(row.callee_id, str(row.callee_id))
        edge_counts[(c1, c2)] = edge_counts.get((c1, c2), 0) + 1

    for (u, v), w in edge_counts.items():
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, weight=w)

    # Compute network measures
    try:
        pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
    except Exception:
        pagerank = {n: 1.0 / len(all_phones) for n in all_phones}

    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))
    in_degree_unweighted = dict(G.in_degree())
    out_degree_unweighted = dict(G.out_degree())

    # Undirected projection for clustering coefficient and ego density
    G_undir = G.to_undirected()
    clustering = nx.clustering(G_undir)

    rows = []
    for phone in all_phones:
        in_d = in_degree.get(phone, 0.0)
        out_d = out_degree.get(phone, 0.0)
        total_d = in_d + out_d

        in_cnt = in_degree_unweighted.get(phone, 0)
        out_cnt = out_degree_unweighted.get(phone, 0)
        total_cnt = in_cnt + out_cnt

        # Reciprocity (fraction of mutual partners)
        succ = set(G.successors(phone)) if G.has_node(phone) else set()
        pred = set(G.predecessors(phone)) if G.has_node(phone) else set()
        mutual = len(succ.intersection(pred))
        reciprocity = (2.0 * mutual / (len(succ) + len(pred))) if (len(succ) + len(pred)) > 0 else 0.0

        # Degree ratio (incoming vs outgoing balance)
        degree_ratio = (in_d - out_d) / total_d if total_d > 0 else 0.0

        rows.append({
            "phone_number": phone,
            "graph__pagerank": float(pagerank.get(phone, 0.0)),
            "graph__in_degree": float(in_d),
            "graph__out_degree": float(out_d),
            "graph__degree_ratio": float(degree_ratio),
            "graph__in_contacts": int(in_cnt),
            "graph__out_contacts": int(out_cnt),
            "graph__total_contacts": int(total_cnt),
            "graph__clustering_coefficient": float(clustering.get(phone, 0.0)),
            "graph__reciprocity": float(reciprocity),
        })

    return pd.DataFrame(rows)


def extract_colocation_features(stay_points_df, population_df, time_window_sec=1800, dist_tol_m=300.0):
    """
    Constructs a spatial co-presence network connecting subscribers who dwell
    at the same location within temporal proximity.
    """
    id_to_phone = dict(zip(population_df["agent_id"], population_df["phone_number"]))
    all_phones = population_df["phone_number"].tolist()

    if len(stay_points_df) == 0:
        return pd.DataFrame({
            "phone_number": all_phones,
            "graph__colocation_degree": 0,
            "graph__colocation_encounters": 0,
        })

    from stay_points import haversine_m

    # Discretize stay intervals to find co-located pairs
    df = stay_points_df.copy()
    df["arr"] = pd.to_datetime(df["arrival_time"])
    df["dep"] = pd.to_datetime(df["departure_time"])

    # Spatial-temporal bins: 1-hour time blocks and ~300m spatial grid
    m_per_deg = 111320.0
    df["grid_x"] = (df["lon"] * m_per_deg // dist_tol_m).astype(int)
    df["grid_y"] = (df["lat"] * m_per_deg // dist_tol_m).astype(int)
    df["hour_block"] = df["arr"].dt.floor("1h")

    encounters = {}
    grouped = df.groupby(["grid_x", "grid_y", "hour_block"])
    for _, grp in grouped:
        agents = grp["agent_id"].unique()
        if len(agents) > 1:
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    a1 = id_to_phone.get(agents[i], str(agents[i]))
                    a2 = id_to_phone.get(agents[j], str(agents[j]))
                    pair = (min(a1, a2), max(a1, a2))
                    encounters[pair] = encounters.get(pair, 0) + 1

    # Aggregate per agent
    partner_sets = {p: set() for p in all_phones}
    encounter_counts = {p: 0 for p in all_phones}

    for (p1, p2), count in encounters.items():
        if p1 in partner_sets:
            partner_sets[p1].add(p2)
            encounter_counts[p1] += count
        if p2 in partner_sets:
            partner_sets[p2].add(p1)
            encounter_counts[p2] += count

    rows = []
    for p in all_phones:
        rows.append({
            "phone_number": p,
            "graph__colocation_degree": len(partner_sets[p]),
            "graph__colocation_encounters": encounter_counts[p],
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract graph-based relational features from CDR logs.")
    parser.add_argument("--comm_file", type=str, default="data/communication_events.csv")
    parser.add_argument("--pop_file", type=str, default="data/population.csv")
    parser.add_argument("--stays_file", type=str, default="output/stay_points_labeled.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data: {args.pop_file}, {args.comm_file}...")
    pop = pd.read_csv(args.pop_file)
    comms = pd.read_csv(args.comm_file, parse_dates=["timestamp"])

    print("Extracting social communication graph features (PageRank, degree, reciprocity)...")
    comm_feats = extract_communication_graph_features(comms, pop)

    stays_path = args.stays_file if os.path.exists(args.stays_file) else "output/stay_points.csv"
    if os.path.exists(stays_path):
        print(f"Extracting spatial co-location graph features from {stays_path}...")
        stays = pd.read_csv(stays_path)
        colo_feats = extract_colocation_features(stays, pop)
        merged = comm_feats.merge(colo_feats, on="phone_number")
    else:
        merged = comm_feats

    out_file = os.path.join(args.out_dir, "graph_features.csv")
    merged.to_csv(out_file, index=False)
    print(f"Extracted {merged.shape[1] - 1} graph features for {len(merged)} anonymized agents -> {out_file}")


if __name__ == "__main__":
    main()
