"""
graph_features.py

Constructs three distinct user graphs from anonymized Call Detail Records and
trajectory telemetry, and extracts topological metrics and neighbor behavioral
feature aggregations:
  1. Social Interaction Graph (from communication_events.csv)
  2. Spatial Co-Location Graph (from stay_points.csv)
  3. Mobility Similarity Graph (k-NN on mobility and spatial anchor features)

Per graph, computes:
  - Structural metrics: degree, strength (weighted degree), clustering coefficient, PageRank
  - Neighbor feature aggregations: mean and standard deviation of core continuous
    behavioral indicators across peer neighborhoods (zero-leakage: no target labels).

Academic References:
  - Eagle, N., Macy, M., & Claxton, R. (2010).
    "Network diversity and economic development."
    Science, 328(5981), 1029–1031. https://doi.org/10.1126/science.1186605
  - Onnela, J.-P., Saramäki, J., Hyvönen, J., et al. (2007).
    "Structure and tie strengths in mobile communication networks."
    Proceedings of the National Academy of Sciences (PNAS), 104(18), 7332–7336.
    https://doi.org/10.1073/pnas.0610245104
  - Dong, Y., Yang, Y., Tang, J., Yang, Y., & Chawla, N. V. (2014).
    "Inferring user demographics and social strategies in mobile social networks."
    Proceedings of the 20th ACM SIGKDD (KDD '14), 771–780.
    https://doi.org/10.1145/2623330.2623703
  - Blumenstock, J., Cadamuro, G., & On, R. (2015).
    "Predicting poverty and wealth from mobile phone metadata."
    Science, 350(6264), 1073–1077. https://doi.org/10.1126/science.aac4420
"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Core continuous behavioral features chosen for neighbor aggregation
AGGREGATION_FEATURES = [
    "commute_distance_m",
    "entropy_of_places",
    "radius_of_gyration__allweek__allday",
    "number_of_interactions__allweek__allday__call",
    "amount_recharges__allweek__allday__mean",
]

MOBILITY_KNN_FEATURES = [
    "commute_distance_m",
    "number_of_places",
    "entropy_of_places",
    "radius_of_gyration__allweek__allday",
]


def _build_feature_lookup(features_df, agg_cols=None):
    """Creates a fast dictionary lookup: phone_number -> {feature: value}."""
    if agg_cols is None:
        agg_cols = AGGREGATION_FEATURES
    available = [c for c in agg_cols if c in features_df.columns]
    if not available:
        return {}
    subset = features_df[["phone_number"] + available].fillna(0.0)
    return subset.set_index("phone_number").to_dict(orient="index")


def extract_social_graph_features(comm_events_df, population_df, feat_lookup=None):
    """
    Constructs a weighted directed communication network among subscribers and
    computes network centrality, topology, and neighbor behavioral feature aggregates.
    """
    id_to_phone = dict(zip(population_df["agent_id"], population_df["phone_number"]))
    all_phones = population_df["phone_number"].tolist()

    G = nx.DiGraph()
    G.add_nodes_from(all_phones)

    edge_weights = {}
    for row in comm_events_df.itertuples():
        c1 = id_to_phone.get(row.caller_id, str(row.caller_id))
        c2 = id_to_phone.get(row.callee_id, str(row.callee_id))
        if c1 in G and c2 in G:
            edge_weights[(c1, c2)] = edge_weights.get((c1, c2), 0) + 1

    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    try:
        pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
    except Exception:
        pagerank = {p: 1.0 / len(all_phones) for p in all_phones}

    G_undir = G.to_undirected()
    clustering = nx.clustering(G_undir)

    rows = []
    for p in all_phones:
        in_d = G.in_degree(p)
        out_d = G.out_degree(p)
        total_d = in_d + out_d

        in_str = sum(d["weight"] for _, _, d in G.in_edges(p, data=True))
        out_str = sum(d["weight"] for _, _, d in G.out_edges(p, data=True))
        total_str = in_str + out_str

        succ = set(G.successors(p))
        pred = set(G.predecessors(p))
        mutual = len(succ.intersection(pred))
        reciprocity = (2.0 * mutual / (len(succ) + len(pred))) if (len(succ) + len(pred)) > 0 else 0.0

        nbrs = list(succ.union(pred) - {p})

        row = {
            "phone_number": p,
            "graph_social__degree": total_d,
            "graph_social__in_degree": in_d,
            "graph_social__out_degree": out_d,
            "graph_social__strength": float(total_str),
            "graph_social__in_strength": float(in_str),
            "graph_social__out_strength": float(out_str),
            "graph_social__clustering": float(clustering.get(p, 0.0)),
            "graph_social__pagerank": float(pagerank.get(p, 0.0)),
            "graph_social__reciprocity": float(reciprocity),
        }

        if feat_lookup:
            for col in AGGREGATION_FEATURES:
                clean_name = col.split("__")[0]
                if nbrs:
                    vals = [feat_lookup[n][col] for n in nbrs if n in feat_lookup and col in feat_lookup[n]]
                    row[f"graph_social__neighbor_{clean_name}_mean"] = float(np.mean(vals)) if vals else 0.0
                    row[f"graph_social__neighbor_{clean_name}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
                else:
                    row[f"graph_social__neighbor_{clean_name}_mean"] = 0.0
                    row[f"graph_social__neighbor_{clean_name}_std"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def extract_colocation_graph_features(stay_points_df, population_df, feat_lookup=None,
                                      dist_tol_m=300.0, time_window_hours=1):
    """
    Constructs a spatial co-presence network connecting subscribers who dwell
    at the same location within temporal proximity.
    """
    id_to_phone = dict(zip(population_df["agent_id"], population_df["phone_number"]))
    all_phones = population_df["phone_number"].tolist()

    if len(stay_points_df) == 0:
        base_cols = ["phone_number", "graph_colocation__degree", "graph_colocation__strength", "graph_colocation__clustering"]
        return pd.DataFrame({c: (all_phones if c == "phone_number" else 0.0) for c in base_cols})

    df = stay_points_df.copy()
    df["arr"] = pd.to_datetime(df["arrival_time"])

    m_per_deg = 111320.0
    df["grid_x"] = (df["lon"] * m_per_deg // dist_tol_m).astype(int)
    df["grid_y"] = (df["lat"] * m_per_deg // dist_tol_m).astype(int)
    freq_str = f"{time_window_hours}h"
    df["time_block"] = df["arr"].dt.floor(freq_str)

    encounters = {}
    for _, grp in df.groupby(["grid_x", "grid_y", "time_block"]):
        agents = grp["agent_id"].unique()
        if len(agents) > 1:
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    p1 = id_to_phone.get(agents[i])
                    p2 = id_to_phone.get(agents[j])
                    if p1 and p2 and p1 != p2:
                        pair = (min(p1, p2), max(p1, p2))
                        encounters[pair] = encounters.get(pair, 0) + 1

    G = nx.Graph()
    G.add_nodes_from(all_phones)
    for (p1, p2), count in encounters.items():
        G.add_edge(p1, p2, weight=count)

    clustering = nx.clustering(G)

    rows = []
    for p in all_phones:
        nbrs = list(G.neighbors(p))
        d = len(nbrs)
        strength = sum(G[p][n]["weight"] for n in nbrs)

        row = {
            "phone_number": p,
            "graph_colocation__degree": d,
            "graph_colocation__strength": float(strength),
            "graph_colocation__clustering": float(clustering.get(p, 0.0)),
        }

        if feat_lookup:
            for col in AGGREGATION_FEATURES:
                clean_name = col.split("__")[0]
                if nbrs:
                    vals = [feat_lookup[n][col] for n in nbrs if n in feat_lookup and col in feat_lookup[n]]
                    row[f"graph_colocation__neighbor_{clean_name}_mean"] = float(np.mean(vals)) if vals else 0.0
                    row[f"graph_colocation__neighbor_{clean_name}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
                else:
                    row[f"graph_colocation__neighbor_{clean_name}_mean"] = 0.0
                    row[f"graph_colocation__neighbor_{clean_name}_std"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


class MobilityKNNGraphBuilder:
    """
    Constructs a k-NN mobility similarity graph with strict fold isolation.
    Fits solely on training set nodes during cross-validation, and transforms
    both training and validation nodes using only training neighbors.
    """

    def __init__(self, k=5, knn_feature_cols=None, agg_features=None):
        self.k = k
        self.knn_feature_cols = knn_feature_cols or MOBILITY_KNN_FEATURES
        self.agg_features = agg_features or AGGREGATION_FEATURES
        self.scaler = StandardScaler()
        self.knn_model = None
        self.train_phones = None
        self.train_feat_lookup = None
        self.train_graph = None

    def fit(self, train_features_df):
        """Fits k-NN index exclusively on the training feature matrix."""
        self.train_phones = train_features_df["phone_number"].to_numpy()
        self.train_feat_lookup = _build_feature_lookup(train_features_df, self.agg_features)

        available_cols = [c for c in self.knn_feature_cols if c in train_features_df.columns]
        X = train_features_df[available_cols].fillna(0.0).to_numpy()
        X_scaled = self.scaler.fit_transform(X)

        n_neighbors = min(self.k + 1, len(X_scaled))
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.knn_model.fit(X_scaled)

        # Build induced training graph for clustering coefficient computation
        dists, indices = self.knn_model.kneighbors(X_scaled)
        self.train_graph = nx.Graph()
        self.train_graph.add_nodes_from(range(len(self.train_phones)))
        for i in range(len(self.train_phones)):
            # indices[i, 0] is self; connect to k neighbors
            for j in indices[i, 1:n_neighbors]:
                self.train_graph.add_edge(i, j)

        return self

    def transform(self, query_features_df, is_train=False):
        """
        Transforms query nodes (training or testing) by projecting onto the
        training k-NN graph. Never leaks query relations into training.
        """
        available_cols = [c for c in self.knn_feature_cols if c in query_features_df.columns]
        X = query_features_df[available_cols].fillna(0.0).to_numpy()
        X_scaled = self.scaler.transform(X)

        if is_train:
            n_query = min(self.k + 1, len(self.train_phones))
            dists, indices = self.knn_model.kneighbors(X_scaled, n_neighbors=n_query)
            # Exclude self
            dists = dists[:, 1:]
            indices = indices[:, 1:]
        else:
            n_query = min(self.k, len(self.train_phones))
            dists, indices = self.knn_model.kneighbors(X_scaled, n_neighbors=n_query)

        train_clustering = nx.clustering(self.train_graph)

        rows = []
        query_phones = query_features_df["phone_number"].to_numpy()
        for idx, phone in enumerate(query_phones):
            nbr_indices = indices[idx]
            nbr_dists = dists[idx]

            # Similarity weights: inverse distance
            weights = 1.0 / (1.0 + nbr_dists)
            strength = float(np.sum(weights))

            # Average clustering coefficient among connected training neighbors
            nbr_clustering = [train_clustering.get(ni, 0.0) for ni in nbr_indices]
            mean_clustering = float(np.mean(nbr_clustering)) if nbr_clustering else 0.0

            nbr_phones = [self.train_phones[ni] for ni in nbr_indices]

            row = {
                "phone_number": phone,
                "graph_mobility__degree": len(nbr_indices),
                "graph_mobility__strength": strength,
                "graph_mobility__clustering": mean_clustering,
            }

            for col in self.agg_features:
                clean_name = col.split("__")[0]
                vals = [self.train_feat_lookup[np_phone][col] for np_phone in nbr_phones
                        if np_phone in self.train_feat_lookup and col in self.train_feat_lookup[np_phone]]
                row[f"graph_mobility__neighbor_{clean_name}_mean"] = float(np.mean(vals)) if vals else 0.0
                row[f"graph_mobility__neighbor_{clean_name}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

            rows.append(row)

        return pd.DataFrame(rows)


def extract_all_graph_features(comm_events_df, stay_points_df, population_df,
                               bandicoot_features_df, anchors_df=None):
    """
    Computes features across all three graphs for the full dataset.
    """
    feat_lookup = _build_feature_lookup(bandicoot_features_df)

    # 1. Social interaction graph
    social_df = extract_social_graph_features(comm_events_df, population_df, feat_lookup=feat_lookup)

    # 2. Co-location graph
    colo_df = extract_colocation_graph_features(stay_points_df, population_df, feat_lookup=feat_lookup)

    # 3. Mobility k-NN graph (full dataset fit and transform)
    mob_builder = MobilityKNNGraphBuilder(k=5)
    mob_builder.fit(bandicoot_features_df)
    mob_df = mob_builder.transform(bandicoot_features_df, is_train=True)

    # Merge all
    merged = social_df.merge(colo_df, on="phone_number").merge(mob_df, on="phone_number")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Extract three user graphs and neighbor feature aggregations.")
    parser.add_argument("--comm_file", type=str, default="data/communication_events.csv")
    parser.add_argument("--pop_file", type=str, default="data/population.csv")
    parser.add_argument("--stays_file", type=str, default="output/stay_points.csv")
    parser.add_argument("--bandicoot_file", type=str, default="output/bandicoot_features.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data for graph construction...")
    pop = pd.read_csv(args.pop_file)
    comms = pd.read_csv(args.comm_file, parse_dates=["timestamp"])
    stays = pd.read_csv(args.stays_file)
    bf = pd.read_csv(args.bandicoot_file)

    print("Extracting features across all three user graphs...")
    graph_df = extract_all_graph_features(comms, stays, pop, bf)

    out_path = os.path.join(args.out_dir, "graph_features.csv")
    graph_df.to_csv(out_path, index=False)
    print(f"Extracted {graph_df.shape[1] - 1} graph features for {len(graph_df)} subscribers -> {out_path}")


if __name__ == "__main__":
    main()
