from __future__ import annotations
import pandas as pd
import networkx as nx

def build_graph_from_transitivity(
    df_trans: pd.DataFrame,
    strategy: str = "chain",
) -> nx.Graph:
    """
    Graph from transitivity clusters (node_id, cluster_id, cluster_size).
    strategy: "chain" | "star"
    """
    if strategy not in {"chain", "star"}:
        raise ValueError("strategy must be 'chain' or 'star'")

    required = {"node_id", "cluster_id", "cluster_size"}
    if not required.issubset(df_trans.columns):
        raise ValueError(
            f"build_graph_from_transitivity requires {required}, got {set(df_trans.columns)}"
        )

    G = nx.Graph()
    # nodes with attrs
    for _, r in df_trans.iterrows():
        n = int(r["node_id"])
        G.add_node(n, cluster_id=int(r["cluster_id"]), cluster_size=int(r["cluster_size"]))

    # light wiring per cluster
    for _, sub in df_trans.groupby("cluster_id"):
        members = sorted(sub["node_id"].astype(int).tolist())
        if len(members) <= 1:
            continue
        if strategy == "chain":
            for a, b in zip(members[:-1], members[1:]):
                G.add_edge(a, b, weight=1.0)
        else:
            center = members[0]
            for m in members[1:]:
                G.add_edge(center, m, weight=1.0)

    return G
