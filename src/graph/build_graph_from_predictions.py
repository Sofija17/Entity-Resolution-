from __future__ import annotations
import pandas as pd
import networkx as nx

def build_graph_from_predictions(
    df_pred: pd.DataFrame,
    prob_col: str = "prob_match",
    u_col: str = "src_id",
    v_col: str = "cand_id",
    keep_threshold: float = 0.45,
) -> nx.Graph:
    """Undirected weighted graph from classifier predictions."""
    for c in (prob_col, u_col, v_col):
        if c not in df_pred.columns:
            raise ValueError(
                f"build_graph_from_predictions expected '{c}'. "
                f"Got: {list(df_pred.columns)}"
            )

    G = nx.Graph()
    for _, r in df_pred.iterrows():
        p = float(r[prob_col])
        if p < keep_threshold:
            continue
        u, v = int(r[u_col]), int(r[v_col])
        if u == v:
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] = max(G[u][v]["weight"], p)
        else:
            G.add_edge(u, v, weight=p)
    return G
