# src/graph/visualize_er_graph.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ------------- Build ---------------------------------------------------------
def build_match_graph(
    df_pred: pd.DataFrame,
    prob_col: str = "prob_match",
    u_col: str = "src_id",
    v_col: str = "cand_id",
    keep_threshold: float = 0.7,
) -> nx.Graph:
    """
    Turn classifier predictions into an undirected weighted graph.
    Keeps only edges with prob >= keep_threshold.
    Edge attribute: weight=probability
    """
    G = nx.Graph()
    # add edges
    for _, r in df_pred.iterrows():
        p = float(r[prob_col])
        if p >= keep_threshold:
            u, v = int(r[u_col]), int(r[v_col])
            if u == v:
                continue
            # keep the max weight if multiple edges appear
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], p)
            else:
                G.add_edge(u, v, weight=p)
    return G

# ------------- Sampling (optional, for readability) -------------------------
def sample_subgraph(
    G: nx.Graph,
    max_nodes: int = 400,
    seed: int = 42,
) -> nx.Graph:
    """
    If G is huge, sample a representative subgraph:
    - take the giant connected component
    - pick a random BFS ego forest up to max_nodes
    """
    if G.number_of_nodes() <= max_nodes:
        return G.copy()

    # start from highest-degree nodes to keep dense regions
    nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    picked: List[int] = []
    rng = np.random.default_rng(seed)

    for start, _ in nodes_sorted:
        if len(picked) >= max_nodes:
            break
        # BFS frontier of up to ~max_nodes/10 per seed
        quota = max(20, max_nodes // 10)
        layer = list(nx.bfs_tree(G, start, depth_limit=2).nodes())
        rng.shuffle(layer)
        for n in layer:
            if n not in picked:
                picked.append(n)
            if len(picked) >= max_nodes:
                break

    return G.subgraph(picked).copy()

# ------------- Clustering / coloring ----------------------------------------
def communities_louvain_or_cc(G: nx.Graph, use_louvain: bool = True) -> Dict[int, int]:
    """
    Returns {node: community_id}
    - If NetworkX has Louvain (>=3.0) use it; else fallback to connected components.
    """
    node2comm: Dict[int, int] = {}
    if use_louvain and hasattr(nx.algorithms.community, "louvain_communities"):
        comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
        for cid, nodes in enumerate(comms):
            for n in nodes:
                node2comm[n] = cid
    else:
        for cid, comp in enumerate(nx.connected_components(G)):
            for n in comp:
                node2comm[n] = cid
    return node2comm

# ------------- Visualization -------------------------------------------------
def visualize_graph(
    G: nx.Graph,
    node2comm: Optional[Dict[int, int]] = None,
    title: str = "ER Match Graph",
    with_labels: bool = False,
    figsize: Tuple[int, int] = (12, 9),
    out_path: Optional[Path] = None,
) -> None:
    """
    Draws a spring-layout graph:
    - node color: community
    - node size: degree
    - edge width: weight
    """
    if G.number_of_nodes() == 0:
        print("[viz] Empty graph.")
        return

    pos = nx.spring_layout(G, weight="weight", seed=42, k=None)
    degrees = dict(G.degree())
    node_sizes = np.array([degrees[n] for n in G.nodes()], dtype=float)
    node_sizes = 200 * (1 + np.log1p(node_sizes))  # compress range

    if node2comm is None:
        node2comm = {n: 0 for n in G.nodes()}
    comm_ids = np.array([node2comm[n] for n in G.nodes()])
    # map communities to 0..C-1
    _, inv = np.unique(comm_ids, return_inverse=True)
    colors = inv

    edge_w = [G[u][v]["weight"] for u, v in G.edges()]
    # normalize widths for visibility
    ew = 0.5 + 3.0 * (np.array(edge_w) - min(edge_w)) / (max(edge_w) - min(edge_w) + 1e-9)

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, width=ew, alpha=0.35)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        cmap="tab20",
        node_size=node_sizes,
        linewidths=0.5,
        edgecolors="k"
    )
    if with_labels and G.number_of_nodes() <= 200:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"[viz] Saved figure to {out_path}")
    plt.show()

# ------------- Exports for external tools -----------------------------------
def export_for_gephi(G: nx.Graph, path: Path) -> None:
    """
    Export to GEXF so you can play with layout/filters in Gephi.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, path)
    print(f"[viz] GEXF written -> {path.resolve()}")
