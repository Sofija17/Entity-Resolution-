from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def sample_subgraph(G: nx.Graph, max_nodes: int = 400, seed: int = 42) -> nx.Graph:
    """Sample via BFS from high-degree nodes for readability."""
    if G.number_of_nodes() <= max_nodes:
        return G.copy()

    nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    picked: List[int] = []
    rng = np.random.default_rng(seed)

    for start, _ in nodes_sorted:
        if len(picked) >= max_nodes:
            break
        layer = list(nx.bfs_tree(G, start, depth_limit=2).nodes())
        rng.shuffle(layer)
        for n in layer:
            if n not in picked:
                picked.append(n)
            if len(picked) >= max_nodes:
                break
    return G.subgraph(picked).copy()

def communities_louvain_or_cc(G: nx.Graph, use_louvain: bool = True) -> Dict[int, int]:
    """Return {node: community_id} using Louvain if available, else CC."""
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

def _colors_from_mapping_or_cluster_attr(G: nx.Graph, node2comm: Optional[Dict[int, int]]) -> np.ndarray:
    if node2comm is None:
        vals = np.array([G.nodes[n].get("cluster_id", 0) for n in G.nodes()])
    else:
        vals = np.array([node2comm.get(n, 0) for n in G.nodes()])
    _, inv = np.unique(vals, return_inverse=True)
    return inv

def visualize_graph(
    G: nx.Graph,
    node2comm: Optional[Dict[int, int]] = None,
    title: str = "ER Graph",
    with_labels: bool = False,
    figsize: Tuple[int, int] = (12, 9),
    out_path: Optional[Path] = None,
) -> None:
    """Spring layout with community/cluster coloring."""
    if G.number_of_nodes() == 0:
        print("[viz] Empty graph.")
        return

    pos = nx.spring_layout(G, weight="weight", seed=42, k=None)
    degrees = dict(G.degree())
    node_sizes = 200 * (1 + np.log1p(np.array([degrees[n] for n in G.nodes()], dtype=float)))
    colors = _colors_from_mapping_or_cluster_attr(G, node2comm)

    if G.number_of_edges() > 0:
        ew_raw = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]
        wmin, wmax = min(ew_raw), max(ew_raw)
        ew = 0.5 + 3.0 * (np.array(ew_raw) - wmin) / (wmax - wmin + 1e-9)
    else:
        ew = []

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, width=ew if len(ew) else 0.5, alpha=0.35)
    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        cmap="tab20",
        node_size=node_sizes,
        linewidths=0.5,
        edgecolors="k",
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

def export_for_gephi(G: nx.Graph, path: Path) -> None:
    """Export to GEXF for Gephi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, path)
    print(f"[viz] GEXF written -> {path.resolve()}")
