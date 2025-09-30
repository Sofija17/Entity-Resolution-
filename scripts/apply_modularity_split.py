# scripts/apply_modularity_split.py
from pathlib import Path
import pandas as pd
import networkx as nx

# optional: Louvain
try:
    import community as community_louvain  # pip install python-louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False
from networkx.algorithms.community import greedy_modularity_communities

EDGES_IN = Path("scripts/out/classifier_predictions_xgb_filtered.csv")
COARSE_IN = Path("scripts/out/er_clusters_transitive.csv")
OUT = Path("scripts/out/er_clusters_modularity.csv")

ID1_CANDIDATES = ["src_id", "left_id", "id_left", "u", "id1"]
ID2_CANDIDATES = ["cand_id", "right_id", "id_right", "v", "id2"]
SCORE_CANDIDATES = ["prob", "prob_match", "score", "p"]

# Праг за внатрешно прво чистење (опционално); None за да користи сите ребра
INTRA_PRUNE_TAU = None  # на пр. 0.50 ако сакаш да ги исечеш најслабите мостови
RESOLUTION = 1.0        # Louvain resolution (повисоко -> повеќе/помали заедници)

def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required columns, none of {cands} found")

def build_graph_for_nodes(edges, id1, id2, w, nodes, tau=None):
    sub = edges[edges[id1].astype(str).isin(nodes) & edges[id2].astype(str).isin(nodes)].copy()
    if tau is not None:
        sub = sub[sub[w] >= tau]
    G = nx.Graph()
    for u, v, wt in zip(sub[id1].astype(str), sub[id2].astype(str), sub[w]):
        # можеш да ре-важиш, напр. weight=-log(1-wt), но држи едноставно прво
        G.add_edge(u, v, weight=float(wt))
    # осигурај се дека изолатите влегуваат (ако немаат ребра)
    for n in nodes:
        if n not in G:
            G.add_node(n)
    return G

def split_component(G):
    if G.number_of_edges() == 0:
        return [set(G.nodes())]
    if HAS_LOUVAIN:
        part = community_louvain.best_partition(G, weight="weight", resolution=RESOLUTION, random_state=42)
        # group by community id
        comm_to_nodes = {}
        for n, cid in part.items():
            comm_to_nodes.setdefault(cid, set()).add(n)
        return list(comm_to_nodes.values())
    else:
        comms = list(greedy_modularity_communities(G, weight="weight"))
        return [set(c) for c in comms]

def main():
    edges = pd.read_csv(EDGES_IN)
    coarse = pd.read_csv(COARSE_IN)

    id1 = pick_col(edges, ID1_CANDIDATES)
    id2 = pick_col(edges, ID2_CANDIDATES)
    w   = pick_col(edges, SCORE_CANDIDATES)

    # coarse clusters
    grouped = coarse.groupby("cluster_id")["node_id"].apply(lambda s: set(map(str, s.tolist())))

    rows = []
    new_id_counter = 1
    for coarse_id, nodes in grouped.items():
        G = build_graph_for_nodes(edges, id1, id2, w, nodes, tau=INTRA_PRUNE_TAU)
        sub_comms = split_component(G)
        for j, comm in enumerate(sub_comms, 1):
            new_cid = f"{coarse_id}::M{j}"
            for n in sorted(comm):
                rows.append({"node_id": n, "cluster_id": new_cid})
            new_id_counter += 1

    out = pd.DataFrame(rows).sort_values(["cluster_id", "node_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[OK] Wrote {len(out)} rows to {OUT} (Louvain={HAS_LOUVAIN}, resolution={RESOLUTION}, prune_tau={INTRA_PRUNE_TAU})")

if __name__ == "__main__":
    main()
