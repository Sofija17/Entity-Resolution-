# scripts/apply_transitivity.py
from pathlib import Path
import pandas as pd
import networkx as nx

IN_DEFAULTS = [
    Path("scripts/out/classifier_predictions_xgb_filtered.csv"),
    Path("classifier_predictions_xgb_filtered.csv"),
]
OUT = Path("scripts/out/er_clusters_transitive.csv")
TAU_LOW = 0.70  # низок праг за да не изгубиме кандидати (прилагоди 0.6–0.8)

ID1_CANDIDATES = ["src_id", "left_id", "id_left", "u", "id1"]
ID2_CANDIDATES = ["cand_id", "right_id", "id_right", "v", "id2"]
SCORE_CANDIDATES = ["prob", "prob_match", "score", "p"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    raise ValueError(f"Missing required columns, none of {cands} found")

def load_input():
    for p in IN_DEFAULTS:
        if p.exists():
            return pd.read_csv(p), p
    raise FileNotFoundError("classifier_predictions_xgb_filtered.csv not found in expected locations")

def components_at_threshold(edges, id1, id2, w, tau):
    sub = edges[edges[w] >= tau].copy()
    if sub.empty:
        return []
    G = nx.Graph()
    G.add_edges_from(zip(sub[id1].astype(str), sub[id2].astype(str)))
    return [set(c) for c in nx.connected_components(G)]

def main():
    edges, used_in = load_input()
    id1 = pick_col(edges, ID1_CANDIDATES)
    id2 = pick_col(edges, ID2_CANDIDATES)
    w   = pick_col(edges, SCORE_CANDIDATES)
    cohort = "country" if "country" in edges.columns else None

    rows = []
    if cohort:
        for coh, sub in edges.groupby(cohort, dropna=False):
            comps = components_at_threshold(sub, id1, id2, w, TAU_LOW)
            for i, comp in enumerate(comps, 1):
                for n in comp:
                    rows.append({"node_id": n, "cluster_id": f"{coh or 'NA'}:C{i}", "country": coh})
    else:
        comps = components_at_threshold(edges, id1, id2, w, TAU_LOW)
        for i, comp in enumerate(comps, 1):
            for n in comp:
                rows.append({"node_id": n, "cluster_id": f"C{i}"})

    out_df = pd.DataFrame(rows).sort_values(["cluster_id", "node_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"[OK] Wrote {len(out_df)} rows to {OUT} using τ_low={TAU_LOW}")

if __name__ == "__main__":
    main()
