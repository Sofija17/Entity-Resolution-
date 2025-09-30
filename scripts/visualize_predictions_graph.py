#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from src.graph.visualize_er_graph import (
    build_match_graph, sample_subgraph,
    communities_louvain_or_cc, visualize_graph, export_for_gephi
)

def main():
    df = pd.read_csv("../data/classifier_predictions_xgb_filtered.csv")

    G = build_match_graph(df, keep_threshold=0.45)
    G_small = sample_subgraph(G, max_nodes=400)

    node2comm = communities_louvain_or_cc(G_small, use_louvain=True)

    visualize_graph(
        G_small,
        node2comm=node2comm,
        title="ER Match Graph (threshold=0.45, Louvain communities)",
        with_labels=False,
        out_path=Path("../src/graph/er_graph_after_pruning.png")
    )

    export_for_gephi(G, Path("../src/graph/er_graph_full.gexf"))

if __name__ == "__main__":
    main()
