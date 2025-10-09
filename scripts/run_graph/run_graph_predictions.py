#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from src.graph.build_graph_from_predictions import build_graph_from_predictions
from src.graph.visualize_graph_utils import (
    sample_subgraph, communities_louvain_or_cc, visualize_graph, export_for_gephi
)

def main():
    df_pred = pd.read_csv("../../data/classifier_predictions/classifier_predictions_xgb_filtered.csv")
    G = build_graph_from_predictions(df_pred, keep_threshold=0.45)
    G_small = sample_subgraph(G, max_nodes=400)
    node2comm = communities_louvain_or_cc(G_small, use_louvain=True)

    visualize_graph(
        G_small,
        node2comm=node2comm,
        title="Pre-Geo/Transitivity (threshold=0.45, Louvain)",
        with_labels=False,
        out_path=Path("../src/graph/er_graph_pred.png"),
    )
    export_for_gephi(G, Path("../../src/graph/er_graph_pred_full.gexf"))

if __name__ == "__main__":
    main()
