#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning/embedding-based blocking:
- Embed with TF-IDF
- Retrieve top-K neighbors via ANN (HNSW if available, else exact cosine kNN)
- Output candidate pairs restricted to those neighbors
- ALSO write a *mapped* CSV with src_text and cand_text for inspection
"""
import re
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    import hnswlib
    HNSW_AVAILABLE = True
except Exception:
    HNSW_AVAILABLE = False

ALNUM = re.compile(r"[A-Za-z0-9]+")

# ----------------------------- HELPERS ---------------------------------
def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    text_candidates = [c for c in df.columns if c.lower() in
                       ("affil1","affiliation","affil","text","affil_clean","affiliation_text","name")]
    id_candidates   = [c for c in df.columns if c.lower() in
                       ("id1","id","record_id","row_id")]
    text_col = text_candidates[0] if text_candidates else df.columns[0]
    id_col   = id_candidates[0]   if id_candidates   else (df.columns[1] if len(df.columns)>1 else df.columns[0])
    return text_col, id_col

def build_tfidf(texts: List[str], ngram_min=1, ngram_max=2, min_df=2, max_df=0.9):
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df, max_df=max_df,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return X, vec

def ann_knn(X, k: int, backend: str = "auto", ef: int = 200, M: int = 32):
    """
    Returns (indices, similarities, backend_used) with shape (N, k+1), including self at [:,0].
    """
    N = X.shape[0]
    k_eff = min(k, N-1)

    if backend == "auto":
        backend = "hnsw" if HNSW_AVAILABLE and X.shape[1] <= 1000 else "sklearn"

    if backend == "hnsw":
        dense = X.astype(np.float32).toarray()
        dim = dense.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=N, ef_construction=ef, M=M)
        index.add_items(dense, np.arange(N, dtype=np.int32))
        index.set_ef(max(ef, k_eff*2))
        labels, dists = index.knn_query(dense, k=k_eff+1)
        sims = 1.0 - dists
        return labels, sims, "hnsw"

    # sklearn fallback (exact)
    knn = NearestNeighbors(n_neighbors=k_eff+1, metric="cosine", algorithm="brute")
    knn.fit(X)
    dists, idx = knn.kneighbors(X, return_distance=True)
    sims = 1.0 - dists
    return idx, sims, "sklearn"

def build_candidates(ids: List, neigh_idx: np.ndarray, sims: np.ndarray, k: int,
                     undirected: bool = False, min_sim: Optional[float] = None) -> pd.DataFrame:
    """
    Build candidate pairs, excluding self. Optional min_sim to filter.
    If undirected=True, collapse (a,b) and (b,a) to one row keeping max sim.
    """
    N = len(ids)
    rows = []
    for i in range(N):
        src = ids[i]
        neighbors = neigh_idx[i, 1:k+1]
        neigh_sims = sims[i, 1:k+1]
        for j, s in zip(neighbors, neigh_sims):
            if int(j) == i:
                continue
            if (min_sim is not None) and (s < min_sim):
                continue
            rows.append((src, ids[int(j)], float(s)))

    df = pd.DataFrame(rows, columns=["src_id","cand_id","cosine_sim"])

    if undirected and not df.empty:
        key = df.apply(lambda r: tuple(sorted((r["src_id"], r["cand_id"]))), axis=1)
        df["pair_key"] = key
        df = (df.sort_values("cosine_sim", ascending=False)
                .groupby("pair_key", as_index=False)
                .first()[["src_id","cand_id","cosine_sim"]])
    return df

def enrich_with_text(cand_df: pd.DataFrame, df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    id_to_text = dict(zip(df[id_col], df[text_col]))
    out = cand_df.copy()
    out["src_text"]  = out["src_id"].map(id_to_text)
    out["cand_text"] = out["cand_id"].map(id_to_text)
    # nicer order
    return out[["src_id","cand_id","cosine_sim","src_text","cand_text"]]

# ----------------------------- RUNNER ----------------------------------
def run_blocking(
    csv_path: Path,
    k: int = 20,                                # <<<<<<<<<<<<<<  set your chosen K here
    out_csv: Optional[Path] = None,             # mapped CSV path (auto-named if None)
    text_col: Optional[str] = None,
    id_col: Optional[str]   = None,
    backend: str = "auto",                      # "auto" | "sklearn" | "hnsw"
    undirected: bool = False,
    min_sim: Optional[float] = None,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.9,
) -> Path:
    df = pd.read_csv(csv_path)
    if not text_col or text_col not in df.columns or (id_col and id_col not in df.columns):
        auto_text, auto_id = detect_cols(df)
        text_col = text_col if (text_col and text_col in df.columns) else auto_text
        id_col   = id_col   if (id_col   and id_col   in df.columns) else auto_id

    texts = df[text_col].fillna("").astype(str).tolist()
    ids   = df[id_col].tolist()

    X, _ = build_tfidf(texts, ngram_min, ngram_max, min_df, max_df)
    neigh_idx, sims, used_backend = ann_knn(X, k, backend=backend)
    print(f"[info] ANN backend: {used_backend}; N={X.shape[0]}, k={k}")

    cand_df = build_candidates(ids, neigh_idx, sims, k, undirected=undirected, min_sim=min_sim)
    cand_df = (cand_df.sort_values(["src_id","cosine_sim"], ascending=[True, False])
                        .groupby("src_id", as_index=False).head(k))

    mapped = enrich_with_text(cand_df, df, id_col=id_col, text_col=text_col)

    # choose output path
    if out_csv is None:
        out_csv = csv_path.parent / f"er_blocking_candidates_k{k}.csv"
    mapped.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv} with {len(mapped):,} rows")
    return out_csv

# ----------------------------- CONFIG ----------------------------------
if __name__ == "__main__":
    # Set your paths and K here and just Run in PyCharm:
    CSV = Path("../../data/affiliationstrings_ids_with_tokens.csv")   # adjust if needed
    K   = 20                                                    # <<<<<<<<<<<<<< your chosen k
    OUT = None  # or Path(f"data/er_blocking_candidates_k{K}.csv")

    run_blocking(
        csv_path=CSV,
        k=K,
        out_csv=OUT,
        backend="auto",        # "auto" tries HNSW if available, else sklearn
        undirected=False,      # set True to collapse (a,b) & (b,a)
        min_sim=None           # e.g. 0.25 if you want a cosine floor
    )
