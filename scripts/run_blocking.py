"""
Learning/embedding-based blocking:
- TF-IDF embeddings
- Top-K neighbors via ANN (HNSW if available, else exact cosine kNN)
- Outputs candidate pairs and a mapped CSV with src_text and cand_text
"""
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


def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristically detect text and id columns from a DataFrame."""
    text_candidates = [c for c in df.columns if c.lower() in
                       ("affil1", "affiliation", "affil", "text", "affil_clean", "affiliation_text", "name")]
    id_candidates = [c for c in df.columns if c.lower() in
                     ("id1", "id", "record_id", "row_id")]
    text_col = text_candidates[0] if text_candidates else df.columns[0]
    id_col = id_candidates[0] if id_candidates else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    return text_col, id_col


def build_tfidf(
    texts: List[str],
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.9,
):
    """Fit TF-IDF vectorizer and transform texts."""
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return X, vec


def ann_knn(X, k: int, backend: str = "auto", ef: int = 200, M: int = 32):
    """
    Find approximate/exact top-k neighbors.
    Returns (indices, similarities, backend_used); each row includes self at [:, 0].
    """
    N = X.shape[0]
    k_eff = max(0, min(k, N - 1))

    if backend == "auto":
        backend = "hnsw" if HNSW_AVAILABLE and X.shape[1] <= 1000 else "sklearn"

    if backend == "hnsw":
        dense = X.astype(np.float32).toarray()
        dim = dense.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=N, ef_construction=ef, M=M)
        index.add_items(dense, np.arange(N, dtype=np.int32))
        index.set_ef(max(ef, k_eff * 2 if k_eff > 0 else 50))
        labels, dists = index.knn_query(dense, k=(k_eff + 1) if k_eff > 0 else 1)
        sims = 1.0 - dists
        return labels, sims, "hnsw"

    knn = NearestNeighbors(n_neighbors=(k_eff + 1) if k_eff > 0 else 1, metric="cosine", algorithm="brute")
    knn.fit(X)
    dists, idx = knn.kneighbors(X, return_distance=True)
    sims = 1.0 - dists
    return idx, sims, "sklearn"


def build_candidates(
    ids: List,
    neigh_idx: np.ndarray,
    sims: np.ndarray,
    undirected: bool = False,
    min_sim: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build candidate pairs (src_id, cand_id, cosine_sim), excluding self rows.
    If undirected=True, collapse (a,b) and (b,a) keeping max similarity.
    """
    N = len(ids)
    rows = []
    for i in range(N):
        # exclude column 0 (self)
        neighbors = neigh_idx[i, 1:]
        neigh_sims = sims[i, 1:]
        for j, s in zip(neighbors, neigh_sims):
            jj = int(j)
            if jj == i:
                continue
            if (min_sim is not None) and (s < min_sim):
                continue
            rows.append((ids[i], ids[jj], float(s)))

    df = pd.DataFrame(rows, columns=["src_id", "cand_id", "cosine_sim"])

    if undirected and not df.empty:
        key = df.apply(lambda r: tuple(sorted((r["src_id"], r["cand_id"]))), axis=1)
        df["pair_key"] = key
        df = (
            df.sort_values("cosine_sim", ascending=False)
              .groupby("pair_key", as_index=False)
              .first()[["src_id", "cand_id", "cosine_sim"]]
        )
    return df


def enrich_with_text(cand_df: pd.DataFrame, df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    """Attach source and candidate texts to the candidate pairs."""
    id_to_text = dict(zip(df[id_col], df[text_col]))
    out = cand_df.copy()
    out["src_text"] = out["src_id"].map(id_to_text)
    out["cand_text"] = out["cand_id"].map(id_to_text)
    return out[["src_id", "cand_id", "cosine_sim", "src_text", "cand_text"]]


def run_blocking(
    csv_path: Path,
    k: int = 20,
    out_csv: Optional[Path] = None,
    text_col: Optional[str] = None,
    id_col: Optional[str] = None,
    backend: str = "auto",
    undirected: bool = False,
    min_sim: Optional[float] = None,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.9,
) -> Path:
    """Run TF-IDF + ANN blocking and write a mapped CSV of candidate pairs."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if (not text_col or text_col not in df.columns) or (id_col and id_col not in df.columns):
        auto_text, auto_id = detect_cols(df)
        text_col = text_col if (text_col and text_col in df.columns) else auto_text
        id_col = id_col if (id_col and id_col in df.columns) else auto_id

    texts = df[text_col].fillna("").astype(str).tolist()
    ids = df[id_col].tolist()

    X, _ = build_tfidf(texts, ngram_min, ngram_max, min_df, max_df)
    neigh_idx, sims, used_backend = ann_knn(X, k, backend=backend)
    print(f"[info] ANN backend: {used_backend}; N={X.shape[0]}, k={k}")

    cand_df = build_candidates(ids, neigh_idx, sims, undirected=undirected, min_sim=min_sim)
    cand_df = (
        cand_df.sort_values(["src_id", "cosine_sim"], ascending=[True, False])
               .groupby("src_id", as_index=False)
               .head(k)
    )

    mapped = enrich_with_text(cand_df, df, id_col=id_col, text_col=text_col)

    out_csv = out_csv or (csv_path.parent / f"er_blocking_candidates_k{k}.csv")
    mapped.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv} with {len(mapped):,} rows")
    return out_csv


if __name__ == "__main__":
    CSV = Path("../data/affiliationstrings_ids_with_tokens.csv")
    K = 20
    run_blocking(csv_path=CSV, k=K, backend="auto", undirected=False, min_sim=None)
