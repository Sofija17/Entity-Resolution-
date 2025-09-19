import re
from pathlib import Path
from typing import List, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------- user knobs (no file outputs) ----------
CSV_PATHS_TRY = [
    "../../data/affiliationstrings_ids_with_tokens.csv",
]
K_VALUES = [5, 10, 20, 30, 40, 50, 75, 100]

# TF-IDF config (used for embeddings + IDF/DF stats)
NGRAM_MIN, NGRAM_MAX = 1, 2
MIN_DF_ABS = 2                 # ignore tokens seen in <2 docs (too rare/noisy)
MAX_DF_FRAC_FOR_VOCAB = 0.99   # ignore tokens present in >99% docs when building vocab

# Automatic stopword learning
LOW_IDF_PERCENTILE = 0.20      # bottom X% IDF -> stopwords (very common)
HIGH_DF_PERCENT = 0.20         # tokens appearing in >20% of docs -> stopwords
KEEP_ACRONYMS = True           # always keep acronyms like 'MIT','ETH','IBM' even if frequent

# Jaccard purity (optional complement to overlap>=1 token)
COMPUTE_JACCARD = True
JACCARD_MIN = 0.30
# ---------------------------------------------------

ALNUM = re.compile(r"[A-Za-z0-9]+")
LOWER_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")
ACRONYM = re.compile(r"^[A-Z0-9]{2,}([&\-][A-Z0-9]{1,})*$")

def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    text_candidates = [c for c in df.columns if c.lower() in
                       ("affil1","affiliation","affil","text","affil_clean","affiliation_text","name")]
    id_candidates   = [c for c in df.columns if c.lower() in
                       ("id1","id","record_id","row_id")]
    text_col = text_candidates[0] if text_candidates else df.columns[0]
    id_col   = id_candidates[0]   if id_candidates   else (df.columns[1] if len(df.columns)>1 else df.columns[0])
    return text_col, id_col

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [t.lower() for t in LOWER_TOKEN.findall(text)]

def find_acronyms(raw: str) -> Set[str]:
    acr = set()
    for w in re.findall(r"[A-Z0-9&\-]{2,}", raw or ""):
        cleaned = re.sub(r"[^A-Za-z0-9]", "", w)
        if len(cleaned) >= 2 and cleaned.isupper():
            acr.add(cleaned.lower())
    return acr

def build_tfidf(texts: List[str]):
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(NGRAM_MIN, NGRAM_MAX),
        min_df=MIN_DF_ABS,
        max_df=MAX_DF_FRAC_FOR_VOCAB,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return X, vec

def derive_stopwords_auto(X, vec, low_idf_pct: float, high_df_percent: float, keep_acronyms: bool):
    """
    Learn stopwords from corpus:
      - low-IDF tokens (<= percentile)
      - high-DF tokens (> high_df_percent of docs)
    Keep acronyms if requested.
    Returns: set[str] stopwords (lowercased).
    """
    vocab = vec.vocabulary_                 # token -> col
    inv_vocab = {j:i for i,j in vocab.items()}
    idf = vec.idf_
    N = X.shape[0]

    # low-IDF cutoff
    idf_cut = np.quantile(idf, low_idf_pct)
    low_idf_cols = set(np.where(idf <= idf_cut)[0])

    # document frequency per term (count of nonzeros in column)
    # X is CSR; get DF via boolean count
    df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    df_frac = df_counts / N
    high_df_cols = set(np.where(df_frac > high_df_percent)[0])

    stop_cols = low_idf_cols | high_df_cols
    stopwords = {inv_vocab[j] for j in stop_cols}

    if keep_acronyms:
        # remove acronyms from stopwords
        acros_to_keep = {tok for tok in stopwords if tok.isupper() or tok.upper() == tok}
        # above check is weak since tokens are lowercased in vocab; instead, we keep common short tokens
        acros_to_keep |= {tok for tok in stopwords if len(tok) <= 5 and tok.isalpha()}
        # safer approach: we can't perfectly detect acronyms here because vec lowercases tokens
        # We'll re-add acronyms later from raw strings; here just return stopwords as-is.
        pass

    return stopwords

def important_token_sets(texts: List[str], X, vec, stopwords: Set[str]) -> List[Set[str]]:
    """
    Important tokens per doc:
      (tokens in vocab that are NOT in stopwords) ∪ acronyms found in raw text.
    """
    vocab = set(vec.vocabulary_.keys())
    imp_sets = []
    for raw in texts:
        toks = set(tokenize(raw))
        acrs = find_acronyms(raw)  # all-caps from raw, kept regardless
        imp = (toks & vocab) - stopwords
        imp |= acrs
        imp_sets.append(imp)
    return imp_sets

def knn_indices(X, max_k: int):
    nn = NearestNeighbors(n_neighbors=max_k+1, metric="cosine", algorithm="brute")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)  # includes self at [:,0]
    sims = 1.0 - dists
    return idx, sims

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def main():
    # locate CSV
    csv_path = None
    for p in CSV_PATHS_TRY:
        pp = Path(p)
        if pp.exists():
            csv_path = pp
            break
    if csv_path is None:
        raise FileNotFoundError(f"CSV not found. Tried: {', '.join(CSV_PATHS_TRY)}")
    print(f"[info] Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    text_col, id_col = detect_cols(df)
    texts = df[text_col].fillna("").astype(str).tolist()
    N = len(texts)
    print(f"[info] N={N}, text_col='{text_col}', id_col='{id_col}'")

    # embeddings + learned stopwords
    X, vec = build_tfidf(texts)
    stopwords = derive_stopwords_auto(
        X, vec,
        low_idf_pct=LOW_IDF_PERCENTILE,
        high_df_percent=HIGH_DF_PERCENT,
        keep_acronyms=KEEP_ACRONYMS
    )
    imp_sets = important_token_sets(texts, X, vec, stopwords)

    # neighbors
    max_k = min(max(K_VALUES), max(2, N-1))
    neigh_idx, sims = knn_indices(X, max_k)

    # sweep k
    ks = [k for k in K_VALUES if k < N]
    overlap_purity, jacc_purity, avgcos, redratio = [], [], [], []
    for k in ks:
        top_idx  = neigh_idx[:, 1:k+1]
        top_sims = sims[:, 1:k+1]

        # Overlap ≥1 important token
        overlap_hits = []
        jacc_hits = []
        for i in range(N):
            src_set = imp_sets[i]
            neighbors = top_idx[i]
            share = 0
            jac = 0
            if k > 0:
                for nb in neighbors:
                    nb_set = imp_sets[int(nb)]
                    if (src_set & nb_set):
                        share += 1
                    if COMPUTE_JACCARD and jaccard(src_set, nb_set) >= JACCARD_MIN:
                        jac += 1
                overlap_hits.append(share / k)
                if COMPUTE_JACCARD:
                    jacc_hits.append(jac / k)
            else:
                overlap_hits.append(0.0)
                if COMPUTE_JACCARD:
                    jacc_hits.append(0.0)

        overlap_purity.append(float(np.mean(overlap_hits)))
        if COMPUTE_JACCARD:
            jacc_purity.append(float(np.mean(jacc_hits)))
        avgcos.append(float(top_sims.mean()) if k > 0 else 0.0)

        total_pairs = N * k
        baseline = N * (N - 1)
        redratio.append(1.0 - (total_pairs / baseline))

    # plots
    def plot_xy(x, y, xl, yl, title):
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel(xl); plt.ylabel(yl); plt.title(title); plt.grid(True)
        plt.tight_layout()

    plot_xy(ks, overlap_purity,
            "k (top-k neighbors)", "OverlapPurity@k (≥1 important token)",
            "Token-set Overlap Purity vs k (auto stopwords)")

    if COMPUTE_JACCARD:
        plot_xy(ks, jacc_purity,
                "k (top-k neighbors)", f"JaccardPurity@k (J ≥ {JACCARD_MIN})",
                "Jaccard Purity vs k (important-token sets)")

    plot_xy(ks, avgcos, "k (top-k neighbors)", "AvgCosine@k", "Average Cosine Similarity vs k")
    plot_xy(ks, redratio, "k (top-k neighbors)", "Reduction Ratio (directed)", "Reduction Ratio vs k")

    plt.show()

if __name__ == "__main__":
    main()
