#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ONLY the selected classifier features for ALL candidate pairs.

Required input columns (from blocking):
    src_id, cand_id, src_text, cand_text
Optional:
    cosine_sim (from blocking) – preserved as a passthrough column

Selected features produced:
    edit_ratio, jaro_winkler, lcs_ratio,
    token_jaccard, token_cosine,
    tfidf_word_cosine, tfidf_char_cosine,
    dmetaphone_match

Output: <input>_features.csv (or custom path)
"""

from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# required deps (assumed installed)
import jellyfish
from rapidfuzz.fuzz import ratio as rf_ratio


# ======================= TEXT & SIMILARITY HELPERS =======================
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")  # keep letters/digits/underscore

def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    s = _strip_accents(s)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def tokenize_words(s: str) -> List[str]:
    s = normalize_text(s)
    return [t for t in s.split(" ") if t]

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    i = len(a & b); u = len(a | b)
    return i / u if u else 0.0

def lcs_len(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0: return 0
    dp = [0]*(lb+1)
    for i in range(1, la+1):
        prev = 0
        for j in range(1, lb+1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = cur
    return dp[lb]

def lcs_ratio(a: str, b: str) -> float:
    a_n = normalize_text(a); b_n = normalize_text(b)
    if not a_n and not b_n: return 1.0
    L = lcs_len(a_n, b_n)
    den = max(len(a_n), len(b_n))
    return L/den if den else 0.0

def edit_ratio(a: str, b: str) -> float:
    """[0..1] – RapidFuzz ratio/100 (Levenshtein-like normalized similarity)."""
    a_n = normalize_text(a); b_n = normalize_text(b)
    return float(rf_ratio(a_n, b_n)) / 100.0

def jaro_winkler(a: str, b: str) -> float:
    a_n = normalize_text(a); b_n = normalize_text(b)
    return float(jellyfish.jaro_winkler_similarity(a_n, b_n))

def dmetaphone_match(a: str, b: str) -> int:
    """Binary 0/1 using Metaphone equality on the first token."""
    aw = tokenize_words(a)[:1]
    bw = tokenize_words(b)[:1]
    if not aw or not bw: return 0
    dm_a = jellyfish.metaphone(aw[0]) if aw[0] else ""
    dm_b = jellyfish.metaphone(bw[0]) if bw[0] else ""
    return int(dm_a == dm_b)

def _rowwise_cosine(X, Y) -> np.ndarray:
    # safe row-wise cosine for sparse or dense matrices (paired rows)
    if hasattr(X, "multiply"):
        num = X.multiply(Y).sum(axis=1).A1
        xnorm = np.sqrt(X.multiply(X).sum(axis=1)).A1
        ynorm = np.sqrt(Y.multiply(Y).sum(axis=1)).A1
    else:
        num = np.einsum("ij,ij->i", X, Y)
        xnorm = np.linalg.norm(X, axis=1)
        ynorm = np.linalg.norm(Y, axis=1)
    den = xnorm * ynorm
    den[den == 0.0] = 1.0
    return (num / den).astype(float)

def _pair_token_cosine(src_tokens: List[List[str]], cand_tokens: List[List[str]]) -> np.ndarray:
    texts_union = [" ".join(toks) for toks in (src_tokens + cand_tokens)]
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
    M = cv.fit_transform(texts_union)
    n = len(src_tokens)
    A, B = M[:n], M[n:]
    return _rowwise_cosine(A, B)

def _pair_tfidf_word_cosine(src_texts: List[str], cand_texts: List[str]) -> np.ndarray:
    corpus = src_texts + cand_texts
    tf = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1,
                         strip_accents="unicode", sublinear_tf=True)
    X = tf.fit_transform(corpus)
    n = len(src_texts)
    A, B = X[:n], X[n:]
    return _rowwise_cosine(A, B)

def _pair_tfidf_char_cosine(src_texts: List[str], cand_texts: List[str]) -> np.ndarray:
    corpus = src_texts + cand_texts
    tfc = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1)
    X = tfc.fit_transform(corpus)
    n = len(src_texts)
    A, B = X[:n], X[n:]
    return _rowwise_cosine(A, B)


# ======================= FEATURE BUILDER (SELECTED ONLY) =======================
SELECTED_FEATURES = [
    "edit_ratio", "jaro_winkler", "lcs_ratio",
    "token_jaccard", "token_cosine",
    "tfidf_word_cosine", "tfidf_char_cosine",
    "dmetaphone_match",
]

def build_matching_features(
    pairs: pd.DataFrame,
    src_col: str = "src_text",
    cand_col: str = "cand_text",
    keep_original_cols: bool = True
) -> pd.DataFrame:
    df = pairs.copy()

    src_raw  = df[src_col].fillna("").astype(str).tolist()
    cand_raw = df[cand_col].fillna("").astype(str).tolist()

    # Precompute normalized tokens/sets once
    src_toks  = [tokenize_words(s) for s in src_raw]
    cand_toks = [tokenize_words(s) for s in cand_raw]
    src_sets  = [set(t) for t in src_toks]
    cand_sets = [set(t) for t in cand_toks]

    # --- selected features ---
    edit_sim        = np.array([edit_ratio(a, b) for a, b in zip(src_raw, cand_raw)], dtype=float)
    lcs_sim         = np.array([lcs_ratio(a, b)  for a, b in zip(src_raw, cand_raw)], dtype=float)
    jw              = np.array([jaro_winkler(a, b) for a, b in zip(src_raw, cand_raw)], dtype=float)
    dm              = np.array([dmetaphone_match(a, b) for a, b in zip(src_raw, cand_raw)], dtype=float)
    jacc            = np.array([jaccard(a, b)   for a, b in zip(src_sets, cand_sets)], dtype=float)
    tok_cos         = _pair_token_cosine(src_toks, cand_toks)
    tfidf_word_cos  = _pair_tfidf_word_cosine(src_raw, cand_raw)
    tfidf_char_cos  = _pair_tfidf_char_cosine(src_raw, cand_raw)

    feat_cols = pd.DataFrame({
        "edit_ratio":         edit_sim,          # [0..1]
        "jaro_winkler":       jw,                # [0..1]
        "lcs_ratio":          lcs_sim,           # [0..1]
        "token_jaccard":      jacc,              # [0..1]
        "token_cosine":       tok_cos,           # [0..1]
        "tfidf_word_cosine":  tfidf_word_cos,    # [0..1]
        "tfidf_char_cosine":  tfidf_char_cos,    # [0..1]
        "dmetaphone_match":   dm,                # {0,1}
    })

    # Keep original cols if present
    base_cols = [c for c in ["src_id","cand_id","cosine_sim","src_text","cand_text"] if c in df.columns]
    out = pd.concat([df[base_cols].reset_index(drop=True), feat_cols], axis=1) if keep_original_cols else feat_cols
    return out


# ======================= CHUNKED WHOLE-DATA RUNNER =======================
def build_features_for_csv(
    in_csv: Path,
    out_csv: Optional[Path] = None,
    chunksize: int = 200_000,
) -> Path:
    """
    Reads the full candidates CSV in chunks, computes ONLY selected features, and writes out.
    """
    in_csv = Path(in_csv)
    if out_csv is None:
        out_csv = in_csv.with_name(in_csv.stem + "_features.csv")
    else:
        out_csv = Path(out_csv)

    if out_csv.exists():
        out_csv.unlink()

    total_rows = 0
    chunk_no = 0

    for chunk in pd.read_csv(in_csv, chunksize=chunksize):
        chunk_no += 1
        print(f"[features] processing chunk {chunk_no} (rows={len(chunk):,})...")
        feats = build_matching_features(chunk)

        header = (chunk_no == 1)
        feats.to_csv(out_csv, mode="a", header=header, index=False)

        total_rows += len(chunk)
        print(f"[features] appended {len(chunk):,} rows (total so far: {total_rows:,})")

    print(f"[done] wrote {total_rows:,} rows to {out_csv}")
    return out_csv


# ======================= MAIN ===========================================
if __name__ == "__main__":
    # Path to your mapped candidates (from blocking)
    IN_CANDIDATES = Path("../../data/er_blocking_candidates_k20.csv")  # <-- change if needed

    # Optional custom output path
    OUT_FEATURES = None  # e.g., Path("../../data/er_blocking_candidates_k20_features.csv")

    # Chunk size (tune for RAM)
    CHUNK = 150_000

    build_features_for_csv(IN_CANDIDATES, OUT_FEATURES, chunksize=CHUNK)
