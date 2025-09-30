import re
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

_alnum = re.compile(r"[A-Za-z]+")

def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return _alnum.findall(text.lower())


def _id2text(
        df: pd.DataFrame,
        id_col: str,
        text_col: str
) -> dict[int, str]:
    """
    Your edges file only has src_id, cand_id, prob_match.
    Constraints like token_overlap_constraint need the actual strings for those IDs.
    _id2text extracts the column with the text from a given id (the affiliations_ids dataframe should be passed)
    """
    tmp = df.dropna(subset=[id_col, text_col]).drop_duplicates(subset=[id_col]).copy()
    # Optional hygiene: strip trivial punctuation/whitespace
    tmp[text_col] = tmp[text_col].astype(str).str.strip(" ,;\t")
    return dict(zip(tmp[id_col].astype(int), tmp[text_col]))


