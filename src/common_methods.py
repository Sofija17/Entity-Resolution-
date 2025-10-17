import re
import pandas as pd
import unicodedata
from typing import List


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

#Remove accent marks (é → e)
def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

#normalize + lowercase + accent strip + extract alphanumeric tokens
def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = _strip_accents(text.lower())
    return TOKEN_RE.findall(text)


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


