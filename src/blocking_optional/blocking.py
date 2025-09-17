import re
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Set

import pandas as pd

# --- helpers ---------------------------------------------------------------

def parse_labeled_tokens(s: str) -> List[Tuple[str, str]]:
    """
    Parse 'text<LABEL>; other<L2>' -> [(text, LABEL), ...]
    """
    if not isinstance(s, str) or not s.strip():
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    out = []
    for p in parts:
        m = re.match(r"^(.*)<([^<>]+)>$", p)
        if m:
            text = m.group(1).strip().rstrip(",:;")
            label = m.group(2).strip()
            if text:
                out.append((text, label))
    return out

_alnum = re.compile(r"[A-Za-z0-9]+")

def normalize_head(text: str) -> str:
    # first alphanumeric token lowercased
    m = _alnum.search(text)
    return m.group(0).lower() if m else text.lower()

def trigram(text: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]", "", text).lower()
    return t[:3] if len(t) >= 3 else t

def is_two_letter_state(tok: str) -> bool:
    return len(tok) == 2 and tok.isupper()

def looks_like_zip(tok: str) -> bool:
    return bool(re.fullmatch(r"\d{4,6}", tok))

def org_acronym(text: str) -> str:
    # initials of alnum words, keep common symbols like & if present (AT&T)
    words = re.findall(r"[A-Za-z0-9]+|&", text)
    # build acronym from uppercase initials or & when present
    letters = []
    for w in words:
        if w == "&":
            letters.append("&")
        elif w.isalpha():
            letters.append(w[0].upper())
        elif w.isdigit():
            letters.append(w[0])
    ac = "".join(letters)
    # prefer original known acronyms if text itself is all-caps short
    if text.strip().upper() == text.strip() and len(text.strip()) <= 6:
        return text.strip().upper()
    return ac

def pairs_from_ids(ids: List[int]) -> Set[Tuple[int, int]]:
    out = set()
    for a, b in itertools.combinations(sorted(ids), 2):
        out.add((a, b))
    return out

# --- blocking --------------------------------------------------------------

def build_blocks(df: pd.DataFrame) -> Dict[str, Set[int]]:
    """
    Build multiple block keys -> set(id1) using typed tokens.
    Returns a dict of {block_key: {id1,...}}
    """
    blocks: Dict[str, Set[int]] = defaultdict(set)

    for row in df.itertuples(index=False):
        rid = int(getattr(row, "id1"))
        labeled = parse_labeled_tokens(getattr(row, "affil_tokens_labeled", ""))

        # split by label
        orgs   = [t for t, lab in labeled if lab == "ORG"]
        gpes   = [t for t, lab in labeled if lab == "GPE"]
        cards  = [t for t, lab in labeled if lab == "CARDINAL"]

        # --- Rule 1: ZIP / Postal (very tight)
        for c in cards:
            tok = c.strip()
            if looks_like_zip(tok):
                blocks[f"ZIP:{tok}"].add(rid)

        # --- Rule 2: GPE tokens (country/state/city)
        for g in gpes:
            tok = g.strip()
            # city-like (>=3 chars)
            if len(tok) >= 3 and not is_two_letter_state(tok):
                blocks[f"CITY:{tok.lower()}"].add(rid)
            # 2-letter states (US) — separate
            if is_two_letter_state(tok):
                blocks[f"STATE:{tok}"].add(rid)
            # generic GPE catch-all
            blocks[f"GPE:{tok.lower()}"].add(rid)

        # --- Rule 3: Organization head / trigram / acronym
        if orgs:
            first_org = orgs[0]
            head = normalize_head(first_org)
            tg   = trigram(first_org)
            ac   = org_acronym(first_org)

            if head:
                blocks[f"ORGHEAD:{head}"].add(rid)
            if tg:
                blocks[f"ORG3:{tg}"].add(rid)
            if ac:
                blocks[f"ACR:{ac}"].add(rid)

    return blocks

def candidate_pairs_from_blocks(blocks: Dict[str, Set[int]],
                                min_block_size: int = 2,
                                max_block_size: int = 2000) -> Set[Tuple[int, int]]:
    """
    Turn blocks into a union of unique candidate pairs.
    Skip trivial or huge blocks (safety).
    """
    candidates: Set[Tuple[int, int]] = set()
    for key, ids in blocks.items():
        n = len(ids)
        if n < min_block_size or n > max_block_size:
            continue
        candidates |= pairs_from_ids(list(ids))
    return candidates

# Optional quick post-filter to keep only pairs sharing >=2 tokens (plain)
def postfilter_by_token_overlap(df: pd.DataFrame,
                                pairs: Set[Tuple[int, int]],
                                min_overlap: int = 2) -> Set[Tuple[int, int]]:
    by_id = {int(r.id1): set(str(r.affil_tokens).lower().split(";")) for r in df.itertuples(index=False)}
    clean = set()
    for a, b in pairs:
        tok_a = {t.strip() for t in by_id.get(a, set()) if t.strip()}
        tok_b = {t.strip() for t in by_id.get(b, set()) if t.strip()}
        if len(tok_a & tok_b) >= min_overlap:
            clean.add((a, b))
    return clean