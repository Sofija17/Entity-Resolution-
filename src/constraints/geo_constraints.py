import re
import pandas as pd
from typing import Dict, Tuple, Set, Iterable, List, Optional, Callable

# Whitelisted countries we care about (canonical names)
GEO_COUNTRIES_WHITE_LIST: Set[str] = {
    "United States", "United Kingdom", "Taiwan", "China", "United Arab Emirates",
    "Switzerland", "Greece", "Singapore", "Germany", "Hong Kong", "Canada",
    "Italy", "France", "Australia", "India", "Netherlands", "Israel",
    "Japan", "Brazil", "Denmark",
}

# Ordered mappings: acronyms/aliases -> canonical country names
ACRONYM_MAP_ORDERED: List[Tuple[str, str]] = [
    (r"\bUSA\b",                 "United States"),
    (r"\bUS\b",                  "United States"),
    (r"\bUK\b",                  "United Kingdom"),
    (r"\bROC\b",                 "Taiwan"),
    (r"\bP\.?\s*R\.?\s*China\b", "China"),
    (r"\bPeople's Republic of China\b", "China"),
    (r"\bUAE\b",                 "United Arab Emirates"),
    (r"\bCH\b",                  "Switzerland"),
    (r"\bGR(?=[\W_]|$)",         "Greece"),
    (r"\bS\'?pore(?=[\W_]|$)",   "Singapore"),
    (r"\bSingapor(?=[\W_]|$)",   "Singapore"),
    (r"\bHong\s*Kong\b",         "Hong Kong"),
]

# Detect dotted acronyms like U.S.A., U.S., E.U. (optionally with spaces)
PATTERNDOTTED = re.compile(
    r"(?<![A-Za-z])(?:[A-Z]\.){2,}[A-Z]?(?=\W|$)"
)

def _undot_acronyms(text: str) -> str:
    """
    1) Finds dotted acronyms (e.g., U.S.A., U.S., E.U.)
    2) Converts them to the no-dot/no-space form (USA, US, EU)
    """
    if not isinstance(text, str) or not text:
        return ""

    def _repl(m: re.Match) -> str:
        token = m.group(0)
        return token.replace(".", "").replace(" ", "")

    return PATTERNDOTTED.sub(_repl, text)

def _build_country_subs(
    acronym_map: Iterable[Tuple[str, str]],
    allowed: Set[str]
) -> List[Tuple[re.Pattern, str]]:
    """Keep only substitutions that normalize to allowed canonical countries and compile them."""
    subs: List[Tuple[re.Pattern, str]] = []
    for pat, repl in acronym_map:
        if repl in allowed:
            subs.append((re.compile(pat, re.IGNORECASE), repl))
    return subs

def _make_text_normalizer(
    subs: List[Tuple[re.Pattern, str]]
) -> Callable[[str], str]:
    """Return a function that normalizes text by applying the compiled substitutions in order."""
    def _normalize(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        out = text
        for preg, repl in subs:
            out = preg.sub(repl, out)
        return out
    return _normalize

def _compile_country_patterns(countries: Set[str]) -> Dict[str, re.Pattern]:
    """
    For each canonical country name, compile a word-boundary-ish regex.
    Aliases are handled upstream (undotting + normalization).
    """
    patterns: Dict[str, re.Pattern] = {}
    for c in countries:
        patterns[c.lower()] = re.compile(rf"(?<!\w){re.escape(c)}(?!\w)", re.IGNORECASE)
    return patterns

def _extract_countries_from_text(
    text: str,
    country_patterns: Dict[str, re.Pattern],
    restrict_to: Optional[Set[str]] = None
) -> Set[str]:
    """Return a set of canonical (lowercased) countries mentioned in the text."""
    out: Set[str] = set()
    if not isinstance(text, str) or not text:
        return out
    for cname, pat in country_patterns.items():
        if pat.search(text):
            out.add(cname)
    if restrict_to:
        out &= {c.lower() for c in restrict_to}
    return out

def extract_canonical_countries(
    text: str,
    normalizer: Callable[[str], str],
    country_patterns: Dict[str, re.Pattern],
    restrict_to: Optional[Set[str]] = None
) -> Set[str]:
    """
    Helper that runs the full pipeline for one text:
    dotted-acronyms -> undotted  -> acronym normalization -> canonical country detection.
    """
    undotted = _undot_acronyms(text)
    normalized = normalizer(undotted)
    return _extract_countries_from_text(normalized, country_patterns, restrict_to)

def geo_mismatch_pairs_to_prune(
    edges_df: pd.DataFrame,
    id2text: Dict[int, str],
    restrict_to_countries: Optional[Iterable[str]] = None,
    conservative_when_unknown: bool = True,
) -> Dict[Tuple[int, int], str]:
    """
    Prune (src_id, cand_id) only if:
      - after undotting + normalization both sides mention at least one country, and
      - the country sets are disjoint.
    """
    # Prepare normalizer and country detectors once
    subs = _build_country_subs(ACRONYM_MAP_ORDERED, GEO_COUNTRIES_WHITE_LIST)
    normalizer = _make_text_normalizer(subs)
    patterns = _compile_country_patterns(GEO_COUNTRIES_WHITE_LIST)
    rset = {c.lower().strip() for c in restrict_to_countries} if restrict_to_countries else None

    # Unique pairs to inspect
    work = edges_df[["src_id", "cand_id"]].drop_duplicates()

    to_prune: Dict[Tuple[int, int], str] = {}
    for _, row in work.iterrows():
        a_id = int(row["src_id"])
        b_id = int(row["cand_id"])

        a_raw = (id2text.get(a_id) or "")
        b_raw = (id2text.get(b_id) or "")

        a_countries = extract_canonical_countries(a_raw, normalizer, patterns, rset)
        b_countries = extract_canonical_countries(b_raw, normalizer, patterns, rset)

        if conservative_when_unknown and (not a_countries or not b_countries):
            # Skip pruning if one side has no detected country (be conservative)
            continue

        if a_countries and b_countries and a_countries.isdisjoint(b_countries):
            left = ";".join(sorted(a_countries))
            right = ";".join(sorted(b_countries))
            to_prune[(a_id, b_id)] = f"geo_mismatch:{left}|{right}"

    return to_prune
