# scripts/make_candidates.py
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.blocking_optional.blocking import build_blocks, candidate_pairs_from_blocks, postfilter_by_token_overlap

def build_candidates(csv_path: str, out_pairs_path: str):
    df = pd.read_csv(csv_path)
    # Build blocks
    blocks = build_blocks(df)
    # Union of within-block pairs
    cand = candidate_pairs_from_blocks(blocks, min_block_size=2, max_block_size=5000)
    # Optional: tighten with token-overlap >= 2
    cand = postfilter_by_token_overlap(df, cand, min_overlap=2)

    # Save as a 2-column CSV: id_left, id_right
    pairs_df = pd.DataFrame(sorted(list(cand)), columns=["id_left", "id_right"])
    Path(out_pairs_path).parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(out_pairs_path, index=False, encoding="utf-8")
    print(f"Candidates: {len(pairs_df)} saved to {out_pairs_path}")

if __name__ == "__main__":
    build_candidates("../data/affiliationstrings_ids_with_tokens.csv",
                     "../data/processed/candidate_pairs.csv")
