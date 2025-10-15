from pathlib import Path
import pandas as pd

from run_blocking import run_blocking

def build_candidates(
    csv_path: str,
    out_pairs_path: str,
    k: int = 20,
    min_sim: float | None = None,
    undirected: bool = True,
    text_col: str | None = None,
    id_col: str | None = None,
    backend: str = "auto",
) -> Path:
    """
    Build candidate pairs using TF-IDF + ANN blocking (from run_blocking) and
    save a 2-column CSV: id_left, id_right (unique undirected pairs).
    """
    csv_path = Path(csv_path).resolve()
    out_pairs_path = Path(out_pairs_path).resolve()

    mapped_path = run_blocking(
        csv_path=csv_path,
        k=k,
        out_csv=None,            # auto-named mapped CSV
        text_col=text_col,
        id_col=id_col,
        backend=backend,
        undirected=undirected,   # collapse (a,b)/(b,a) early
        min_sim=min_sim,
    )

    mapped = pd.read_csv(mapped_path)
    if not {"src_id", "cand_id"}.issubset(mapped.columns):
        raise ValueError(f"Mapped CSV must contain 'src_id' and 'cand_id'. Got: {mapped.columns.tolist()}")

    pairs_df = (
        mapped.loc[mapped["src_id"] != mapped["cand_id"], ["src_id", "cand_id"]]
              .assign(
                  id_left=lambda df: df[["src_id", "cand_id"]].min(axis=1),
                  id_right=lambda df: df[["src_id", "cand_id"]].max(axis=1),
              )[["id_left", "id_right"]]
              .drop_duplicates()
              .reset_index(drop=True)
    )

    out_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(out_pairs_path, index=False, encoding="utf-8")
    print(f"Candidates: {len(pairs_df)} saved to {out_pairs_path}")
    return out_pairs_path


if __name__ == "__main__":
    build_candidates(
        "../data/tokenized_dataset/affiliationstrings_ids_with_tokens.csv",
        "../data/processed/candidate_pairs.csv",
        k=20,
        min_sim=None,
        undirected=True,
    )
