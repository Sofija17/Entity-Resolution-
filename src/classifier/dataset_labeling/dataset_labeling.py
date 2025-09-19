from pathlib import Path
import pandas as pd

def attach_labels(features_csv: Path, mapping_csv: Path, out_csv: Path | None = None) -> Path:
    # Load features and mapping
    df = pd.read_csv(features_csv)
    gold = pd.read_csv(mapping_csv, header=None, names=["src","cand"])

    # Normalize order so (a,b) == (b,a)
    gold["pair_key"] = gold.apply(lambda r: tuple(sorted((r["src"], r["cand"]))), axis=1)
    gold_set = set(gold["pair_key"])

    # Apply to features
    df["pair_key"] = df.apply(lambda r: tuple(sorted((r["src_id"], r["cand_id"]))), axis=1)
    df["label"] = df["pair_key"].map(lambda k: 1 if k in gold_set else 0)

    df = df.drop(columns=["pair_key"])

    if out_csv is None:
        out_csv = features_csv.with_name(features_csv.stem + "_labeled.csv")
    df.to_csv(out_csv, index=False)
    print(f"[labels] {out_csv} written with {df['label'].sum()} positives out of {len(df)} rows")
    return out_csv


# Example usage:
attach_labels(
    features_csv=Path("../../../data/er_blocking_candidates_k20_features.csv"),
    mapping_csv=Path("../../../data/affiliationstrings_mapping.csv")
)
