# scripts/run_affiliations_ner.py
import sys
from pathlib import Path
import pandas as pd

# Add project src/ to path
sys.path.append(str((Path(__file__).parent.parent / "src").resolve()))

from models.ner.ner_extractor import NERExtractor
from models.ner.token_processor import TokenProcessor


def process_csv_and_write_tokens(csv_path: str, model_type: str = "spacy", model_name: str = "en_core_web_trf"):
    csv_path = Path(csv_path).resolve()

    # Load original CSV
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Dataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample affiliation: {df['affil1'].iloc[0] if len(df) > 0 else 'None'}")

    # Init extractor
    # primary: spaCy xx_ent_wiki_sm (fast)
    # backoff: dslim/bert-base-NER (only when spaCy finds 0 entities)
    extractor = NERExtractor(
        model_type="spacy",
        model_name="en_core_web_trf",
        transformers_backoff_model="dslim/bert-base-NER",
    )

    # Run NER
    results = extractor.process_affiliations_dataset(
        data=df,
        affiliation_column="affil1",
        id_column="id1",         # column already exists per your header
        batch_size=100,
        entity_types=None,       # all entity types
    )

    # Build tokens DataFrame (both labeled + plain) and merge into original
    tokens_df = TokenProcessor.results_to_tokens_df(results, id_column="id1")
    merged = TokenProcessor.merge_tokens_into_original_csv(df, tokens_df, id_column="id1")

    # Output path: same folder as input, with suffix
    output_path = csv_path.with_name(csv_path.stem + "_with_tokens.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✅ Updated CSV written (with 'affil_tokens_labeled' and 'affil_tokens'): {output_path}")


if __name__ == "__main__":
    process_csv_and_write_tokens("../data/affiliationstrings_ids.csv")
