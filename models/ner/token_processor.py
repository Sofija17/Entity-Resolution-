# models/ner/token_processor.py
import logging
from typing import List, Dict, Any
import pandas as pd
from models.ner.ner_extractor import ExtractedEntity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenProcessor:
    """Post-processing utilities for extracted tokens."""

    @staticmethod
    def to_tokens_string_labeled(entities: List[ExtractedEntity]) -> str:
        """
        Semicolon-separated, order-preserving, de-duplicated by (text,label),
        rendered as 'text<label>'.
        """
        seen = set()
        out = []
        for e in entities:
            text = (e.text or "").strip().rstrip(";:,")
            label = (e.label or "").strip()
            if not text:
                continue
            key = (text.lower(), label)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{text}<{label}>")
        return "; ".join(out)

    @staticmethod
    def to_tokens_string_plain(entities: List[ExtractedEntity]) -> str:
        """Semicolon-separated, order-preserving, de-duplicated by text only."""
        seen = set()
        out = []
        for e in entities:
            text = (e.text or "").strip().rstrip(";:,")
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return "; ".join(out)

    @staticmethod
    def results_to_tokens_df(results: List[Dict[str, Any]], id_column: str = "id1") -> pd.DataFrame:
        """
        Convert extractor results -> DataFrame with:
          - id1
          - affil_tokens_labeled (e.g., 'IBM<ORG>; San Jose<GPE>')
          - affil_tokens (plain)
          - total_entities
        """
        rows = []
        for item in results:
            ents = item.get("entities", [])
            rows.append(
                {
                    id_column: item["id"],
                    "affil_tokens_labeled": TokenProcessor.to_tokens_string_labeled(ents),
                    "affil_tokens": TokenProcessor.to_tokens_string_plain(ents),
                    "total_entities": item.get("total_entities", len(ents)),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def merge_tokens_into_original_csv(
        original_df: pd.DataFrame,
        tokens_df: pd.DataFrame,
        id_column: str = "id1",
    ) -> pd.DataFrame:
        """Left-merge tokens onto original DF using id_column (or index if missing)."""
        if id_column in original_df.columns and id_column in tokens_df.columns:
            merged = original_df.merge(tokens_df, on=id_column, how="left")
        else:
            tokens_df = tokens_df.rename(columns={id_column: "index"})
            merged = original_df.reset_index().merge(tokens_df, on="index", how="left").drop(columns=["index"])
        return merged
