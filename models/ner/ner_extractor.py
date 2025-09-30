import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = None


class NERExtractor:
    """
    Named Entity Recognition екстрактор со:
      - главен модел: spaCy (en_core_web_trf)
      - backoff модел: HF transformers (dslim/bert-base-NER), се користи само кога примарниот модел резултира со 0 ентитети при екстракција на исти
    """

    def __init__(
        self,
        model_type: str = "spacy",
        model_name: str = "en_core_web_trf",
        transformers_backoff_model: Optional[str] = "dslim/bert-base-NER",
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        self.backoff_pipeline = None
        self.transformers_backoff_model = transformers_backoff_model
        self.load_model()

    def load_model(self) -> None:
        try:
            if self.model_type == "spacy":
                self.load_spacy_model()
            elif self.model_type == "transformers":
                self.load_transformer_model(self.model_name)  # primary is HF
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            logger.info(f"Successfully loaded {self.model_type} model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def load_spacy_model(self) -> None:
        try:
            self.model = spacy.load(self.model_name)
        except OSError as e:
            raise OSError(
                f"spaCy model '{self.model_name}' not found. "
                f"Install with: python -m spacy download {self.model_name}"
            ) from e

    def load_backoff_model(self) -> None:
        if not self.transformers_backoff_model or self.backoff_pipeline is not None:
            return
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        logger.info(f"Loading backoff NER model: {self.transformers_backoff_model}")
        tok = AutoTokenizer.from_pretrained(self.transformers_backoff_model)
        mdl = AutoModelForTokenClassification.from_pretrained(self.transformers_backoff_model)
        self.backoff_pipeline = pipeline(
            "ner",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple",
            device=-1,
        )

    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[ExtractedEntity]:
        if not text or not str(text).strip():
            return []
        # spaCy first (примарен)
        try:
            ents = self._extract_spacy_entities(text, entity_types)
        except Exception as e:
            logger.exception("spaCy extraction failed; will try backoff: %s", e)
            ents = []
        # HF fallback
        if (not ents) and self.transformers_backoff_model:
            self.load_backoff_model()
            ents = self._extract_hf_pipeline(text, entity_types)
        return ents

    def _extract_spacy_entities(self, text: str, entity_types: Optional[List[str]]) -> List[ExtractedEntity]:
        doc = self.model(text)
        out: List[ExtractedEntity] = []
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                out.append(ExtractedEntity(ent.text, ent.label_, ent.start_char, ent.end_char))
        return out

    def _extract_hf_pipeline(self, text: str, entity_types: Optional[List[str]]) -> List[ExtractedEntity]:
        # Ensure lazy backoff pipeline is loaded
        if self.backoff_pipeline is None:
            self.load_backoff_model()
        pipe = self.backoff_pipeline
        if pipe is None:
            raise RuntimeError("Backoff pipeline not initialized; check transformers_backoff_model.")

        results = pipe(text)
        out: List[ExtractedEntity] = []
        for r in results:
            lbl = r.get("entity_group") or r.get("entity")
            if entity_types is None or lbl in entity_types:
                out.append(
                    ExtractedEntity(
                        text=r.get("word", text[r["start"]:r["end"]]),
                        label=lbl,
                        start=r["start"],
                        end=r["end"],
                        confidence=r.get("score"),
                    )
                )
        return out

    def process_affiliations_dataset(
            self,
            data: pd.DataFrame,
            affiliation_column: str = "affil1",
            id_column: str = "id1",
            batch_size: int = 100,
            entity_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Извршување на NER врз affiliations датасет и враќање на листа од dictionaries:
        {'id', 'affiliation', 'entities'}
        """
        results: List[Dict[str, Any]] = []
        affiliations = data[affiliation_column].fillna("").astype(str).tolist()
        ids = data[id_column].tolist() if id_column in data.columns else list(range(len(affiliations)))

        logger.info(f"Processing {len(affiliations)} affiliations for entity extraction...")
        for i in range(0, len(affiliations), batch_size):
            batch_affiliations = affiliations[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]

            for rec_id, aff in zip(batch_ids, batch_affiliations):
                try:
                    ents = self.extract_entities(aff, entity_types)
                    results.append(
                        {
                            "id": rec_id,
                            "affiliation": aff,
                            "entities": ents,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing affiliation ID {rec_id}: {e}")
                    results.append(
                        {
                            "id": rec_id,
                            "affiliation": aff,
                            "entities": [],
                            "error": str(e),
                        }
                    )

            if (i + batch_size) % 1000 == 0:
                logger.info(f"Processed {min(i + batch_size, len(affiliations))} affiliations...")

        logger.info(f"Affiliation entity extraction complete. Processed {len(results)} affiliations.")
        return results
