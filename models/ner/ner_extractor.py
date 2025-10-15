import pandas as pd
import spacy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = None

# Using a NER model to extract entities from an entry
class NERExtractor:
    # Initializing main Spacy transformer model, as well as a backup bert-based NER model
    def __init__(self, model_type: str = "spacy", model_name: str = "en_core_web_trf",
                 transformers_backoff_model: Optional[str] = "dslim/bert-base-NER",):
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.backoff_pipeline = None
        self.transformers_backoff_model = transformers_backoff_model
        self.load_model()

    # Loading main Spacy transformer model
    def load_spacy_model(self):
        try:
            self.model = spacy.load(self.model_name)
        except OSError:
            raise RuntimeError(f"spaCy model '{self.model_name}' not found.")

    # Loading backoff bert-based NER model
    def load_backoff_model(self):
        if not self.transformers_backoff_model or self.backoff_pipeline is not None:
            return
        from transformers import pipeline
        self.backoff_pipeline = pipeline(
            task="token-classification",
            model=self.transformers_backoff_model,
            aggregation_strategy="simple",
            device=-1,
        )

    # Loading defined models
    def load_model(self) -> None:
        try:
            if self.model_type == "spacy":
                self.load_spacy_model()
            elif self.model_type == "transformers":
                self.load_backoff_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            logger.info(f"Successfully loaded {self.model_type} model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    # Extract entities with Spacy model
    def _extract_spacy_entities(self, text: str, entity_types: Optional[List[str]]) -> List[ExtractedEntity]:
        doc = self.model(text)
        out: List[ExtractedEntity] = []
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                out.append(ExtractedEntity(ent.text, ent.label_, ent.start_char, ent.end_char))
        return out

    # Extract entities with bert-based Hugging Face NER
    def _extract_hf_pipeline(self, text: str, entity_types: Optional[List[str]]) -> List[ExtractedEntity]:
        if self.backoff_pipeline is None:
            self.load_backoff_model()
        pipe = self.backoff_pipeline
        if pipe is None:
            raise RuntimeError("Backoff pipeline not initialized.")

        results = pipe(text)
        out: List[ExtractedEntity] = []
        for result in results:
            label = result.get("entity_group") or result.get("entity")
            if entity_types is None or label in entity_types:
                out.append(ExtractedEntity(text=result.get("word", text[result["start"]:result["end"]]),
                        label=label, start=result["start"], end=result["end"], confidence=result.get("score"))
                )
        return out

    # Calls on extract functions for Spacy or backoff Bert model
    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[ExtractedEntity]:
        if not text or not str(text).strip():
            return []

        # Try extracting entities using Spacy
        try:
            entities = self._extract_spacy_entities(text, entity_types)
        except Exception as error:
            logger.exception("Spacy extraction failed, attempting backoff model: %s", error)
            entities = []

        # If no entities found, use the backoff transformer model
        if not entities and self.transformers_backoff_model:
            self.load_backoff_model()
            entities = self._extract_hf_pipeline(text, entity_types)
        return entities

    # Executing a NER model on the original dataset and returning a list of dictionaries of form:
    # {'id', 'affiliation', 'entities'}
    def process_dataset(self, data: pd.DataFrame, affiliation_column: str,
                        id_column: str, batch_size: int, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []
        affiliations = data[affiliation_column].fillna("").astype(str).tolist()
        ids = data[id_column].tolist() if id_column in data.columns else list(range(len(affiliations)))

        logger.info(f"Processing {len(affiliations)} rows for entity extraction...")
        for i in range(0, len(affiliations), batch_size):
            batch_affiliations = affiliations[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]

            for record_id, affiliation in zip(batch_ids, batch_affiliations):
                try:
                    ents = self.extract_entities(affiliation, entity_types)
                    results.append({
                        "id": record_id,
                        "affiliation": affiliation,
                        "entities": ents,
                    })
                except Exception as e:
                    logger.warning(f"Error processing row id {record_id}: {e}")
                    results.append({
                        "id": record_id,
                        "affiliation": affiliation,
                        "entities": [],
                        "error": str(e),
                    })

            if (i + batch_size) % 1000 == 0:
                logger.info(f"Processed {min(i + batch_size, len(affiliations))} rows...")

        logger.info(f"Entity extraction complete. Processed {len(results)} rows.")
        return results
