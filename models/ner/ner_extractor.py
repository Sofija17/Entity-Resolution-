# # models/ner/ner_extractor.py
# import logging
# from dataclasses import dataclass
# from typing import List, Dict, Any, Optional
#
# import pandas as pd
# import spacy
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class ExtractedEntity:
#     text: str
#     label: str
#     start: int
#     end: int
#     confidence: Optional[float] = None
#
#
# class NERExtractor:
#     """Named Entity Recognition extractor supporting spaCy and HF transformers."""
#
#     def __init__(self, model_type: str = "spacy", model_name: str = "en_core_web_trf"):
#         self.model_type = model_type
#         self.model_name = model_name
#         self.model = None
#         self.tokenizer = None
#         self.ner_pipeline = None
#         self._load_model()
#
#     # ---------- Loading ----------
#     def _load_model(self) -> None:
#         try:
#             if self.model_type == "spacy":
#                 self._load_spacy_model()
#             elif self.model_type == "transformers":
#                 self._load_transformers_model()
#             else:
#                 raise ValueError(f"Unsupported model type: {self.model_type}")
#             logger.info(f"Successfully loaded {self.model_type} model: {self.model_name}")
#         except Exception as e:
#             logger.error(f"Failed to load model {self.model_name}: {e}")
#             raise
#
#     def _load_spacy_model(self) -> None:
#         try:
#             self.model = spacy.load(self.model_name)
#         except OSError as e:
#             raise OSError(
#                 f"spaCy model '{self.model_name}' not found. "
#                 f"Install with: python -m spacy download {self.model_name}"
#             ) from e
#
#     def _load_transformers_model(self) -> None:
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
#         self.ner_pipeline = pipeline(
#             "ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple"
#         )
#
#     # ---------- Inference ----------
#     def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[ExtractedEntity]:
#         if not text or not str(text).strip():
#             return []
#         if self.model_type == "spacy":
#             return self._extract_spacy_entities(text, entity_types)
#         return self._extract_transformers_entities(text, entity_types)
#
#     def _extract_spacy_entities(self, text: str, entity_types: Optional[List[str]]) -> List[ExtractedEntity]:
#         doc = self.model(text)
#         out: List[ExtractedEntity] = []
#         for ent in doc.ents:
#             if entity_types is None or ent.label_ in entity_types:
#                 out.append(ExtractedEntity(ent.text, ent.label_, ent.start_char, ent.end_char))
#         return out
#
#     def _extract_transformers_entities(self, text: str, entity_types: Optional[List[str]]) -> List[ExtractedEntity]:
#         results = self.ner_pipeline(text)
#         out: List[ExtractedEntity] = []
#         for r in results:
#             lbl = r["entity_group"]
#             if entity_types is None or lbl in entity_types:
#                 out.append(ExtractedEntity(r["word"], lbl, r["start"], r["end"], confidence=r.get("score")))
#         return out
#
#     # ---------- Batch over affiliations DataFrame ----------
#     def process_affiliations_dataset(
#         self,
#         data: pd.DataFrame,
#         affiliation_column: str = "affil1",
#         id_column: str = "id1",
#         batch_size: int = 100,
#         entity_types: Optional[List[str]] = None,
#     ) -> List[Dict[str, Any]]:
#         """
#         Runs NER over a DataFrame of affiliation strings and returns a list of dicts:
#         {'id', 'affiliation', 'entities', 'total_entities'}
#         """
#         results: List[Dict[str, Any]] = []
#         affiliations = data[affiliation_column].fillna("").astype(str).tolist()
#         ids = data[id_column].tolist() if id_column in data.columns else list(range(len(affiliations)))
#
#         logger.info(f"Processing {len(affiliations)} affiliations for entity extraction...")
#         for i in range(0, len(affiliations), batch_size):
#             batch_affiliations = affiliations[i : i + batch_size]
#             batch_ids = ids[i : i + batch_size]
#
#             for rec_id, aff in zip(batch_ids, batch_affiliations):
#                 try:
#                     ents = self.extract_entities(aff, entity_types)
#                     results.append(
#                         {
#                             "id": rec_id,
#                             "affiliation": aff,
#                             "entities": ents,  # canonical key
#                             "total_entities": len(ents),
#                         }
#                     )
#                 except Exception as e:
#                     logger.warning(f"Error processing affiliation ID {rec_id}: {e}")
#                     results.append(
#                         {
#                             "id": rec_id,
#                             "affiliation": aff,
#                             "entities": [],
#                             "total_entities": 0,
#                             "error": str(e),
#                         }
#                     )
#
#             if (i + batch_size) % 1000 == 0:
#                 logger.info(f"Processed {min(i + batch_size, len(affiliations))} affiliations...")
#
#         logger.info(f"Affiliation entity extraction complete. Processed {len(results)} affiliations.")
#         return results
# models/ner/ner_extractor.py
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
    Named Entity Recognition extractor with:
      - primary: spaCy (default: xx_ent_wiki_sm, CPU fast)
      - backoff: HF transformers (default: dslim/bert-base-NER) used ONLY when primary finds 0 entities
    """

    def __init__(
        self,
        model_type: str = "spacy",
        model_name: str = "xx_ent_wiki_sm",                 # <— default to multilingual small
        transformers_backoff_model: Optional[str] = "dslim/bert-base-NER",  # <— CPU-friendly fallback
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.model = None  # spaCy Language OR HF model (if model_type == transformers)
        self.tokenizer = None
        self.ner_pipeline = None  # HF pipeline for primary when transformers
        self.backoff_pipeline = None  # HF pipeline used as backoff when primary is spaCy
        self.transformers_backoff_model = transformers_backoff_model
        self._load_model()

    # ---------- Loading ----------
    def _load_model(self) -> None:
        try:
            if self.model_type == "spacy":
                self._load_spacy_model()
            elif self.model_type == "transformers":
                self._load_transformers_model(self.model_name)  # primary is HF
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            logger.info(f"Successfully loaded {self.model_type} model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _load_spacy_model(self) -> None:
        try:
            self.model = spacy.load(self.model_name)
        except OSError as e:
            raise OSError(
                f"spaCy model '{self.model_name}' not found. "
                f"Install with: python -m spacy download {self.model_name}"
            ) from e

    def _load_transformers_model(self, hf_model_name: str) -> None:
        # lazy import transformers to keep spaCy-only environments lean
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(hf_model_name)
        # device=-1 forces CPU
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

    def _ensure_backoff(self) -> None:
        """Load the backoff HF pipeline once (CPU) if configured."""
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
            device=-1,  # CPU
        )

    # ---------- Inference ----------
    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[ExtractedEntity]:
        if not text or not str(text).strip():
            return []

        if self.model_type == "spacy":
            ents = self._extract_spacy_entities(text, entity_types)
            # Backoff ONLY if spaCy finds nothing
            if not ents and self.transformers_backoff_model:
                self._ensure_backoff()
                ents = self._extract_hf_pipeline(text, entity_types, use_backoff=True)
            return ents

        # primary is transformers
        return self._extract_hf_pipeline(text, entity_types, use_backoff=False)

    def _extract_spacy_entities(self, text: str, entity_types: Optional[List[str]]) -> List[ExtractedEntity]:
        doc = self.model(text)
        out: List[ExtractedEntity] = []
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                out.append(ExtractedEntity(ent.text, ent.label_, ent.start_char, ent.end_char))
        return out

    def _extract_hf_pipeline(
        self, text: str, entity_types: Optional[List[str]], use_backoff: bool
    ) -> List[ExtractedEntity]:
        pipe = self.backoff_pipeline if use_backoff else self.ner_pipeline
        if pipe is None:
            # if someone set model_type="transformers" but didn't load, ensure loading
            if use_backoff:
                self._ensure_backoff()
                pipe = self.backoff_pipeline
            else:
                self._load_transformers_model(self.model_name)
                pipe = self.ner_pipeline

        results = pipe(text)
        out: List[ExtractedEntity] = []
        for r in results:
            lbl = r.get("entity_group") or r.get("entity")
            if entity_types is None or lbl in entity_types:
                out.append(
                    ExtractedEntity(
                        text=r["word"],
                        label=lbl,
                        start=r["start"],
                        end=r["end"],
                        confidence=r.get("score"),
                    )
                )
        return out

    # ---------- Batch over affiliations DataFrame ----------
    def process_affiliations_dataset(
        self,
        data: pd.DataFrame,
        affiliation_column: str = "affil1",
        id_column: str = "id1",
        batch_size: int = 100,
        entity_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs NER over a DataFrame of affiliation strings and returns a list of dicts:
        {'id', 'affiliation', 'entities', 'total_entities'}
        """
        results: List[Dict[str, Any]] = []
        affiliations = data[affiliation_column].fillna("").astype(str).tolist()
        ids = data[id_column].tolist() if id_column in data.columns else list(range(len(affiliations)))

        logger.info(f"Processing {len(affiliations)} affiliations for entity extraction...")
        for i in range(0, len(affiliations), batch_size):
            batch_affiliations = affiliations[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            for rec_id, aff in zip(batch_ids, batch_affiliations):
                try:
                    ents = self.extract_entities(aff, entity_types)
                    results.append(
                        {
                            "id": rec_id,
                            "affiliation": aff,
                            "entities": ents,  # canonical key
                            "total_entities": len(ents),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing affiliation ID {rec_id}: {e}")
                    results.append(
                        {
                            "id": rec_id,
                            "affiliation": aff,
                            "entities": [],
                            "total_entities": 0,
                            "error": str(e),
                        }
                    )

            if (i + batch_size) % 1000 == 0:
                logger.info(f"Processed {min(i + batch_size, len(affiliations))} affiliations...")

        logger.info(f"Affiliation entity extraction complete. Processed {len(results)} affiliations.")
        return results
