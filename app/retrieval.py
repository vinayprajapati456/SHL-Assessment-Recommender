"""Retrieval engine for SHL assessments.

Uses a hybrid approach combining:
1. TF-IDF cosine similarity for semantic matching
2. Keyword boosting for exact technology/domain term matches
3. Metadata filtering by job level and test type
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import CatalogItem

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hybrid TF-IDF + keyword retrieval over the SHL catalog."""

    def __init__(self, catalog: list[CatalogItem]) -> None:
        self.catalog = catalog
        self._name_index: dict[str, int] = {}  # lowercase name -> index
        self._build_indices()

    def _build_indices(self) -> None:
        """Build TF-IDF index and keyword lookup structures."""
        # TF-IDF over search_text
        corpus = [item.search_text for item in self.catalog]
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Name-based lookup (lowercase, normalized)
        for i, item in enumerate(self.catalog):
            normalized = item.name.lower().strip()
            self._name_index[normalized] = i

        # Build keyword -> item indices map for technology terms
        self._keyword_index: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(self.catalog):
            # Index by words in name and description
            text = f"{item.name} {item.description}".lower()
            # Extract meaningful tokens
            tokens = set(re.findall(r"\b[a-z][a-z0-9#+.]{1,}\b", text))
            for token in tokens:
                self._keyword_index[token].append(i)

        logger.info(
            "Built indices: %d items, %d keyword entries",
            len(self.catalog),
            len(self._keyword_index),
        )

    def _extract_query_terms(self, query: str) -> list[str]:
        """Extract meaningful search terms from the query."""
        # Clean and tokenize
        query_lower = query.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9#+.]{1,}\b", query_lower)
        return tokens

    # Core assessments that are commonly recommended across many roles
    CORE_ASSESSMENTS = [
        "Occupational Personality Questionnaire OPQ32r",
        "SHL Verify Interactive G+",
    ]

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        job_level_filter: str | None = None,
        test_type_filter: str | None = None,
    ) -> list[tuple[CatalogItem, float]]:
        """Retrieve the most relevant catalog items for a query.

        Uses a multi-phase approach:
        1. Full-query TF-IDF similarity
        2. Per-term TF-IDF searches (captures each technology individually)
        3. Keyword boosting for exact term matches
        4. Name matching for directly mentioned assessments
        5. Core assessment injection (OPQ32r, Verify G+)
        6. Metadata boosting for job level / test type filters

        Args:
            query: Natural language query combining user intent
            top_k: Number of candidates to return
            job_level_filter: Optional job level to boost (e.g. "Graduate")
            test_type_filter: Optional test type to boost (e.g. "K", "P")

        Returns:
            List of (CatalogItem, score) tuples, sorted by relevance.
        """
        n = len(self.catalog)
        scores = np.zeros(n)

        # --- Phase 1: Full-query TF-IDF similarity ---
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        scores += tfidf_scores * 1.0

        # --- Phase 2: Per-term TF-IDF searches ---
        # This ensures each individual technology/skill gets its best match
        query_terms = self._extract_query_terms(query)
        # Also extract multi-word tech terms
        tech_phrases = self._extract_tech_phrases(query)
        search_terms = list(set(query_terms + tech_phrases))

        for term in search_terms:
            if len(term) >= 2:
                term_vec = self.vectorizer.transform([term])
                term_scores = cosine_similarity(
                    term_vec, self.tfidf_matrix
                ).flatten()
                # Add a fraction of per-term scores
                scores += term_scores * 0.3

        # --- Phase 3: Keyword boosting ---
        keyword_boost = np.zeros(n)
        for term in query_terms:
            if term in self._keyword_index:
                for idx in self._keyword_index[term]:
                    keyword_boost[idx] += 0.15
            # Partial matches for technology names
            for key, indices in self._keyword_index.items():
                if len(term) > 2 and (term in key or key in term):
                    for idx in indices:
                        keyword_boost[idx] += 0.05
        scores += keyword_boost

        # --- Phase 4: Name matching ---
        query_lower = query.lower()
        for i, item in enumerate(self.catalog):
            name_lower = item.name.lower()
            if name_lower in query_lower:
                scores[i] += 0.5
            elif any(
                term in name_lower
                for term in query_terms
                if len(term) > 3
            ):
                scores[i] += 0.2

        # --- Phase 5: Core assessment injection ---
        # Always include OPQ32r and Verify G+ with a small baseline score
        # so the LLM can consider them for personality/cognitive dimensions
        for core_name in self.CORE_ASSESSMENTS:
            core_lower = core_name.lower()
            if core_lower in self._name_index:
                idx = self._name_index[core_lower]
                scores[idx] = max(scores[idx], 0.15)

        # --- Phase 6: Metadata boosting ---
        if job_level_filter:
            filter_lower = job_level_filter.lower()
            for i, item in enumerate(self.catalog):
                for level in item.job_levels:
                    if filter_lower in level.lower():
                        scores[i] += 0.1

        if test_type_filter:
            for i, item in enumerate(self.catalog):
                if test_type_filter in item.test_type:
                    scores[i] += 0.1

        # Rank and return top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.catalog[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.01
        ]

        # Ensure core assessments are always in results for the LLM to consider
        result_names = {item.name for item, _ in results}
        for core_name in self.CORE_ASSESSMENTS:
            if core_name not in result_names:
                core_lower = core_name.lower()
                if core_lower in self._name_index:
                    idx = self._name_index[core_lower]
                    results.append((self.catalog[idx], 0.10))

        logger.debug("Retrieved %d items for query: %s", len(results), query[:80])
        return results

    @staticmethod
    def _extract_tech_phrases(query: str) -> list[str]:
        """Extract multi-word technology phrases from the query."""
        phrases = []
        query_lower = query.lower()

        # Common multi-word tech terms
        tech_patterns = [
            r"core\s+java", r"rest\s*(?:ful)?\s*(?:api|web\s*services?)?",
            r"sql\s*(?:server)?", r"spring\s*(?:boot|framework)?",
            r"angular\s*\d*", r"react\s*(?:js)?", r"node\s*(?:js)?",
            r"amazon\s+web\s+services|aws", r"docker", r"kubernetes",
            r"machine\s+learning", r"data\s+(?:science|analytics)",
            r"microsoft\s+(?:excel|word|office|sql)",
            r"contact\s+cent(?:er|re)", r"customer\s+service",
            r"(?:spoken?\s+)?english", r"financial\s+accounting",
            r"situational\s+judg(?:e?ment)", r"personality",
            r"cognitive|reasoning|numerical|verbal|inductive|deductive",
            r"leadership", r"safety|dependability",
            r"hipaa", r"medical\s+terminology",
        ]
        for pattern in tech_patterns:
            matches = re.findall(pattern, query_lower)
            phrases.extend(matches)

        return phrases

    def get_by_name(self, name: str) -> CatalogItem | None:
        """Look up a catalog item by exact or fuzzy name match."""
        name_lower = name.lower().strip()

        # Exact match
        if name_lower in self._name_index:
            return self.catalog[self._name_index[name_lower]]

        # Fuzzy: check if query is substring of any name or vice versa
        best_match = None
        best_overlap = 0
        for stored_name, idx in self._name_index.items():
            if name_lower in stored_name or stored_name in name_lower:
                overlap = min(len(name_lower), len(stored_name))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = self.catalog[idx]

        return best_match

    def get_by_names(self, names: list[str]) -> list[CatalogItem]:
        """Look up multiple catalog items by name."""
        results = []
        for name in names:
            item = self.get_by_name(name)
            if item:
                results.append(item)
        return results

    def validate_url(self, url: str) -> bool:
        """Check if a URL exists in the catalog."""
        return any(item.link == url for item in self.catalog)

    def get_all_items(self) -> list[CatalogItem]:
        """Return the full catalog."""
        return self.catalog
