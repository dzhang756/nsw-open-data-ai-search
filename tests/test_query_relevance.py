from __future__ import annotations

import unittest

from src.query_relevance import (
    assess_query_relevance,
)
from src.search_engine import SearchResult


def make_result(
    semantic_score: float,
    keyword_score: float,
) -> SearchResult:
    """Create a minimal result for relevance tests."""

    return SearchResult(
        row_index=0,
        dataset_id="test-dataset",
        title="Test Dataset",
        description="Test description",
        organisation="Test Organisation",
        resource_formats=("CSV",),
        metadata_modified=(
            "2026-07-22T00:00:00"
        ),
        dataset_url=(
            "https://data.nsw.gov.au/"
            "data/dataset/test-dataset"
        ),
        hybrid_score=0.01,
        semantic_score=semantic_score,
        semantic_rank=1,
        keyword_score=keyword_score,
        keyword_rank=1,
    )


class QueryRelevanceTests(unittest.TestCase):
    """Tests for the calibrated no-match rule."""

    def test_symbol_only_query_is_rejected(
        self,
    ) -> None:
        """Symbols without letters or numbers are invalid."""

        decision = assess_query_relevance(
            query="@@@ ### !!!",
            results=(
                make_result(
                    semantic_score=0.50,
                    keyword_score=0.20,
                ),
            ),
        )

        self.assertFalse(
            decision.is_relevant
        )

    def test_random_letters_are_rejected(
        self,
    ) -> None:
        """Calibrated nonsense scores should be rejected."""

        decision = assess_query_relevance(
            query="qqq 9182 zzz",
            results=(
                make_result(
                    semantic_score=0.3033,
                    keyword_score=0.0,
                ),
            ),
        )

        self.assertFalse(
            decision.is_relevant
        )

    def test_incoherent_weak_query_is_rejected(
        self,
    ) -> None:
        """Weak evidence from real but unrelated words is insufficient."""

        decision = assess_query_relevance(
            query=(
                "banana asteroid hospital quantum"
            ),
            results=(
                make_result(
                    semantic_score=0.2499,
                    keyword_score=0.0771,
                ),
            ),
        )

        self.assertFalse(
            decision.is_relevant
        )

    def test_valid_query_is_accepted(
        self,
    ) -> None:
        """A normal valid query should pass."""

        decision = assess_query_relevance(
            query="hospital waiting times",
            results=(
                make_result(
                    semantic_score=0.5242,
                    keyword_score=0.3816,
                ),
            ),
        )

        self.assertTrue(
            decision.is_relevant
        )

    def test_supported_misspelling_is_accepted(
        self,
    ) -> None:
        """Moderate misspellings with evidence should pass."""

        decision = assess_query_relevance(
            query=(
                "emergncy departmnt wait times"
            ),
            results=(
                make_result(
                    semantic_score=0.3620,
                    keyword_score=0.1390,
                ),
            ),
        )

        self.assertTrue(
            decision.is_relevant
        )

    def test_strong_semantic_only_match_is_accepted(
        self,
    ) -> None:
        """Strong semantic evidence can compensate for no keywords."""

        decision = assess_query_relevance(
            query="specialised dataset concept",
            results=(
                make_result(
                    semantic_score=0.40,
                    keyword_score=0.0,
                ),
            ),
        )

        self.assertTrue(
            decision.is_relevant
        )


if __name__ == "__main__":
    unittest.main()