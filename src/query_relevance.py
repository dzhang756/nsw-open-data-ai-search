from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.search_engine import SearchResult

MINIMUM_SEMANTIC_SCORE = 0.32
MINIMUM_KEYWORD_SCORE = 0.10


@dataclass(frozen=True)
class QueryRelevanceDecision:
    """Decision about whether search results are meaningful."""

    is_relevant: bool
    strongest_semantic_score: float
    strongest_keyword_score: float
    reason: str


def has_searchable_characters(
    query: str,
) -> bool:
    """Return whether the query contains a letter or number."""

    return any(
        character.isalnum()
        for character in query
    )


def assess_query_relevance(
    query: str,
    results: Sequence[SearchResult],
) -> QueryRelevanceDecision:
    """
    Decide whether a query has enough retrieval evidence.

    A query is rejected only when both semantic and keyword
    evidence are weak. This allows natural-language and
    moderately misspelled searches to continue working.
    """

    cleaned_query = " ".join(
        query.split()
    )

    if not cleaned_query:
        return QueryRelevanceDecision(
            is_relevant=False,
            strongest_semantic_score=0.0,
            strongest_keyword_score=0.0,
            reason="The query is blank.",
        )

    if not has_searchable_characters(
        cleaned_query
    ):
        return QueryRelevanceDecision(
            is_relevant=False,
            strongest_semantic_score=0.0,
            strongest_keyword_score=0.0,
            reason=(
                "The query contains no searchable "
                "letters or numbers."
            ),
        )

    if not results:
        return QueryRelevanceDecision(
            is_relevant=False,
            strongest_semantic_score=0.0,
            strongest_keyword_score=0.0,
            reason="The search returned no datasets.",
        )

    strongest_semantic_score = max(
        float(result.semantic_score)
        for result in results
    )

    strongest_keyword_score = max(
        float(result.keyword_score)
        for result in results
    )

    weak_semantic_evidence = (
        strongest_semantic_score
        < MINIMUM_SEMANTIC_SCORE
    )

    weak_keyword_evidence = (
        strongest_keyword_score
        < MINIMUM_KEYWORD_SCORE
    )

    if (
        weak_semantic_evidence
        and weak_keyword_evidence
    ):
        return QueryRelevanceDecision(
            is_relevant=False,
            strongest_semantic_score=(
                strongest_semantic_score
            ),
            strongest_keyword_score=(
                strongest_keyword_score
            ),
            reason=(
                "The strongest results had insufficient "
                "semantic and keyword evidence."
            ),
        )

    return QueryRelevanceDecision(
        is_relevant=True,
        strongest_semantic_score=(
            strongest_semantic_score
        ),
        strongest_keyword_score=(
            strongest_keyword_score
        ),
        reason=(
            "The query has sufficient retrieval evidence."
        ),
    )