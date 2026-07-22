from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.search_engine import SearchResult


MAXIMUM_QUERY_RESULTS = 200

ABSOLUTE_SEMANTIC_FLOOR = 0.30
RELATIVE_SEMANTIC_FLOOR = 0.45

ABSOLUTE_KEYWORD_FLOOR = 0.08
RELATIVE_KEYWORD_FLOOR = 0.25


@dataclass(frozen=True)
class ResultRelevanceSummary:
    """Summary of result-level relevance filtering."""

    results: tuple[SearchResult, ...]
    semantic_floor: float
    keyword_floor: float
    examined_count: int
    qualifying_count: int
    returned_count: int


def safe_score(
    result: SearchResult,
    attribute_name: str,
) -> float:
    """Return a result score as a safe float."""

    value = getattr(
        result,
        attribute_name,
        0.0,
    )

    if value is None:
        return 0.0

    return float(value)


def strongest_score(
    results: Sequence[SearchResult],
    attribute_name: str,
) -> float:
    """Return the strongest selected score."""

    if not results:
        return 0.0

    return max(
        safe_score(
            result,
            attribute_name,
        )
        for result in results
    )


def calculate_result_floors(
    results: Sequence[SearchResult],
) -> tuple[float, float]:
    """Calculate dynamic semantic and keyword floors."""

    strongest_semantic_score = strongest_score(
        results=results,
        attribute_name="semantic_score",
    )

    strongest_keyword_score = strongest_score(
        results=results,
        attribute_name="keyword_score",
    )

    semantic_floor = max(
        ABSOLUTE_SEMANTIC_FLOOR,
        (
            strongest_semantic_score
            * RELATIVE_SEMANTIC_FLOOR
        ),
    )

    keyword_floor = max(
        ABSOLUTE_KEYWORD_FLOOR,
        (
            strongest_keyword_score
            * RELATIVE_KEYWORD_FLOOR
        ),
    )

    return (
        semantic_floor,
        keyword_floor,
    )


def filter_relevant_results(
    results: Sequence[SearchResult],
    maximum_results: int = MAXIMUM_QUERY_RESULTS,
) -> ResultRelevanceSummary:
    """
    Remove the weak result tail from a query search.

    The original hybrid ranking order is preserved. A result
    qualifies when it has either sufficient semantic evidence
    or sufficient keyword evidence.
    """

    if maximum_results <= 0:
        raise ValueError(
            "maximum_results must be greater than zero."
        )

    result_tuple = tuple(results)

    if not result_tuple:
        return ResultRelevanceSummary(
            results=(),
            semantic_floor=(
                ABSOLUTE_SEMANTIC_FLOOR
            ),
            keyword_floor=(
                ABSOLUTE_KEYWORD_FLOOR
            ),
            examined_count=0,
            qualifying_count=0,
            returned_count=0,
        )

    (
        semantic_floor,
        keyword_floor,
    ) = calculate_result_floors(
        result_tuple
    )

    qualifying_results = tuple(
        result
        for result in result_tuple
        if (
            safe_score(
                result,
                "semantic_score",
            )
            >= semantic_floor
            or safe_score(
                result,
                "keyword_score",
            )
            >= keyword_floor
        )
    )

    returned_results = qualifying_results[
        :maximum_results
    ]

    return ResultRelevanceSummary(
        results=returned_results,
        semantic_floor=semantic_floor,
        keyword_floor=keyword_floor,
        examined_count=len(result_tuple),
        qualifying_count=len(
            qualifying_results
        ),
        returned_count=len(
            returned_results
        ),
    )