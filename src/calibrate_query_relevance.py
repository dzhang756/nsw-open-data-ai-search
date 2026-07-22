from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.search_engine import (
    SearchConfig,
    SearchEngine,
    SearchResult,
)

OUTPUT_PATH = Path(
    "outputs/tables/query_relevance_calibration.csv"
)

TOP_RESULTS_TO_DISPLAY = 3


@dataclass(frozen=True)
class CalibrationQuery:
    """One query with its expected query type."""

    query_type: str
    query: str


DEFAULT_QUERIES = (
    # Clear, valid searches.
    CalibrationQuery(
        query_type="valid",
        query="road crash data",
    ),
    CalibrationQuery(
        query_type="valid",
        query="Sydney public transport passenger data",
    ),
    CalibrationQuery(
        query_type="valid",
        query="water quality monitoring",
    ),
    CalibrationQuery(
        query_type="valid",
        query="hospital waiting times",
    ),
    CalibrationQuery(
        query_type="valid",
        query="ambulance off stretcher times",
    ),
    CalibrationQuery(
        query_type="valid",
        query="koala habitat corridors",
    ),

    # Queries that are meaningful despite spelling errors.
    CalibrationQuery(
        query_type="misspelled",
        query="sdyney trafic volmue",
    ),
    CalibrationQuery(
        query_type="misspelled",
        query="emergncy departmnt wait times",
    ),
    CalibrationQuery(
        query_type="misspelled",
        query="bushfir risk managment plans",
    ),

    # Random character combinations.
    CalibrationQuery(
        query_type="nonsense",
        query="asdfghjkl",
    ),
    CalibrationQuery(
        query_type="nonsense",
        query="zxqvplm",
    ),
    CalibrationQuery(
        query_type="nonsense",
        query="qqq 9182 zzz",
    ),

    # Symbols or very weak query content.
    CalibrationQuery(
        query_type="symbols",
        query="@@@ ### !!!",
    ),
    CalibrationQuery(
        query_type="symbols",
        query="/// &&& ***",
    ),

    # Valid words without a coherent dataset meaning.
    CalibrationQuery(
        query_type="unrelated_words",
        query="banana asteroid hospital quantum",
    ),
    CalibrationQuery(
        query_type="unrelated_words",
        query="purple submarine taxation koala",
    ),
)


def result_title(
    result: SearchResult | None,
) -> str:
    """Return a safe title for an optional result."""

    if result is None:
        return ""

    return result.title


def result_score(
    result: SearchResult | None,
    score_name: str,
) -> float:
    """Return a score from an optional result."""

    if result is None:
        return 0.0

    return float(
        getattr(
            result,
            score_name,
            0.0,
        )
    )


def strongest_result(
    results: tuple[SearchResult, ...],
    score_name: str,
) -> SearchResult | None:
    """Return the result with the strongest selected score."""

    if not results:
        return None

    return max(
        results,
        key=lambda result: float(
            getattr(
                result,
                score_name,
                0.0,
            )
        ),
    )


def top_titles(
    results: tuple[SearchResult, ...],
    limit: int = TOP_RESULTS_TO_DISPLAY,
) -> str:
    """Return a compact list of leading hybrid results."""

    selected_results = results[:limit]

    return " | ".join(
        result.title
        for result in selected_results
    )


def analyse_query(
    engine: SearchEngine,
    calibration_query: CalibrationQuery,
    config: SearchConfig,
) -> dict[str, object]:
    """Run one query and collect relevance signals."""

    response = engine.search(
        query=calibration_query.query,
        config=config,
    )

    results = response.results

    top_hybrid_result = (
        results[0]
        if results
        else None
    )

    strongest_semantic_result = strongest_result(
        results=results,
        score_name="semantic_score",
    )

    strongest_keyword_result = strongest_result(
        results=results,
        score_name="keyword_score",
    )

    second_hybrid_score = (
        results[1].hybrid_score
        if len(results) >= 2
        else 0.0
    )

    top_hybrid_score = result_score(
        top_hybrid_result,
        "hybrid_score",
    )

    return {
        "query_type": calibration_query.query_type,
        "query": calibration_query.query,
        "results_returned": len(results),
        "recognised_keyword_features": (
            response.keyword_query_feature_count
        ),
        "top_hybrid_score": top_hybrid_score,
        "second_hybrid_score": second_hybrid_score,
        "hybrid_score_gap": (
            top_hybrid_score
            - second_hybrid_score
        ),
        "top_hybrid_title": result_title(
            top_hybrid_result
        ),
        "top_result_semantic_score": result_score(
            top_hybrid_result,
            "semantic_score",
        ),
        "top_result_keyword_score": result_score(
            top_hybrid_result,
            "keyword_score",
        ),
        "strongest_semantic_score": result_score(
            strongest_semantic_result,
            "semantic_score",
        ),
        "strongest_semantic_title": result_title(
            strongest_semantic_result
        ),
        "strongest_keyword_score": result_score(
            strongest_keyword_result,
            "keyword_score",
        ),
        "strongest_keyword_title": result_title(
            strongest_keyword_result
        ),
        "top_three_titles": top_titles(results),
    }


def print_analysis(
    analysis: dict[str, object],
) -> None:
    """Print one query analysis in a readable format."""

    print()
    print("=" * 80)
    print(
        f"Query type: {analysis['query_type']}"
    )
    print(
        f"Query: {analysis['query']!r}"
    )
    print("-" * 80)

    print(
        "Recognised keyword features: "
        f"{analysis['recognised_keyword_features']}"
    )

    print(
        "Top hybrid score: "
        f"{analysis['top_hybrid_score']:.6f}"
    )

    print(
        "Top result semantic score: "
        f"{analysis['top_result_semantic_score']:.4f}"
    )

    print(
        "Top result keyword score: "
        f"{analysis['top_result_keyword_score']:.4f}"
    )

    print(
        "Strongest semantic score: "
        f"{analysis['strongest_semantic_score']:.4f}"
    )

    print(
        "Strongest keyword score: "
        f"{analysis['strongest_keyword_score']:.4f}"
    )

    print(
        "Hybrid gap between first and second: "
        f"{analysis['hybrid_score_gap']:.6f}"
    )

    print()
    print(
        "Top hybrid result: "
        f"{analysis['top_hybrid_title']}"
    )

    print(
        "Strongest semantic result: "
        f"{analysis['strongest_semantic_title']}"
    )

    print(
        "Strongest keyword result: "
        f"{analysis['strongest_keyword_title']}"
    )

    print()
    print(
        "Top three hybrid results:"
    )

    titles = str(
        analysis["top_three_titles"]
    ).split(" | ")

    for rank, title in enumerate(
        titles,
        start=1,
    ):
        if title:
            print(
                f"  {rank}. {title}"
            )


def write_results(
    analyses: list[dict[str, object]],
) -> None:
    """Save all calibration results to CSV."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not analyses:
        raise RuntimeError(
            "No calibration results were generated."
        )

    fieldnames = list(
        analyses[0].keys()
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(analyses)


def main() -> None:
    """Run relevance calibration queries."""

    print(
        "Loading search engine..."
    )

    engine = SearchEngine()

    config = SearchConfig(
        top_k=50,
        candidate_pool=1_000,
        diversity_lambda=1.0,
        diversity_pool=50,
    ).validated()

    analyses: list[
        dict[str, object]
    ] = []

    for calibration_query in DEFAULT_QUERIES:
        analysis = analyse_query(
            engine=engine,
            calibration_query=calibration_query,
            config=config,
        )

        analyses.append(
            analysis
        )

        print_analysis(
            analysis
        )

    write_results(
        analyses
    )

    print()
    print("=" * 80)
    print(
        f"Calibration results saved to: "
        f"{OUTPUT_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()