from __future__ import annotations

import argparse
from typing import Any

from src.search_engine import (
    SearchConfig,
    SearchEngine,
    SearchResponse,
)

DEFAULT_CONFIG = SearchConfig()


def shorten_text(
    value: Any,
    maximum_length: int = 300,
) -> str:
    """Create a compact single-line description preview."""

    if not isinstance(value, str):
        return ""

    compact_value = " ".join(value.split())

    if len(compact_value) <= maximum_length:
        return compact_value

    shortened = compact_value[: maximum_length + 1]
    final_space = shortened.rfind(" ")

    if final_space >= maximum_length * 0.75:
        shortened = shortened[:final_space]
    else:
        shortened = shortened[:maximum_length]

    return shortened.rstrip(" ,.;:-") + "…"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Search the Data.NSW catalogue using hybrid "
            "semantic and keyword retrieval."
        )
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CONFIG.top_k,
        help=(
            "Number of results to return "
            f"(default: {DEFAULT_CONFIG.top_k})."
        ),
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=DEFAULT_CONFIG.candidate_pool,
        help=(
            "Number of leading results from each retrieval "
            "method considered for fusion "
            f"(default: {DEFAULT_CONFIG.candidate_pool})."
        ),
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=DEFAULT_CONFIG.semantic_weight,
        help=(
            "Semantic-ranking weight "
            f"(default: {DEFAULT_CONFIG.semantic_weight})."
        ),
    )

    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=DEFAULT_CONFIG.keyword_weight,
        help=(
            "Keyword-ranking weight "
            f"(default: {DEFAULT_CONFIG.keyword_weight})."
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_CONFIG.rrf_k,
        help=(
            "Reciprocal Rank Fusion smoothing constant "
            f"(default: {DEFAULT_CONFIG.rrf_k})."
        ),
    )

    parser.add_argument(
        "--diversity-lambda",
        type=float,
        default=DEFAULT_CONFIG.diversity_lambda,
        help=(
            "Balance between relevance and diversity. "
            "Use 1.0 for no diversification "
            f"(default: {DEFAULT_CONFIG.diversity_lambda})."
        ),
    )

    parser.add_argument(
        "--diversity-pool",
        type=int,
        default=DEFAULT_CONFIG.diversity_pool,
        help=(
            "Number of leading hybrid results considered "
            "during diversification "
            f"(default: {DEFAULT_CONFIG.diversity_pool})."
        ),
    )

    return parser


def print_response(response: SearchResponse) -> None:
    """Display a structured search response in the terminal."""

    config = response.config

    print()
    print(f'Query: "{response.query}"')
    print(
        "Weights: "
        f"semantic={config.semantic_weight:.2f}, "
        f"keyword={config.keyword_weight:.2f}"
    )
    print(
        "Diversification: "
        f"lambda={config.diversity_lambda:.2f}, "
        f"candidate pool={config.diversity_pool}"
    )
    print(
        "Recognised keyword features: "
        f"{response.keyword_query_feature_count:,}"
    )
    print(
        f"Catalogue datasets searched: "
        f"{response.catalogue_size:,}"
    )
    print(f"Results returned: {len(response.results)}")
    print("=" * 80)

    for rank, result in enumerate(
        response.results,
        start=1,
    ):
        keyword_rank = (
            str(result.keyword_rank)
            if result.keyword_rank is not None
            else "No positive keyword match"
        )

        format_text = (
            ", ".join(result.resource_formats)
            if result.resource_formats
            else "No formats specified"
        )

        modified = (
            result.metadata_modified
            or "Unknown"
        )

        description = shorten_text(
            result.description
        )

        print()
        print(f"{rank}. {result.title}")
        print(
            f"   Hybrid score: "
            f"{result.hybrid_score:.6f}"
        )
        print(
            "   Semantic: "
            f"score={result.semantic_score:.4f}, "
            f"rank={result.semantic_rank}"
        )
        print(
            "   Keyword: "
            f"score={result.keyword_score:.4f}, "
            f"rank={keyword_rank}"
        )
        print(
            f"   Organisation: "
            f"{result.organisation}"
        )
        print(f"   Formats: {format_text}")
        print(f"   Modified: {modified}")

        if description:
            print(f"   Description: {description}")

        if result.dataset_url:
            print(f"   URL: {result.dataset_url}")

    print()
    print("=" * 80)


def main() -> None:
    """Run hybrid search from the command line."""

    parser = build_parser()
    arguments = parser.parse_args()

    query = " ".join(arguments.query).strip()

    if not query:
        parser.error(
            "A non-empty search query is required."
        )

    try:
        config = SearchConfig(
            top_k=arguments.top_k,
            candidate_pool=arguments.candidate_pool,
            semantic_weight=arguments.semantic_weight,
            keyword_weight=arguments.keyword_weight,
            rrf_k=arguments.rrf_k,
            diversity_lambda=arguments.diversity_lambda,
            diversity_pool=arguments.diversity_pool,
        ).validated()
    except ValueError as error:
        parser.error(str(error))

    engine = SearchEngine()

    response = engine.search(
        query=query,
        config=config,
    )

    print_response(response)


if __name__ == "__main__":
    main()