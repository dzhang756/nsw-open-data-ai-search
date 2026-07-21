from __future__ import annotations

import argparse

from src.search_engine import (
    SearchConfig,
    SearchEngine,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Verify the reusable hybrid-search engine."
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
        default=10,
        help="Number of results to return.",
    )

    return parser.parse_args()


def main() -> None:
    """Run a search and print a concise verification result."""

    arguments = parse_arguments()
    query = " ".join(arguments.query)

    engine = SearchEngine()

    response = engine.search(
        query=query,
        config=SearchConfig(
            top_k=arguments.top_k,
        ),
    )

    print()
    print(f'Query: "{response.query}"')
    print(
        f"Catalogue size: "
        f"{response.catalogue_size:,}"
    )
    print(
        "Recognised keyword features: "
        f"{response.keyword_query_feature_count:,}"
    )
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

        print()
        print(f"{rank}. {result.title}")
        print(
            f"   Dataset ID: {result.dataset_id}"
        )
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

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()