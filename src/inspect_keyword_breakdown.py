from __future__ import annotations

import argparse
import re
from typing import Any

import numpy as np
from scipy import sparse

from src.search_engine import (
    SearchConfig,
    SearchEngine,
)

WORD_PATTERN = re.compile(r"[a-z0-9]+")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect field-level TF-IDF contributions "
            "for hybrid-search results."
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
        help="Number of hybrid results to inspect.",
    )

    return parser.parse_args()


def compact_text(value: Any) -> str:
    """Return normalised searchable text."""

    if not isinstance(value, str):
        return ""

    words = WORD_PATTERN.findall(
        value.casefold()
    )

    return " ".join(words)


def named_values(values: Any) -> list[str]:
    """Extract searchable values from strings or dictionaries."""

    if not isinstance(values, list):
        return []

    extracted: list[str] = []

    for value in values:
        if isinstance(value, str):
            text = value

        elif isinstance(value, dict):
            text = (
                value.get("title")
                or value.get("display_name")
                or value.get("name")
                or ""
            )

        else:
            text = ""

        compact_value = compact_text(text)

        if compact_value:
            extracted.append(compact_value)

    return extracted


def build_dataset_fields(
    dataset: dict[str, Any],
) -> dict[str, str]:
    """Reconstruct the searchable text for each keyword field."""

    organisation = dataset.get("organisation")

    if isinstance(organisation, dict):
        organisation_values = [
            organisation.get("title", ""),
            organisation.get("name", ""),
        ]
    elif isinstance(organisation, str):
        organisation_values = [organisation]
    else:
        organisation_values = []

    resource_values = named_values(
        dataset.get("resource_formats")
    )

    resources = dataset.get("resources")

    if isinstance(resources, list):
        for resource in resources:
            if not isinstance(resource, dict):
                continue

            resource_name = compact_text(
                resource.get("name")
            )

            if resource_name:
                resource_values.append(
                    resource_name
                )

    return {
        "title": compact_text(
            dataset.get("title")
        ),
        "subjects": " ".join(
            named_values(dataset.get("tags"))
            + named_values(dataset.get("groups"))
        ),
        "organisation": " ".join(
            compact_text(value)
            for value in organisation_values
            if compact_text(value)
        ),
        "resources": " ".join(
            resource_values
        ),
        "description": compact_text(
            dataset.get("description")
        ),
    }


def query_phrases(query: str) -> list[str]:
    """Create ordered two-word phrases from the query."""

    words = WORD_PATTERN.findall(
        query.casefold()
    )

    return [
        f"{words[index]} {words[index + 1]}"
        for index in range(len(words) - 1)
    ]


def main() -> None:
    """Inspect field scores for the current hybrid results."""

    arguments = parse_arguments()

    query = " ".join(arguments.query).strip()

    if not query:
        raise ValueError(
            "A non-empty query is required."
        )

    config = SearchConfig(
        top_k=arguments.top_k,
    ).validated()

    engine = SearchEngine()

    response = engine.search(
        query=query,
        config=config,
    )

    query_vector = sparse.csr_matrix(
        engine.keyword_vectorizer.transform(
            [query]
        ),
        dtype=np.float32,
    )

    query_transpose = query_vector.transpose()

    field_scores: dict[str, np.ndarray] = {}

    for field_name, matrix in (
        engine.keyword_matrices.items()
    ):
        field_scores[field_name] = np.asarray(
            (
                matrix
                @ query_transpose
            ).toarray()
        ).ravel()

    weights = (
        config.keyword_field_weights.as_dict()
    )

    phrases = query_phrases(query)

    print()
    print(f'Query: "{query}"')
    print(
        "Query phrases: "
        + ", ".join(phrases)
    )
    print("=" * 80)

    for rank, result in enumerate(
        response.results,
        start=1,
    ):
        dataset = engine.metadata_by_id[
            result.dataset_id
        ]

        dataset_fields = build_dataset_fields(
            dataset
        )

        print()
        print(f"{rank}. {result.title}")
        print(
            f"   Combined keyword score: "
            f"{result.keyword_score:.6f}"
        )
        print(
            f"   Keyword rank: "
            f"{result.keyword_rank}"
        )

        total_contribution = 0.0

        for field_name, weight in weights.items():
            raw_score = float(
                field_scores[field_name][
                    result.row_index
                ]
            )

            contribution = weight * raw_score
            total_contribution += contribution

            matched_phrases = [
                phrase
                for phrase in phrases
                if phrase in dataset_fields[
                    field_name
                ]
            ]

            phrase_text = (
                ", ".join(matched_phrases)
                if matched_phrases
                else "None"
            )

            print(
                f"   {field_name}: "
                f"raw={raw_score:.6f}, "
                f"weight={weight:.2f}, "
                f"contribution={contribution:.6f}"
            )
            print(
                f"      Matched query phrases: "
                f"{phrase_text}"
            )

        print(
            f"   Recalculated total: "
            f"{total_contribution:.6f}"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()