from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from src.search_semantic import (
    get_organisation_title,
    load_catalogue_metadata,
    shorten_text,
)

SEARCH_DOCUMENTS_PATH = Path(
    "data/processed/search_documents.jsonl.gz"
)

DEFAULT_LIMIT = 30


def iter_search_documents() -> Iterator[dict[str, Any]]:
    """Yield generated search documents."""

    if not SEARCH_DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Search documents not found: {SEARCH_DOCUMENTS_PATH}"
        )

    with gzip.open(
        SEARCH_DOCUMENTS_PATH,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}."
                ) from error


def count_occurrences(text: str, value: str) -> int:
    """Count case-insensitive occurrences of a term or phrase."""

    return text.casefold().count(value.casefold())


def inspect_coverage(
    required_phrase: str,
    related_terms: list[str],
    limit: int,
) -> None:
    """Find datasets containing both query components."""

    metadata_by_id = load_catalogue_metadata()

    phrase_match_count = 0
    related_term_match_count = 0

    combined_matches: list[
        tuple[int, int, str, list[str]]
    ] = []

    for document in iter_search_documents():
        dataset_id = document.get("dataset_id")
        keyword_text = document.get("keyword_text")

        if (
            not isinstance(dataset_id, str)
            or not isinstance(keyword_text, str)
        ):
            continue

        phrase_occurrences = count_occurrences(
            keyword_text,
            required_phrase,
        )

        matched_terms = [
            term
            for term in related_terms
            if count_occurrences(keyword_text, term) > 0
        ]

        if phrase_occurrences > 0:
            phrase_match_count += 1

        if matched_terms:
            related_term_match_count += 1

        if phrase_occurrences > 0 and matched_terms:
            dataset = metadata_by_id.get(dataset_id, {})
            title = str(dataset.get("title", ""))

            title_has_phrase = int(
                required_phrase.casefold()
                in title.casefold()
            )

            combined_matches.append(
                (
                    title_has_phrase,
                    len(matched_terms),
                    dataset_id,
                    matched_terms,
                )
            )

    combined_matches.sort(
        key=lambda match: (
            match[0],
            match[1],
        ),
        reverse=True,
    )

    print(f'Required phrase: "{required_phrase}"')
    print(
        "Related terms: "
        + ", ".join(f'"{term}"' for term in related_terms)
    )
    print()
    print(
        f"Datasets containing required phrase: "
        f"{phrase_match_count:,}"
    )
    print(
        f"Datasets containing at least one related term: "
        f"{related_term_match_count:,}"
    )
    print(
        f"Datasets containing both components: "
        f"{len(combined_matches):,}"
    )
    print("=" * 80)

    for rank, (
        _,
        _,
        dataset_id,
        matched_terms,
    ) in enumerate(
        combined_matches[:limit],
        start=1,
    ):
        dataset = metadata_by_id[dataset_id]

        print()
        print(f"{rank}. {dataset.get('title', 'Untitled dataset')}")
        print(
            f"   Organisation: "
            f"{get_organisation_title(dataset)}"
        )
        print(
            "   Matched related terms: "
            + ", ".join(matched_terms)
        )

        description = shorten_text(
            dataset.get("description"),
        )

        if description:
            print(f"   Description: {description}")

        dataset_url = dataset.get("dataset_url")

        if dataset_url:
            print(f"   URL: {dataset_url}")

    print()
    print("=" * 80)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect whether multiple query concepts occur "
            "together in catalogue records."
        )
    )

    parser.add_argument(
        "--required-phrase",
        required=True,
        help="Exact phrase that must occur in the document.",
    )

    parser.add_argument(
        "--related-terms",
        nargs="+",
        required=True,
        help=(
            "One or more related terms, at least one of which "
            "must occur."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Maximum combined matches to display "
            f"(default: {DEFAULT_LIMIT})."
        ),
    )

    arguments = parser.parse_args()

    if arguments.limit <= 0:
        parser.error("--limit must be greater than zero.")

    return arguments


def main() -> None:
    """Run the term-coverage inspection."""

    arguments = parse_arguments()

    inspect_coverage(
        required_phrase=arguments.required_phrase.strip(),
        related_terms=[
            term.strip()
            for term in arguments.related_terms
            if term.strip()
        ],
        limit=arguments.limit,
    )


if __name__ == "__main__":
    main()