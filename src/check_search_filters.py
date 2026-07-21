from __future__ import annotations

import argparse

import numpy as np

from src.search_engine import SearchEngine
from src.search_filters import (
    SearchFilters,
    build_eligible_mask,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate structured search filters before "
            "integrating them into ranking."
        )
    )

    parser.add_argument(
        "--format",
        action="append",
        default=[],
        dest="formats",
        help=(
            "Require a resource format such as CSV, JSON "
            "or XLSX. Repeat to accept multiple formats."
        ),
    )

    parser.add_argument(
        "--machine-readable-only",
        action="store_true",
        help=(
            "Require at least one machine-readable "
            "resource."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help=(
            "Number of matching sample datasets to display "
            "(default: 10)."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Validate command-line values."""

    if arguments.limit <= 0:
        raise ValueError(
            "limit must be greater than zero."
        )


def main() -> None:
    """Apply filters and display matching samples."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    filters = SearchFilters(
        formats=tuple(
            arguments.formats
        ),
        machine_readable_only=(
            arguments.machine_readable_only
        ),
    ).validated()

    print("Loading search indexes and metadata...")

    engine = SearchEngine()

    filter_result = build_eligible_mask(
        index_records=engine.embedding_records,
        metadata_by_id=engine.metadata_by_id,
        filters=filters,
    )

    matching_indices = np.flatnonzero(
        filter_result.eligible_mask
    )

    print()
    print(
        "Requested formats: "
        + (
            ", ".join(filters.formats)
            if filters.formats
            else "None"
        )
    )
    print(
        "Machine-readable only: "
        + (
            "Yes"
            if filters.machine_readable_only
            else "No"
        )
    )
    print(
        f"Total indexed datasets: "
        f"{filter_result.total_count:,}"
    )
    print(
        f"Eligible datasets: "
        f"{filter_result.eligible_count:,}"
    )
    print(
        f"Excluded datasets: "
        f"{filter_result.excluded_count:,}"
    )
    print("=" * 80)

    for sample_number, row_index in enumerate(
        matching_indices[: arguments.limit],
        start=1,
    ):
        record = engine.embedding_records[
            int(row_index)
        ]

        dataset_id = record[
            "dataset_id"
        ]

        dataset = engine.metadata_by_id[
            dataset_id
        ]

        title = (
            dataset.get("title")
            or "Untitled dataset"
        )

        organisation = dataset.get(
            "organisation"
        )

        if isinstance(organisation, dict):
            organisation_title = (
                organisation.get("title")
                or organisation.get("name")
                or "Organisation not specified"
            )
        else:
            organisation_title = (
                organisation
                if isinstance(organisation, str)
                and organisation
                else "Organisation not specified"
            )

        resource_formats = dataset.get(
            "resource_formats"
        )

        if not isinstance(resource_formats, list):
            resource_formats = []

        format_text = (
            ", ".join(resource_formats)
            if resource_formats
            else "No formats specified"
        )

        machine_readable = (
            dataset.get(
                "has_machine_readable_resource"
            )
            is True
        )

        print()
        print(
            f"{sample_number}. {title}"
        )
        print(
            f"   Organisation: "
            f"{organisation_title}"
        )
        print(
            f"   Formats: {format_text}"
        )
        print(
            "   Machine-readable: "
            + (
                "Yes"
                if machine_readable
                else "No"
            )
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()