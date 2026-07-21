from __future__ import annotations

import argparse

from src.filter_options import (
    FilterOption,
    build_filter_option_summary,
)
from src.search_engine import SearchEngine

DEFAULT_LIMIT = 30

VALID_SECTIONS = (
    "all",
    "organisations",
    "formats",
    "categories",
    "dates",
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect available Data.NSW search-filter "
            "values and dataset counts."
        )
    )

    parser.add_argument(
        "--section",
        choices=VALID_SECTIONS,
        default="all",
        help=(
            "Filter-option section to display "
            "(default: all)."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Maximum options displayed per section "
            f"(default: {DEFAULT_LIMIT})."
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


def print_options(
    title: str,
    options: tuple[FilterOption, ...],
    limit: int,
) -> None:
    """Print one counted filter-option section."""

    print()
    print(title)
    print("-" * 80)
    print(
        f"Distinct values: {len(options):,}"
    )

    if not options:
        print("No values found.")
        return

    for rank, option in enumerate(
        options[:limit],
        start=1,
    ):
        print(
            f"{rank:>3}. "
            f"{option.value} "
            f"({option.dataset_count:,} datasets)"
        )

    hidden_count = (
        len(options)
        - min(limit, len(options))
    )

    if hidden_count > 0:
        print()
        print(
            f"... {hidden_count:,} additional values "
            "not displayed"
        )


def main() -> None:
    """Load catalogue metadata and list filter options."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    print(
        "Loading search indexes and catalogue metadata..."
    )

    engine = SearchEngine()

    summary = build_filter_option_summary(
        engine.metadata_by_id.values()
    )

    print()
    print(
        f"Catalogue datasets: "
        f"{len(engine.metadata_by_id):,}"
    )
    print("=" * 80)

    if arguments.section in (
        "all",
        "organisations",
    ):
        print_options(
            title="ORGANISATIONS",
            options=summary.organisations,
            limit=arguments.limit,
        )

    if arguments.section in (
        "all",
        "formats",
    ):
        print_options(
            title="RESOURCE FORMATS",
            options=summary.formats,
            limit=arguments.limit,
        )

    if arguments.section in (
        "all",
        "categories",
    ):
        print_options(
            title="DATA.NSW CATEGORIES",
            options=summary.categories,
            limit=arguments.limit,
        )

    if arguments.section in (
        "all",
        "dates",
    ):
        print()
        print("MODIFICATION DATES")
        print("-" * 80)

        earliest = (
            summary.earliest_modified_date
        )

        latest = (
            summary.latest_modified_date
        )

        print(
            "Earliest modification date: "
            + (
                earliest.isoformat()
                if earliest is not None
                else "Unknown"
            )
        )

        print(
            "Latest modification date: "
            + (
                latest.isoformat()
                if latest is not None
                else "Unknown"
            )
        )

        print(
            "Datasets without a valid date: "
            f"{summary.datasets_without_modified_date:,}"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()