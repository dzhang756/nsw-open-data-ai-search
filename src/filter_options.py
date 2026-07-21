from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from src.search_filters import (
    compact_text,
    dataset_modified_date,
    dataset_organisation_title,
    normalise_format_value,
)


@dataclass(frozen=True)
class FilterOption:
    """One selectable filter value and its dataset count."""

    value: str
    dataset_count: int


@dataclass(frozen=True)
class FilterOptionSummary:
    """Available values for the search-filter interface."""

    organisations: tuple[FilterOption, ...]
    formats: tuple[FilterOption, ...]
    categories: tuple[FilterOption, ...]
    earliest_modified_date: date | None
    latest_modified_date: date | None
    datasets_without_modified_date: int


def extract_named_values(
    values: Any,
) -> tuple[str, ...]:
    """Extract unique names from strings or dictionaries."""

    if not isinstance(values, list):
        return ()

    extracted: list[str] = []
    seen_values: set[str] = set()

    for value in values:
        if isinstance(value, str):
            text = compact_text(value)

        elif isinstance(value, dict):
            raw_text = (
                value.get("title")
                or value.get("display_name")
                or value.get("name")
            )

            text = (
                compact_text(raw_text)
                if isinstance(raw_text, str)
                else ""
            )

        else:
            text = ""

        normalised = text.casefold()

        if not text or normalised in seen_values:
            continue

        seen_values.add(normalised)
        extracted.append(text)

    return tuple(extracted)


def dataset_categories(
    dataset: dict[str, Any],
) -> tuple[str, ...]:
    """Return Data.NSW CKAN group titles for one dataset."""

    return extract_named_values(
        dataset.get("groups")
    )


def dataset_formats(
    dataset: dict[str, Any],
) -> tuple[str, ...]:
    """Return unique normalised resource-format labels."""

    raw_formats = dataset.get(
        "resource_formats"
    )

    if not isinstance(raw_formats, list):
        return ()

    formats: list[str] = []
    seen_formats: set[str] = set()

    for raw_format in raw_formats:
        if not isinstance(raw_format, str):
            continue

        normalised_format = (
            normalise_format_value(
                raw_format
            )
        )

        if (
            not normalised_format
            or normalised_format in seen_formats
        ):
            continue

        seen_formats.add(
            normalised_format
        )

        formats.append(
            normalised_format
        )

    return tuple(formats)


def counter_to_options(
    counter: Counter[str],
) -> tuple[FilterOption, ...]:
    """Convert counted values into consistently sorted options."""

    sorted_items = sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            item[0].casefold(),
        ),
    )

    return tuple(
        FilterOption(
            value=value,
            dataset_count=count,
        )
        for value, count in sorted_items
    )


def build_filter_option_summary(
    datasets: Iterable[dict[str, Any]],
) -> FilterOptionSummary:
    """Collect available filter values from catalogue metadata."""

    organisation_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    modified_dates: list[date] = []
    datasets_without_modified_date = 0

    for dataset in datasets:
        organisation = (
            dataset_organisation_title(
                dataset
            )
        )

        if organisation:
            organisation_counts[
                organisation
            ] += 1

        # Count each value at most once per dataset.
        format_counts.update(
            dataset_formats(dataset)
        )

        category_counts.update(
            dataset_categories(dataset)
        )

        modified_date = (
            dataset_modified_date(
                dataset
            )
        )

        if modified_date is None:
            datasets_without_modified_date += 1
        else:
            modified_dates.append(
                modified_date
            )

    earliest_modified_date = (
        min(modified_dates)
        if modified_dates
        else None
    )

    latest_modified_date = (
        max(modified_dates)
        if modified_dates
        else None
    )

    return FilterOptionSummary(
        organisations=counter_to_options(
            organisation_counts
        ),
        formats=counter_to_options(
            format_counts
        ),
        categories=counter_to_options(
            category_counts
        ),
        earliest_modified_date=(
            earliest_modified_date
        ),
        latest_modified_date=(
            latest_modified_date
        ),
        datasets_without_modified_date=(
            datasets_without_modified_date
        ),
    )