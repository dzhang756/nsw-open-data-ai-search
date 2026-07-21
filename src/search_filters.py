from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

NON_ALPHANUMERIC_PATTERN = re.compile(
    r"[^A-Z0-9]+"
)


@dataclass(frozen=True)
class SearchFilters:
    """Structured constraints applied before search ranking."""

    formats: tuple[str, ...] = ()
    organisations: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    modified_from: date | None = None
    modified_to: date | None = None
    machine_readable_only: bool = False

    def validated(self) -> SearchFilters:
        """Normalise and validate filter values."""

        normalised_formats: list[str] = []
        seen_formats: set[str] = set()

        for value in self.formats:
            if not isinstance(value, str):
                raise ValueError(
                    "Every requested format must be a string."
                )

            normalised = normalise_format_value(value)

            if (
                not normalised
                or normalised in seen_formats
            ):
                continue

            seen_formats.add(normalised)
            normalised_formats.append(normalised)

        cleaned_organisations: list[str] = []
        seen_organisations: set[str] = set()

        for value in self.organisations:
            if not isinstance(value, str):
                raise ValueError(
                    "Every requested organisation must "
                    "be a string."
                )

            display_value = compact_text(value)

            normalised_value = (
                normalise_organisation_value(
                    display_value
                )
            )

            if (
                not normalised_value
                or normalised_value
                in seen_organisations
            ):
                continue

            seen_organisations.add(
                normalised_value
            )

            cleaned_organisations.append(
                display_value
            )

        cleaned_categories: list[str] = []
        seen_categories: set[str] = set()

        for value in self.categories:
            if not isinstance(value, str):
                raise ValueError(
                    "Every requested category must "
                    "be a string."
                )

            display_value = compact_text(value)

            normalised_value = (
                normalise_category_value(
                    display_value
                )
            )

            if (
                not normalised_value
                or normalised_value in seen_categories
            ):
                continue

            seen_categories.add(
                normalised_value
            )

            cleaned_categories.append(
                display_value
            )

        if (
            self.modified_from is not None
            and not isinstance(
                self.modified_from,
                date,
            )
        ):
            raise ValueError(
                "modified_from must be a date."
            )

        if (
            self.modified_to is not None
            and not isinstance(
                self.modified_to,
                date,
            )
        ):
            raise ValueError(
                "modified_to must be a date."
            )

        if (
            self.modified_from is not None
            and self.modified_to is not None
            and self.modified_from > self.modified_to
        ):
            raise ValueError(
                "modified_from cannot be later than "
                "modified_to."
            )

        return SearchFilters(
            formats=tuple(normalised_formats),
            organisations=tuple(
                cleaned_organisations
            ),
            categories=tuple(
                cleaned_categories
            ),
            modified_from=self.modified_from,
            modified_to=self.modified_to,
            machine_readable_only=(
                self.machine_readable_only
            ),
        )

    @property
    def is_active(self) -> bool:
        """Return whether at least one filter is enabled."""

        return bool(
            self.formats
            or self.organisations
            or self.categories
            or self.modified_from is not None
            or self.modified_to is not None
            or self.machine_readable_only
        )


@dataclass(frozen=True)
class FilterResult:
    """Result of applying filters to indexed datasets."""

    eligible_mask: np.ndarray
    eligible_count: int
    total_count: int

    @property
    def excluded_count(self) -> int:
        """Return the number of excluded datasets."""

        return (
            self.total_count
            - self.eligible_count
        )


def compact_text(value: str) -> str:
    """Return compact single-line text."""

    return " ".join(value.split())


def normalise_format_value(
    value: str,
) -> str:
    """Normalise a requested or indexed format value."""

    compact_value = compact_text(
        value.upper()
    )

    return compact_value.strip()


def format_match_values(
    value: str,
) -> set[str]:
    """
    Create searchable aliases for one resource format.

    Examples:
    - EXCEL (XLSX) matches EXCEL and XLSX
    - CSV matches CSV
    - ARCGIS REST matches ARCGIS, REST and ARCGIS REST
    """

    normalised = normalise_format_value(
        value
    )

    if not normalised:
        return set()

    match_values = {
        normalised,
    }

    tokens = [
        token
        for token
        in NON_ALPHANUMERIC_PATTERN.split(
            normalised
        )
        if token
    ]

    match_values.update(tokens)

    return match_values


def normalise_organisation_value(
    value: str,
) -> str:
    """Normalise an organisation name for exact matching."""

    return compact_text(
        value
    ).casefold()


def normalise_category_value(
    value: str,
) -> str:
    """Normalise a category name for exact matching."""

    return compact_text(
        value
    ).casefold()


def dataset_organisation_title(
    dataset: dict[str, Any],
) -> str:
    """Extract a dataset organisation's display title."""

    organisation = dataset.get(
        "organisation"
    )

    if isinstance(organisation, str):
        return compact_text(
            organisation
        )

    if not isinstance(organisation, dict):
        return ""

    value = (
        organisation.get("title")
        or organisation.get("display_name")
        or organisation.get("name")
    )

    if not isinstance(value, str):
        return ""

    return compact_text(value)


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
            display_value = compact_text(
                value
            )

        elif isinstance(value, dict):
            raw_value = (
                value.get("title")
                or value.get("display_name")
                or value.get("name")
            )

            display_value = (
                compact_text(raw_value)
                if isinstance(raw_value, str)
                else ""
            )

        else:
            display_value = ""

        normalised_value = (
            display_value.casefold()
        )

        if (
            not display_value
            or normalised_value in seen_values
        ):
            continue

        seen_values.add(
            normalised_value
        )

        extracted.append(
            display_value
        )

    return tuple(extracted)


def dataset_category_values(
    dataset: dict[str, Any],
) -> set[str]:
    """Return normalised Data.NSW category values."""

    categories = extract_named_values(
        dataset.get("groups")
    )

    return {
        normalise_category_value(category)
        for category in categories
        if normalise_category_value(category)
    }


def dataset_format_values(
    dataset: dict[str, Any],
) -> set[str]:
    """Return all matchable format values for one dataset."""

    resource_formats = dataset.get(
        "resource_formats"
    )

    if not isinstance(resource_formats, list):
        return set()

    values: set[str] = set()

    for resource_format in resource_formats:
        if not isinstance(
            resource_format,
            str,
        ):
            continue

        values.update(
            format_match_values(
                resource_format
            )
        )

    return values


def dataset_is_machine_readable(
    dataset: dict[str, Any],
) -> bool:
    """Return the cleaned machine-readable flag."""

    return (
        dataset.get(
            "has_machine_readable_resource"
        )
        is True
    )


def dataset_modified_date(
    dataset: dict[str, Any],
) -> date | None:
    """Extract an ISO modification date from metadata."""

    value = dataset.get(
        "metadata_modified"
    )

    if not isinstance(value, str):
        return None

    compact_value = compact_text(value)

    if not compact_value:
        return None

    try:
        parsed_datetime = datetime.fromisoformat(
            compact_value.replace(
                "Z",
                "+00:00",
            )
        )

        return parsed_datetime.date()

    except ValueError:
        pass

    try:
        return date.fromisoformat(
            compact_value[:10]
        )

    except ValueError:
        return None


def dataset_matches_filters(
    dataset: dict[str, Any],
    filters: SearchFilters,
) -> bool:
    """Return whether one dataset satisfies all filters."""

    if filters.formats:
        available_formats = (
            dataset_format_values(
                dataset
            )
        )

        requested_formats = set(
            filters.formats
        )

        # Multiple formats use OR logic.
        if not (
            available_formats
            & requested_formats
        ):
            return False

    if filters.organisations:
        dataset_organisation = (
            normalise_organisation_value(
                dataset_organisation_title(
                    dataset
                )
            )
        )

        requested_organisations = {
            normalise_organisation_value(
                organisation
            )
            for organisation
            in filters.organisations
        }

        # Multiple organisations use OR logic.
        if (
            dataset_organisation
            not in requested_organisations
        ):
            return False

    if filters.categories:
        available_categories = (
            dataset_category_values(
                dataset
            )
        )

        requested_categories = {
            normalise_category_value(
                category
            )
            for category in filters.categories
        }

        # Multiple categories use OR logic.
        if not (
            available_categories
            & requested_categories
        ):
            return False

    if (
        filters.modified_from is not None
        or filters.modified_to is not None
    ):
        modified_date = dataset_modified_date(
            dataset
        )

        if modified_date is None:
            return False

        if (
            filters.modified_from is not None
            and modified_date
            < filters.modified_from
        ):
            return False

        if (
            filters.modified_to is not None
            and modified_date
            > filters.modified_to
        ):
            return False

    if (
        filters.machine_readable_only
        and not dataset_is_machine_readable(
            dataset
        )
    ):
        return False

    return True


def build_eligible_mask(
    index_records: list[dict[str, Any]],
    metadata_by_id: dict[
        str,
        dict[str, Any],
    ],
    filters: SearchFilters | None = None,
) -> FilterResult:
    """Build a Boolean mask for datasets satisfying filters."""

    active_filters = (
        filters or SearchFilters()
    ).validated()

    total_count = len(index_records)

    if not active_filters.is_active:
        eligible_mask = np.ones(
            total_count,
            dtype=bool,
        )

        return FilterResult(
            eligible_mask=eligible_mask,
            eligible_count=total_count,
            total_count=total_count,
        )

    eligible_mask = np.zeros(
        total_count,
        dtype=bool,
    )

    for row_index, record in enumerate(
        index_records
    ):
        dataset_id = record.get(
            "dataset_id"
        )

        if (
            not isinstance(dataset_id, str)
            or not dataset_id
        ):
            raise RuntimeError(
                f"Index row {row_index} has no valid "
                "dataset ID."
            )

        dataset = metadata_by_id.get(
            dataset_id
        )

        if dataset is None:
            raise RuntimeError(
                "No cleaned catalogue metadata found for "
                f"dataset {dataset_id}."
            )

        eligible_mask[row_index] = (
            dataset_matches_filters(
                dataset=dataset,
                filters=active_filters,
            )
        )

    eligible_count = int(
        np.count_nonzero(
            eligible_mask
        )
    )

    return FilterResult(
        eligible_mask=eligible_mask,
        eligible_count=eligible_count,
        total_count=total_count,
    )