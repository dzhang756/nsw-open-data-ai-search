from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

NON_ALPHANUMERIC_PATTERN = re.compile(
    r"[^A-Z0-9]+"
)


@dataclass(frozen=True)
class SearchFilters:
    """Structured constraints applied before search ranking."""

    formats: tuple[str, ...] = ()
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

            if not normalised:
                continue

            if normalised in seen_formats:
                continue

            seen_formats.add(normalised)
            normalised_formats.append(normalised)

        return SearchFilters(
            formats=tuple(normalised_formats),
            machine_readable_only=(
                self.machine_readable_only
            ),
        )

    @property
    def is_active(self) -> bool:
        """Return whether at least one filter is enabled."""

        return bool(
            self.formats
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

        return self.total_count - self.eligible_count


def normalise_format_value(value: str) -> str:
    """Normalise a requested or indexed format value."""

    compact_value = " ".join(
        value.upper().split()
    )

    return compact_value.strip()


def format_match_values(value: str) -> set[str]:
    """
    Create searchable aliases for one resource format.

    Examples:
    - EXCEL (XLSX) matches EXCEL and XLSX
    - CSV matches CSV
    - ARCGIS REST matches ARCGIS, REST and ARCGIS REST
    """

    normalised = normalise_format_value(value)

    if not normalised:
        return set()

    match_values = {
        normalised,
    }

    tokens = [
        token
        for token in NON_ALPHANUMERIC_PATTERN.split(
            normalised
        )
        if token
    ]

    match_values.update(tokens)

    return match_values


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
        if not isinstance(resource_format, str):
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


def dataset_matches_filters(
    dataset: dict[str, Any],
    filters: SearchFilters,
) -> bool:
    """Return whether one dataset satisfies all filters."""

    if filters.formats:
        available_formats = (
            dataset_format_values(dataset)
        )

        requested_formats = set(
            filters.formats
        )

        # Multiple requested formats use OR logic:
        # CSV + JSON means either CSV or JSON is accepted.
        if not (
            available_formats
            & requested_formats
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
    metadata_by_id: dict[str, dict[str, Any]],
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