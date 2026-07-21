from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from src.search_engine import (
    SearchEngine,
    SearchResult,
)
from src.search_filters import (
    SearchFilters,
    build_eligible_mask,
    dataset_modified_date,
    dataset_organisation_title,
)


@dataclass(frozen=True)
class BrowseResponse:
    """Results returned from filter-only catalogue browsing."""

    filters: SearchFilters
    results: tuple[SearchResult, ...]
    catalogue_size: int
    eligible_dataset_count: int

    @property
    def excluded_dataset_count(self) -> int:
        """Return the number of datasets excluded by filters."""

        return (
            self.catalogue_size
            - self.eligible_dataset_count
        )


def resource_formats(
    dataset: dict[str, Any],
) -> tuple[str, ...]:
    """Return clean resource-format display values."""

    values = dataset.get(
        "resource_formats"
    )

    if not isinstance(values, list):
        return ()

    return tuple(
        value
        for value in values
        if isinstance(value, str)
        and value.strip()
    )


def dataset_title(
    dataset: dict[str, Any],
) -> str:
    """Return a dataset display title."""

    value = dataset.get("title")

    if isinstance(value, str) and value.strip():
        return value.strip()

    return "Untitled dataset"


def browse_catalogue(
    engine: SearchEngine,
    filters: SearchFilters,
    limit: int = 200,
) -> BrowseResponse:
    """
    Browse datasets satisfying structured filters.

    Eligible datasets are ordered by most recent metadata
    modification date, with titles used as a stable secondary
    ordering value.
    """

    if limit <= 0:
        raise ValueError(
            "Browse limit must be greater than zero."
        )

    active_filters = filters.validated()

    filter_result = build_eligible_mask(
        index_records=engine.embedding_records,
        metadata_by_id=engine.metadata_by_id,
        filters=active_filters,
    )

    eligible_indices = [
        int(index)
        for index in np.flatnonzero(
            filter_result.eligible_mask
        )
    ]

    # Establish title order first. Python sorting is stable,
    # so this remains the secondary order after date sorting.
    eligible_indices.sort(
        key=lambda row_index: dataset_title(
            engine.metadata_by_id[
                engine.embedding_records[
                    row_index
                ]["dataset_id"]
            ]
        ).casefold()
    )

    eligible_indices.sort(
        key=lambda row_index: (
            dataset_modified_date(
                engine.metadata_by_id[
                    engine.embedding_records[
                        row_index
                    ]["dataset_id"]
                ]
            )
            or date.min,
            engine.metadata_by_id[
                engine.embedding_records[
                    row_index
                ]["dataset_id"]
            ].get(
                "metadata_modified",
                "",
            ),
        ),
        reverse=True,
    )

    results: list[SearchResult] = []

    for browse_rank, row_index in enumerate(
        eligible_indices[:limit],
        start=1,
    ):
        dataset_id = engine.embedding_records[
            row_index
        ]["dataset_id"]

        dataset = engine.metadata_by_id[
            dataset_id
        ]

        organisation = (
            dataset_organisation_title(
                dataset
            )
            or "Organisation not specified"
        )

        results.append(
            SearchResult(
                row_index=row_index,
                dataset_id=dataset_id,
                title=dataset_title(dataset),
                description=(
                    dataset.get("description")
                    if isinstance(
                        dataset.get("description"),
                        str,
                    )
                    else ""
                ),
                organisation=organisation,
                resource_formats=(
                    resource_formats(dataset)
                ),
                metadata_modified=(
                    dataset.get(
                        "metadata_modified"
                    )
                    if isinstance(
                        dataset.get(
                            "metadata_modified"
                        ),
                        str,
                    )
                    else ""
                ),
                dataset_url=(
                    dataset.get("dataset_url")
                    if isinstance(
                        dataset.get("dataset_url"),
                        str,
                    )
                    else ""
                ),
                hybrid_score=0.0,
                semantic_score=0.0,
                semantic_rank=browse_rank,
                keyword_score=0.0,
                keyword_rank=None,
            )
        )

    return BrowseResponse(
        filters=active_filters,
        results=tuple(results),
        catalogue_size=(
            filter_result.total_count
        ),
        eligible_dataset_count=(
            filter_result.eligible_count
        ),
    )