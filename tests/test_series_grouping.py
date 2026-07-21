from __future__ import annotations

import unittest

from src.search_engine import SearchResult
from src.series_grouping import (
    group_search_results,
)


def make_result(
    dataset_id: str,
    title: str,
    organisation: str,
) -> SearchResult:
    """Create a minimal search result for grouping tests."""

    return SearchResult(
        row_index=0,
        dataset_id=dataset_id,
        title=title,
        description="Test description",
        organisation=organisation,
        resource_formats=("CSV",),
        metadata_modified="2026-01-01T00:00:00",
        dataset_url=(
            "https://data.nsw.gov.au/data/dataset/"
            + dataset_id
        ),
        hybrid_score=0.01,
        semantic_score=0.50,
        semantic_rank=1,
        keyword_score=0.25,
        keyword_rank=1,
    )


class SeriesGroupingTests(unittest.TestCase):
    """Tests for conservative dataset-series grouping."""

    def test_quarterly_editions_are_grouped(self) -> None:
        """Quarterly editions should become one result group."""

        results = (
            make_result(
                dataset_id="hospital-quarterly-2015-q3",
                title=(
                    "Hospital Quarterly: Performance of "
                    "NSW public hospitals - July to "
                    "September 2015"
                ),
                organisation=(
                    "Bureau of Health Information"
                ),
            ),
            make_result(
                dataset_id="hospital-quarterly-2016-q1",
                title=(
                    "Hospital Quarterly: Performance of "
                    "NSW public hospitals - January to "
                    "March 2016"
                ),
                organisation=(
                    "Bureau of Health Information"
                ),
            ),
        )

        groups = group_search_results(results)

        self.assertEqual(
            len(groups),
            1,
        )

        self.assertTrue(
            groups[0].is_series
        )

        self.assertEqual(
            len(groups[0].members),
            2,
        )

        self.assertEqual(
            groups[0].display_title,
            (
                "Hospital Quarterly: Performance of "
                "NSW public hospitals"
            ),
        )

    def test_unrelated_results_remain_separate(self) -> None:
        """Unrelated titles must not be grouped."""

        results = (
            make_result(
                dataset_id="crash-data",
                title="NSW Crash Data",
                organisation="Transport for NSW",
            ),
            make_result(
                dataset_id="road-speed-data",
                title="Road and Speed Data",
                organisation="Transport for NSW",
            ),
        )

        groups = group_search_results(results)

        self.assertEqual(
            len(groups),
            2,
        )

        self.assertFalse(
            groups[0].is_series
        )

        self.assertFalse(
            groups[1].is_series
        )

    def test_volume_editions_are_grouped(self) -> None:
        """Numbered volumes should become one result group."""

        results = (
            make_result(
                dataset_id="orbital-volume-1",
                title=(
                    "Representations report: "
                    "Western Sydney Orbital: volume 1"
                ),
                organisation="NSW Government",
            ),
            make_result(
                dataset_id="orbital-volume-2",
                title=(
                    "Representations report: "
                    "Western Sydney Orbital: volume 2"
                ),
                organisation="NSW Government",
            ),
        )

        groups = group_search_results(results)

        self.assertEqual(
            len(groups),
            1,
        )

        self.assertTrue(
            groups[0].is_series
        )


if __name__ == "__main__":
    unittest.main()