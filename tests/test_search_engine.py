from __future__ import annotations

import unittest

from src.search_engine import (
    SearchConfig,
    SearchEngine,
)


class SearchEngineTests(unittest.TestCase):
    """Regression and quality checks for hybrid search."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the indexes and AI model once for all tests."""

        cls.engine = SearchEngine()
        cls.config = SearchConfig(
            top_k=10,
        )

    def search_titles(self, query: str) -> list[str]:
        """Run a query and return the result titles."""

        response = self.engine.search(
            query=query,
            config=self.config,
        )

        return [
            result.title
            for result in response.results
        ]

    def assert_title_in_top_n(
        self,
        titles: list[str],
        expected_title: str,
        top_n: int,
    ) -> None:
        """Assert that a title occurs within the first N results."""

        self.assertIn(
            expected_title,
            titles[:top_n],
            msg=(
                f'Expected "{expected_title}" within the '
                f"top {top_n} results.\n"
                f"Received: {titles[:top_n]}"
            ),
        )

    def test_crash_query_returns_primary_dataset_first(
        self,
    ) -> None:
        """The main NSW crash dataset should rank first."""

        titles = self.search_titles(
            "road crash data for Western Sydney"
        )

        self.assertEqual(
            titles[0],
            "NSW Crash Data",
        )

    def test_sydney_harbour_water_quality_exact_match(
        self,
    ) -> None:
        """The dedicated harbour database should rank first."""

        titles = self.search_titles(
            "Sydney Harbour water quality monitoring data"
        )

        self.assertEqual(
            titles[0],
            (
                "Sydney Harbour Water Quality Database "
                "2021-2023"
            ),
        )

    def test_emergency_department_measure_is_prominent(
        self,
    ) -> None:
        """The direct emergency-treatment measure should be prominent."""

        titles = self.search_titles(
            (
                "emergency department waiting times "
                "by local health district"
            )
        )

        self.assert_title_in_top_n(
            titles=titles,
            expected_title=(
                "Proportion of patients commencing emergency "
                "treatment on time, NSW, quarterly"
            ),
            top_n=3,
        )

    def test_passenger_transport_result_is_returned(
        self,
    ) -> None:
        """Passenger-travel data should appear in the result set."""

        titles = self.search_titles(
            "Sydney public transport passenger data"
        )

        self.assert_title_in_top_n(
            titles=titles,
            expected_title=(
                "TfNSW Passenger travel performance reports"
            ),
            top_n=10,
        )

    def test_response_contains_requested_number_of_results(
        self,
    ) -> None:
        """A normal query should return the requested result count."""

        response = self.engine.search(
            query="NSW water quality monitoring",
            config=SearchConfig(top_k=5),
        )

        self.assertEqual(
            len(response.results),
            5,
        )

    def test_search_results_have_unique_dataset_ids(
        self,
    ) -> None:
        """One dataset must not appear twice in one result set."""

        response = self.engine.search(
            query="NSW road safety statistics",
            config=self.config,
        )

        dataset_ids = [
            result.dataset_id
            for result in response.results
        ]

        self.assertEqual(
            len(dataset_ids),
            len(set(dataset_ids)),
        )

    def test_blank_query_is_rejected(self) -> None:
        """Whitespace-only queries should not be searched."""

        with self.assertRaises(ValueError):
            self.engine.search(
                query="   ",
                config=self.config,
            )

    def test_config_weights_are_normalised(self) -> None:
        """Equivalent unnormalised weights should be accepted."""

        config = SearchConfig(
            semantic_weight=7,
            keyword_weight=3,
        ).validated()

        self.assertAlmostEqual(
            config.semantic_weight,
            0.70,
        )
        self.assertAlmostEqual(
            config.keyword_weight,
            0.30,
        )


if __name__ == "__main__":
    unittest.main()