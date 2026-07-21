from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.search_engine import SearchResult

MONTH_NAMES = (
    "january|february|march|april|may|june|"
    "july|august|september|october|november|december"
)

MONTH_RANGE_PATTERN = re.compile(
    rf"\b(?:{MONTH_NAMES})\s+"
    rf"(?:to|through|-)\s+"
    rf"(?:{MONTH_NAMES})\s+"
    rf"(?:19|20)\d{{2}}\b",
    flags=re.IGNORECASE,
)

QUARTER_PATTERN = re.compile(
    r"\b(?:q[1-4]|quarter\s*[1-4])"
    r"(?:\s+(?:19|20)\d{2})?\b",
    flags=re.IGNORECASE,
)

FINANCIAL_YEAR_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\s*[-/]\s*"
    r"(?:\d{2}|(?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)

YEAR_RANGE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\s*-\s*"
    r"(?:19|20)\d{2}\b",
    flags=re.IGNORECASE,
)

STANDALONE_YEAR_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b",
    flags=re.IGNORECASE,
)

VOLUME_SUFFIX_PATTERN = re.compile(
    r"\s*[:\-]\s*"
    r"(?:volume|vol\.?|part)\s+"
    r"(?:\d+|[ivxlcdm]+)\b.*$",
    flags=re.IGNORECASE,
)

TRAILING_SEPARATOR_PATTERN = re.compile(
    r"(?:\s*[-:;,/]\s*)+$"
)

NON_ALPHANUMERIC_PATTERN = re.compile(
    r"[^a-z0-9]+"
)

MULTIPLE_WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


@dataclass(frozen=True)
class RankedSearchResult:
    """One search result and its original relevance rank."""

    result: SearchResult
    rank: int


@dataclass(frozen=True)
class SearchResultGroup:
    """A standalone result or group of related datasets."""

    key: str
    display_title: str
    members: tuple[RankedSearchResult, ...]

    @property
    def best_member(self) -> RankedSearchResult:
        """Return the highest-ranked member."""

        if not self.members:
            raise RuntimeError(
                "Cannot access the best member of an "
                "empty result group."
            )

        return self.members[0]

    @property
    def is_series(self) -> bool:
        """Return whether the group contains multiple records."""

        return len(self.members) > 1


def compact_text(value: str) -> str:
    """Return compact single-line text."""

    return " ".join(value.split())


def normalise_group_value(value: str) -> str:
    """Create a stable value for title and organisation matching."""

    normalised = NON_ALPHANUMERIC_PATTERN.sub(
        " ",
        value.casefold(),
    )

    return MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        normalised,
    ).strip()


def remove_series_markers(
    title: str,
) -> tuple[str, bool]:
    """Remove conservative time and volume edition markers."""

    cleaned_title = compact_text(title)

    cleaned_title = (
        cleaned_title
        .replace("–", "-")
        .replace("—", "-")
    )

    original_title = cleaned_title

    volume_base_title = VOLUME_SUFFIX_PATTERN.sub(
        "",
        cleaned_title,
    )

    volume_base_title = TRAILING_SEPARATOR_PATTERN.sub(
        "",
        volume_base_title,
    ).strip()

    if (
        volume_base_title
        and volume_base_title.casefold()
        != cleaned_title.casefold()
    ):
        cleaned_title = volume_base_title

    for pattern in (
        MONTH_RANGE_PATTERN,
        QUARTER_PATTERN,
        FINANCIAL_YEAR_PATTERN,
        YEAR_RANGE_PATTERN,
        STANDALONE_YEAR_PATTERN,
    ):
        cleaned_title = pattern.sub(
            " ",
            cleaned_title,
        )

    cleaned_title = MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        cleaned_title,
    ).strip()

    cleaned_title = TRAILING_SEPARATOR_PATTERN.sub(
        "",
        cleaned_title,
    ).strip()

    marker_removed = (
        cleaned_title.casefold()
        != original_title.casefold()
    )

    return cleaned_title, marker_removed


def build_group_identity(
    ranked_result: RankedSearchResult,
) -> tuple[str, str]:
    """Create a conservative grouping identity."""

    result = ranked_result.result

    display_title, marker_removed = (
        remove_series_markers(
            result.title
        )
    )

    normalised_title = normalise_group_value(
        display_title
    )

    normalised_organisation = normalise_group_value(
        result.organisation
    )

    organisation_is_known = (
        bool(normalised_organisation)
        and normalised_organisation
        != "organisation not specified"
    )

    title_word_count = len(
        normalised_title.split()
    )

    can_group_as_series = (
        marker_removed
        and title_word_count >= 4
        and organisation_is_known
    )

    if can_group_as_series:
        return (
            (
                f"series::{normalised_organisation}"
                f"::{normalised_title}"
            ),
            display_title,
        )

    # Exact duplicate titles from the same organisation
    # may also be displayed as one group.
    if (
        title_word_count >= 4
        and organisation_is_known
    ):
        return (
            (
                f"exact::{normalised_organisation}"
                f"::{normalised_title}"
            ),
            display_title,
        )

    return (
        f"dataset::{result.dataset_id}",
        result.title,
    )


def group_search_results(
    results: Sequence[SearchResult],
) -> tuple[SearchResultGroup, ...]:
    """
    Group related editions without altering relevance order.

    A group's position is determined by its highest-ranked member.
    """

    members_by_key: dict[
        str,
        list[RankedSearchResult],
    ] = {}

    titles_by_key: dict[str, str] = {}

    for rank, result in enumerate(
        results,
        start=1,
    ):
        ranked_result = RankedSearchResult(
            result=result,
            rank=rank,
        )

        group_key, display_title = (
            build_group_identity(
                ranked_result
            )
        )

        members_by_key.setdefault(
            group_key,
            [],
        ).append(
            ranked_result
        )

        titles_by_key.setdefault(
            group_key,
            display_title,
        )

    groups: list[SearchResultGroup] = []

    for group_key, members in members_by_key.items():
        sorted_members = tuple(
            sorted(
                members,
                key=lambda member: member.rank,
            )
        )

        groups.append(
            SearchResultGroup(
                key=group_key,
                display_title=(
                    titles_by_key[group_key]
                ),
                members=sorted_members,
            )
        )

    groups.sort(
        key=lambda group: (
            group.best_member.rank
        )
    )

    return tuple(groups)