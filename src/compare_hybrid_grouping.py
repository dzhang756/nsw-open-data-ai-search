from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from src.search_engine import (
    SearchConfig,
    SearchEngine,
    SearchResult,
)

DEFAULT_CANDIDATE_COUNT = 50
DEFAULT_TOP_K = 10
DEFAULT_MAX_GROUP_MEMBERS = 10

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
    r"\b(?:19|20)\d{2}\s*[-–—]\s*"
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
class HybridCandidate:
    """One result in strict hybrid-ranking order."""

    result: SearchResult
    hybrid_rank: int


@dataclass
class ResultGroup:
    """A standalone result or related dataset series."""

    key: str
    display_title: str
    members: list[HybridCandidate] = field(
        default_factory=list
    )

    @property
    def best_member(self) -> HybridCandidate:
        """Return the highest-ranked member."""

        if not self.members:
            raise RuntimeError(
                "Cannot access the best member of an "
                "empty result group."
            )

        return self.members[0]

    @property
    def is_series(self) -> bool:
        """Return whether multiple records share the group."""

        return len(self.members) > 1


def compact_text(value: str) -> str:
    """Return compact single-line text."""

    return " ".join(value.split())


def remove_series_markers(
    title: str,
) -> tuple[str, bool]:
    """Remove conservative time or volume edition markers."""

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

    volume_base_title = (
        TRAILING_SEPARATOR_PATTERN.sub(
            "",
            volume_base_title,
        ).strip()
    )

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

    cleaned_title = (
        MULTIPLE_WHITESPACE_PATTERN.sub(
            " ",
            cleaned_title,
        ).strip()
    )

    cleaned_title = (
        TRAILING_SEPARATOR_PATTERN.sub(
            "",
            cleaned_title,
        ).strip()
    )

    marker_removed = (
        cleaned_title.casefold()
        != original_title.casefold()
    )

    return cleaned_title, marker_removed


def normalise_group_value(value: str) -> str:
    """Create a stable comparison value."""

    normalised = NON_ALPHANUMERIC_PATTERN.sub(
        " ",
        value.casefold(),
    )

    return MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        normalised,
    ).strip()


def build_group_identity(
    candidate: HybridCandidate,
) -> tuple[str, str]:
    """Create a conservative series-group identity."""

    result = candidate.result

    display_title, marker_removed = (
        remove_series_markers(
            result.title
        )
    )

    normalised_title = normalise_group_value(
        display_title
    )

    normalised_organisation = (
        normalise_group_value(
            result.organisation
        )
    )

    title_word_count = len(
        normalised_title.split()
    )

    organisation_is_known = (
        bool(normalised_organisation)
        and normalised_organisation
        != "organisation not specified"
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
    # may also be grouped.
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


def group_hybrid_results(
    candidates: list[HybridCandidate],
) -> list[ResultGroup]:
    """Group related records without changing hybrid order."""

    groups_by_key: dict[str, ResultGroup] = {}

    for candidate in candidates:
        group_key, display_title = (
            build_group_identity(candidate)
        )

        group = groups_by_key.get(
            group_key
        )

        if group is None:
            group = ResultGroup(
                key=group_key,
                display_title=display_title,
            )

            groups_by_key[group_key] = group

        group.members.append(candidate)

    groups = list(
        groups_by_key.values()
    )

    for group in groups:
        group.members.sort(
            key=lambda candidate: (
                candidate.hybrid_rank
            )
        )

    # Each group keeps the position of its highest-ranked member.
    groups.sort(
        key=lambda group: (
            group.best_member.hybrid_rank
        )
    )

    return groups


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Test strict hybrid ranking with conservative "
            "dataset-series grouping."
        )
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
        help=(
            "Number of hybrid candidates to retrieve before "
            "grouping "
            f"(default: {DEFAULT_CANDIDATE_COUNT})."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of visible result groups to display "
            f"(default: {DEFAULT_TOP_K})."
        ),
    )

    parser.add_argument(
        "--max-group-members",
        type=int,
        default=DEFAULT_MAX_GROUP_MEMBERS,
        help=(
            "Maximum records displayed inside each series "
            f"(default: {DEFAULT_MAX_GROUP_MEMBERS})."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Validate command-line values."""

    if arguments.candidate_count <= 0:
        raise ValueError(
            "candidate-count must be greater than zero."
        )

    if arguments.top_k <= 0:
        raise ValueError(
            "top-k must be greater than zero."
        )

    if arguments.max_group_members <= 0:
        raise ValueError(
            "max-group-members must be greater than zero."
        )


def print_candidate_details(
    candidate: HybridCandidate,
    indentation: str = "   ",
) -> None:
    """Print ranking and metadata for one result."""

    result = candidate.result

    keyword_rank = (
        str(result.keyword_rank)
        if result.keyword_rank is not None
        else "No positive keyword match"
    )

    print(
        f"{indentation}Hybrid rank: "
        f"{candidate.hybrid_rank}"
    )
    print(
        f"{indentation}Hybrid score: "
        f"{result.hybrid_score:.6f}"
    )
    print(
        f"{indentation}Semantic: "
        f"score={result.semantic_score:.4f}, "
        f"rank={result.semantic_rank}"
    )
    print(
        f"{indentation}Keyword: "
        f"score={result.keyword_score:.4f}, "
        f"rank={keyword_rank}"
    )
    print(
        f"{indentation}Organisation: "
        f"{result.organisation}"
    )

    if result.resource_formats:
        print(
            f"{indentation}Formats: "
            + ", ".join(
                result.resource_formats
            )
        )

    if result.dataset_url:
        print(
            f"{indentation}URL: "
            f"{result.dataset_url}"
        )


def print_group(
    group: ResultGroup,
    visible_rank: int,
    maximum_members: int,
) -> None:
    """Print one visible group or standalone result."""

    if not group.is_series:
        candidate = group.best_member

        print()
        print(
            f"{visible_rank}. "
            f"{candidate.result.title}"
        )

        print_candidate_details(
            candidate
        )

        return

    best_candidate = group.best_member

    print()
    print(
        f"{visible_rank}. [SERIES] "
        f"{group.display_title}"
    )
    print(
        f"   Best matching edition: "
        f"{best_candidate.result.title}"
    )
    print(
        f"   Grouped candidate records: "
        f"{len(group.members)}"
    )
    print(
        f"   Best hybrid rank: "
        f"{best_candidate.hybrid_rank}"
    )
    print(
        f"   Best hybrid score: "
        f"{best_candidate.result.hybrid_score:.6f}"
    )
    print(
        f"   Organisation: "
        f"{best_candidate.result.organisation}"
    )
    print("   Editions:")

    displayed_members = group.members[
        :maximum_members
    ]

    for member_number, candidate in enumerate(
        displayed_members,
        start=1,
    ):
        result = candidate.result

        print(
            f"      {member_number}. "
            f"{result.title}"
        )
        print(
            f"         Hybrid rank: "
            f"{candidate.hybrid_rank}"
        )
        print(
            f"         Hybrid score: "
            f"{result.hybrid_score:.6f}"
        )
        print(
            f"         Semantic rank: "
            f"{result.semantic_rank}"
        )

        keyword_rank = (
            str(result.keyword_rank)
            if result.keyword_rank is not None
            else "No positive keyword match"
        )

        print(
            f"         Keyword rank: "
            f"{keyword_rank}"
        )

        if result.resource_formats:
            print(
                "         Formats: "
                + ", ".join(
                    result.resource_formats
                )
            )

        if result.dataset_url:
            print(
                f"         URL: "
                f"{result.dataset_url}"
            )

    hidden_member_count = (
        len(group.members)
        - len(displayed_members)
    )

    if hidden_member_count > 0:
        print(
            f"      ... {hidden_member_count} "
            "additional grouped records not displayed"
        )


def main() -> None:
    """Retrieve strict hybrid results and group them."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    query = " ".join(
        arguments.query
    ).strip()

    if not query:
        raise ValueError(
            "A non-empty query is required."
        )

    print("Loading hybrid search engine...")

    engine = SearchEngine()

    # MMR is disabled so the result order reflects only the
    # hybrid relevance ranking.
    response = engine.search(
        query=query,
        config=SearchConfig(
            top_k=arguments.candidate_count,
            diversity_lambda=1.0,
            diversity_pool=(
                arguments.candidate_count
            ),
        ),
    )

    hybrid_candidates = [
        HybridCandidate(
            result=result,
            hybrid_rank=hybrid_rank,
        )
        for hybrid_rank, result in enumerate(
            response.results,
            start=1,
        )
    ]

    result_groups = group_hybrid_results(
        hybrid_candidates
    )

    visible_groups = result_groups[
        :arguments.top_k
    ]

    series_count = sum(
        1
        for group in visible_groups
        if group.is_series
    )

    grouped_record_count = sum(
        len(group.members)
        for group in visible_groups
        if group.is_series
    )

    print()
    print(f'Query: "{response.query}"')
    print(
        "Ranking: strict hybrid relevance order"
    )
    print("CrossEncoder: disabled")
    print("Diversification: disabled")
    print(
        "Series grouping: same organisation and "
        "conservative time/volume title matching"
    )
    print(
        f"Candidates retrieved: "
        f"{len(hybrid_candidates):,}"
    )
    print(
        f"Candidate groups identified: "
        f"{len(result_groups):,}"
    )
    print(
        f"Visible groups returned: "
        f"{len(visible_groups)}"
    )
    print(
        f"Visible series groups: "
        f"{series_count}"
    )
    print(
        f"Records inside visible series groups: "
        f"{grouped_record_count}"
    )
    print("=" * 80)

    for visible_rank, group in enumerate(
        visible_groups,
        start=1,
    ):
        print_group(
            group=group,
            visible_rank=visible_rank,
            maximum_members=(
                arguments.max_group_members
            ),
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()