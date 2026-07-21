from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from src.search_engine import (
    SearchConfig,
    SearchEngine,
    SearchResult,
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

DEFAULT_CANDIDATE_COUNT = 50
DEFAULT_TOP_K = 10
DEFAULT_MAX_GROUP_MEMBERS = 10

DESCRIPTION_CHARACTER_LIMIT = 2_000
MAX_TAGS = 20
MAX_GROUPS = 10
MAX_RESOURCE_NAMES = 8

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

STANDALONE_YEAR_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b",
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
class RerankedCandidate:
    """One candidate after CrossEncoder reranking."""

    result: SearchResult
    reranker_score: float
    original_hybrid_rank: int
    reranker_rank: int


@dataclass
class ResultGroup:
    """A standalone result or collection of series editions."""

    key: str
    display_title: str
    members: list[RerankedCandidate] = field(
        default_factory=list
    )

    @property
    def best_member(self) -> RerankedCandidate:
        """Return the strongest result in the group."""

        if not self.members:
            raise RuntimeError(
                "Cannot access the best member of an "
                "empty result group."
            )

        return self.members[0]

    @property
    def is_series(self) -> bool:
        """Return whether this group contains multiple records."""

        return len(self.members) > 1


def compact_text(value: Any) -> str:
    """Return compact single-line text."""

    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def extract_named_values(
    values: Any,
    maximum_values: int,
) -> list[str]:
    """Extract display names from strings or dictionaries."""

    if not isinstance(values, list):
        return []

    extracted: list[str] = []
    seen: set[str] = set()

    for value in values:
        if isinstance(value, str):
            text = compact_text(value)

        elif isinstance(value, dict):
            text = compact_text(
                value.get("title")
                or value.get("display_name")
                or value.get("name")
            )

        else:
            text = ""

        normalised = text.casefold()

        if not text or normalised in seen:
            continue

        seen.add(normalised)
        extracted.append(text)

        if len(extracted) >= maximum_values:
            break

    return extracted


def organisation_text(
    dataset: dict[str, Any],
) -> str:
    """Return the organisation display name."""

    organisation = dataset.get("organisation")

    if isinstance(organisation, str):
        return compact_text(organisation)

    if not isinstance(organisation, dict):
        return ""

    return compact_text(
        organisation.get("title")
        or organisation.get("name")
    )


def resource_names(
    dataset: dict[str, Any],
) -> list[str]:
    """Extract a limited set of resource names."""

    resources = dataset.get("resources")

    if not isinstance(resources, list):
        return []

    names: list[str] = []
    seen: set[str] = set()

    for resource in resources:
        if not isinstance(resource, dict):
            continue

        name = compact_text(
            resource.get("name")
        )

        normalised = name.casefold()

        if not name or normalised in seen:
            continue

        seen.add(normalised)
        names.append(name)

        if len(names) >= MAX_RESOURCE_NAMES:
            break

    return names


def build_reranker_document(
    dataset: dict[str, Any],
) -> str:
    """Build metadata text for query-document reranking."""

    title = compact_text(
        dataset.get("title")
    )

    organisation = organisation_text(
        dataset
    )

    tags = extract_named_values(
        dataset.get("tags"),
        maximum_values=MAX_TAGS,
    )

    groups = extract_named_values(
        dataset.get("groups"),
        maximum_values=MAX_GROUPS,
    )

    formats = extract_named_values(
        dataset.get("resource_formats"),
        maximum_values=50,
    )

    names = resource_names(dataset)

    description = compact_text(
        dataset.get("description")
    )

    if len(description) > DESCRIPTION_CHARACTER_LIMIT:
        description = description[
            :DESCRIPTION_CHARACTER_LIMIT
        ].rstrip()

    sections = [
        f"Title: {title}",
    ]

    if organisation:
        sections.append(
            f"Organisation: {organisation}"
        )

    if tags:
        sections.append(
            "Tags: " + ", ".join(tags)
        )

    if groups:
        sections.append(
            "Categories: " + ", ".join(groups)
        )

    if formats:
        sections.append(
            "Formats: " + ", ".join(formats)
        )

    if names:
        sections.append(
            "Resources: " + "; ".join(names)
        )

    if description:
        sections.append(
            f"Description: {description}"
        )

    return "\n".join(sections)


def remove_time_edition_markers(
    title: str,
) -> tuple[str, bool]:
    """Remove obvious reporting-period markers from a title."""

    cleaned_title = compact_text(title)

    cleaned_title = (
        cleaned_title
        .replace("–", "-")
        .replace("—", "-")
    )

    original_title = cleaned_title

    for pattern in (
        MONTH_RANGE_PATTERN,
        QUARTER_PATTERN,
        FINANCIAL_YEAR_PATTERN,
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


def normalise_series_title(
    title: str,
) -> tuple[str, str, bool]:
    """Create display and comparison values for series grouping."""

    display_title, marker_removed = (
        remove_time_edition_markers(title)
    )

    normalised_title = (
        NON_ALPHANUMERIC_PATTERN.sub(
            " ",
            display_title.casefold(),
        )
    )

    normalised_title = MULTIPLE_WHITESPACE_PATTERN.sub(
        " ",
        normalised_title,
    ).strip()

    return (
        display_title,
        normalised_title,
        marker_removed,
    )


def build_group_identity(
    candidate: RerankedCandidate,
) -> tuple[str, str]:
    """Build a conservative result-group key and title."""

    result = candidate.result

    (
        display_title,
        normalised_title,
        marker_removed,
    ) = normalise_series_title(
        result.title
    )

    normalised_organisation = (
        NON_ALPHANUMERIC_PATTERN.sub(
            " ",
            result.organisation.casefold(),
        )
    )

    normalised_organisation = (
        MULTIPLE_WHITESPACE_PATTERN.sub(
            " ",
            normalised_organisation,
        ).strip()
    )

    title_word_count = len(
        normalised_title.split()
    )

    # Time-based grouping is only allowed when the remaining
    # title is sufficiently descriptive. This prevents generic
    # titles such as "Annual Report 2024" from being grouped too
    # broadly.
    can_group_as_series = (
        marker_removed
        and title_word_count >= 4
        and bool(normalised_organisation)
    )

    if can_group_as_series:
        key = (
            f"series::{normalised_organisation}"
            f"::{normalised_title}"
        )

        return key, display_title

    # Exact duplicate titles from the same organisation may
    # still be grouped, but otherwise the dataset remains unique.
    if (
        title_word_count >= 4
        and bool(normalised_organisation)
    ):
        key = (
            f"exact::{normalised_organisation}"
            f"::{normalised_title}"
        )

        return key, display_title

    return (
        f"dataset::{result.dataset_id}",
        result.title,
    )


def group_reranked_results(
    candidates: list[RerankedCandidate],
) -> list[ResultGroup]:
    """Group conservative dataset series without changing rank."""

    groups_by_key: dict[str, ResultGroup] = {}

    for candidate in candidates:
        group_key, display_title = (
            build_group_identity(candidate)
        )

        group = groups_by_key.get(group_key)

        if group is None:
            group = ResultGroup(
                key=group_key,
                display_title=display_title,
            )

            groups_by_key[group_key] = group

        group.members.append(candidate)

    groups = list(groups_by_key.values())

    for group in groups:
        group.members.sort(
            key=lambda candidate: (
                candidate.reranker_rank,
                candidate.original_hybrid_rank,
            )
        )

    groups.sort(
        key=lambda group: (
            group.best_member.reranker_rank,
            group.best_member.original_hybrid_rank,
        )
    )

    return groups


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare hybrid retrieval with CrossEncoder "
            "reranking and conservative dataset-series "
            "grouping."
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
            "Number of first-stage candidates to rerank "
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
            "Maximum number of records displayed inside "
            "each grouped series "
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
    candidate: RerankedCandidate,
    indentation: str = "   ",
) -> None:
    """Print scores and metadata for one candidate."""

    result = candidate.result

    print(
        f"{indentation}Original hybrid rank: "
        f"{candidate.original_hybrid_rank}"
    )
    print(
        f"{indentation}Reranker rank: "
        f"{candidate.reranker_rank}"
    )
    print(
        f"{indentation}Reranker score: "
        f"{candidate.reranker_score:.6f}"
    )
    print(
        f"{indentation}Original hybrid score: "
        f"{result.hybrid_score:.6f}"
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

        print_candidate_details(candidate)
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
        f"   Best reranker score: "
        f"{best_candidate.reranker_score:.6f}"
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
            f"         Reranker rank: "
            f"{candidate.reranker_rank}"
        )
        print(
            f"         Reranker score: "
            f"{candidate.reranker_score:.6f}"
        )
        print(
            f"         Hybrid rank: "
            f"{candidate.original_hybrid_rank}"
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
    """Retrieve, rerank and group search results."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    query = " ".join(
        arguments.query
    ).strip()

    if not query:
        raise ValueError(
            "A non-empty query is required."
        )

    print(
        "Loading first-stage search engine..."
    )

    engine = SearchEngine()

    # First-stage diversification is disabled. The strongest
    # hybrid candidates are passed directly to the reranker.
    first_stage_response = engine.search(
        query=query,
        config=SearchConfig(
            top_k=arguments.candidate_count,
            diversity_lambda=1.0,
            diversity_pool=(
                arguments.candidate_count
            ),
        ),
    )

    if not first_stage_response.results:
        print(
            "No first-stage results returned."
        )
        return

    candidate_documents: list[str] = []
    original_ranks: dict[str, int] = {}

    for original_rank, result in enumerate(
        first_stage_response.results,
        start=1,
    ):
        original_ranks[result.dataset_id] = (
            original_rank
        )

        dataset = engine.metadata_by_id[
            result.dataset_id
        ]

        candidate_documents.append(
            build_reranker_document(dataset)
        )

    print(
        f"Loading reranker: "
        f"{RERANKER_MODEL_NAME}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME,
        max_length=512,
    )

    pairs = [
        (query, document)
        for document in candidate_documents
    ]

    print(
        f"Reranking {len(pairs):,} candidates..."
    )

    reranker_scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    reranker_scores = np.asarray(
        reranker_scores,
        dtype=np.float32,
    ).reshape(-1)

    if len(reranker_scores) != len(
        first_stage_response.results
    ):
        raise RuntimeError(
            "Reranker score count does not match "
            "the candidate count."
        )

    scored_results: list[
        tuple[SearchResult, float]
    ] = list(
        zip(
            first_stage_response.results,
            reranker_scores,
            strict=True,
        )
    )

    scored_results.sort(
        key=lambda value: (
            value[1],
            value[0].hybrid_score,
        ),
        reverse=True,
    )

    reranked_candidates: list[
        RerankedCandidate
    ] = []

    for reranker_rank, (
        result,
        reranker_score,
    ) in enumerate(
        scored_results,
        start=1,
    ):
        reranked_candidates.append(
            RerankedCandidate(
                result=result,
                reranker_score=float(
                    reranker_score
                ),
                original_hybrid_rank=(
                    original_ranks[
                        result.dataset_id
                    ]
                ),
                reranker_rank=reranker_rank,
            )
        )

    result_groups = group_reranked_results(
        reranked_candidates
    )

    visible_groups = result_groups[
        :arguments.top_k
    ]

    grouped_record_count = sum(
        len(group.members)
        for group in visible_groups
        if group.is_series
    )

    series_count = sum(
        1
        for group in visible_groups
        if group.is_series
    )

    print()
    print(f'Query: "{query}"')
    print(
        f"First-stage candidates: "
        f"{len(reranked_candidates):,}"
    )
    print(
        f"Reranker: {RERANKER_MODEL_NAME}"
    )
    print(
        "Diversification: disabled"
    )
    print(
        "Series grouping: same organisation and "
        "conservative time-edition title matching"
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