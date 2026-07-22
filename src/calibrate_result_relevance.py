from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from src.search_engine import (
    SearchConfig,
    SearchEngine,
    SearchResult,
)


MAX_RESULTS = 1_000
CANDIDATE_POOL = 1_000

FULL_RESULTS_OUTPUT_PATH = Path(
    "outputs/tables/result_relevance_full.csv"
)

CHECKPOINT_OUTPUT_PATH = Path(
    "outputs/tables/result_relevance_checkpoints.csv"
)

BUCKET_OUTPUT_PATH = Path(
    "outputs/tables/result_relevance_buckets.csv"
)

RULE_SUMMARY_OUTPUT_PATH = Path(
    "outputs/tables/result_relevance_rule_summary.csv"
)

CHECKPOINT_RANKS = (
    1,
    5,
    10,
    20,
    30,
    40,
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    750,
    1_000,
)

RANK_BUCKETS = (
    (1, 10),
    (11, 25),
    (26, 50),
    (51, 100),
    (101, 200),
    (201, 300),
    (301, 500),
    (501, 750),
    (751, 1_000),
)


@dataclass(frozen=True)
class CalibrationQuery:
    """One representative catalogue query."""

    query_type: str
    query: str


@dataclass(frozen=True)
class RetentionRule:
    """
    One possible result-level relevance rule.

    A result is retained when either its semantic score
    passes the semantic floor or its keyword score passes
    the keyword floor.
    """

    name: str
    absolute_semantic_floor: float
    relative_semantic_floor: float
    absolute_keyword_floor: float
    relative_keyword_floor: float


CALIBRATION_QUERIES = (
    # Narrow and clearly defined searches.
    CalibrationQuery(
        query_type="narrow",
        query="road crash data",
    ),
    CalibrationQuery(
        query_type="narrow",
        query="ambulance off stretcher times",
    ),
    CalibrationQuery(
        query_type="narrow",
        query="koala habitat corridors",
    ),
    CalibrationQuery(
        query_type="narrow",
        query="hospital waiting times",
    ),
    CalibrationQuery(
        query_type="narrow",
        query="water quality monitoring",
    ),

    # Broader searches that should retain more results.
    CalibrationQuery(
        query_type="broad",
        query="transport",
    ),
    CalibrationQuery(
        query_type="broad",
        query="health",
    ),
    CalibrationQuery(
        query_type="broad",
        query="environment",
    ),
    CalibrationQuery(
        query_type="broad",
        query="housing",
    ),

    # Longer natural-language searches.
    CalibrationQuery(
        query_type="natural_language",
        query=(
            "Sydney public transport passenger data"
        ),
    ),
    CalibrationQuery(
        query_type="natural_language",
        query=(
            "bushfire risk management plans"
        ),
    ),

    # Meaningful searches containing spelling errors.
    CalibrationQuery(
        query_type="misspelled",
        query=(
            "emergncy departmnt wait times"
        ),
    ),
)

RETENTION_RULES = (
    RetentionRule(
        name="strict",
        absolute_semantic_floor=0.30,
        relative_semantic_floor=0.45,
        absolute_keyword_floor=0.08,
        relative_keyword_floor=0.25,
    ),
    RetentionRule(
        name="balanced",
        absolute_semantic_floor=0.25,
        relative_semantic_floor=0.40,
        absolute_keyword_floor=0.05,
        relative_keyword_floor=0.20,
    ),
    RetentionRule(
        name="permissive",
        absolute_semantic_floor=0.20,
        relative_semantic_floor=0.35,
        absolute_keyword_floor=0.03,
        relative_keyword_floor=0.15,
    ),
)


def safe_score(
    result: SearchResult,
    attribute_name: str,
) -> float:
    """Return a result score as a safe float."""

    value = getattr(
        result,
        attribute_name,
        0.0,
    )

    if value is None:
        return 0.0

    return float(value)


def safe_ratio(
    score: float,
    best_score: float,
) -> float:
    """Return a score as a proportion of the best score."""

    if best_score <= 0.0:
        return 0.0

    return score / best_score


def compact_text(
    value: str,
    maximum_length: int = 240,
) -> str:
    """Create a compact single-line description preview."""

    compact_value = " ".join(
        value.split()
    )

    if len(compact_value) <= maximum_length:
        return compact_value

    shortened = compact_value[
        :maximum_length + 1
    ]

    final_space = shortened.rfind(" ")

    if final_space >= maximum_length * 0.75:
        shortened = shortened[
            :final_space
        ]

    else:
        shortened = shortened[
            :maximum_length
        ]

    return (
        shortened.rstrip(" ,.;:-")
        + "…"
    )


def strongest_score(
    results: tuple[SearchResult, ...],
    attribute_name: str,
) -> float:
    """Return the strongest selected score."""

    if not results:
        return 0.0

    return max(
        safe_score(
            result,
            attribute_name,
        )
        for result in results
    )


def rule_thresholds(
    rule: RetentionRule,
    best_semantic_score: float,
    best_keyword_score: float,
) -> tuple[float, float]:
    """Calculate dynamic semantic and keyword floors."""

    semantic_floor = max(
        rule.absolute_semantic_floor,
        (
            best_semantic_score
            * rule.relative_semantic_floor
        ),
    )

    keyword_floor = max(
        rule.absolute_keyword_floor,
        (
            best_keyword_score
            * rule.relative_keyword_floor
        ),
    )

    return (
        semantic_floor,
        keyword_floor,
    )


def result_passes_rule(
    result: SearchResult,
    rule: RetentionRule,
    best_semantic_score: float,
    best_keyword_score: float,
) -> bool:
    """Return whether a result passes a candidate rule."""

    (
        semantic_floor,
        keyword_floor,
    ) = rule_thresholds(
        rule=rule,
        best_semantic_score=best_semantic_score,
        best_keyword_score=best_keyword_score,
    )

    semantic_score = safe_score(
        result,
        "semantic_score",
    )

    keyword_score = safe_score(
        result,
        "keyword_score",
    )

    return (
        semantic_score >= semantic_floor
        or keyword_score >= keyword_floor
    )


def create_full_result_row(
    calibration_query: CalibrationQuery,
    result: SearchResult,
    rank: int,
    best_semantic_score: float,
    best_keyword_score: float,
    recognised_keyword_features: int,
) -> dict[str, object]:
    """Create one detailed output row."""

    semantic_score = safe_score(
        result,
        "semantic_score",
    )

    keyword_score = safe_score(
        result,
        "keyword_score",
    )

    hybrid_score = safe_score(
        result,
        "hybrid_score",
    )

    row: dict[str, object] = {
        "query_type": calibration_query.query_type,
        "query": calibration_query.query,
        "recognised_keyword_features": (
            recognised_keyword_features
        ),
        "rank": rank,
        "title": result.title,
        "organisation": result.organisation,
        "description_preview": compact_text(
            result.description
        ),
        "dataset_url": result.dataset_url,
        "semantic_score": semantic_score,
        "semantic_ratio_to_best": safe_ratio(
            semantic_score,
            best_semantic_score,
        ),
        "keyword_score": keyword_score,
        "keyword_ratio_to_best": safe_ratio(
            keyword_score,
            best_keyword_score,
        ),
        "hybrid_score": hybrid_score,
    }

    for rule in RETENTION_RULES:
        row[
            f"passes_{rule.name}_rule"
        ] = result_passes_rule(
            result=result,
            rule=rule,
            best_semantic_score=(
                best_semantic_score
            ),
            best_keyword_score=(
                best_keyword_score
            ),
        )

    return row


def create_checkpoint_rows(
    full_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Select rows at important result ranks."""

    checkpoint_rank_set = set(
        CHECKPOINT_RANKS
    )

    return [
        row
        for row in full_rows
        if int(row["rank"])
        in checkpoint_rank_set
    ]


def bucket_values(
    full_rows: list[dict[str, object]],
    start_rank: int,
    end_rank: int,
) -> list[dict[str, object]]:
    """Return rows within a rank interval."""

    return [
        row
        for row in full_rows
        if (
            start_rank
            <= int(row["rank"])
            <= end_rank
        )
    ]


def numeric_values(
    rows: list[dict[str, object]],
    field_name: str,
) -> list[float]:
    """Extract numeric values from output rows."""

    return [
        float(row[field_name])
        for row in rows
    ]


def create_bucket_rows(
    calibration_query: CalibrationQuery,
    full_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarise scores within result-rank buckets."""

    bucket_rows: list[
        dict[str, object]
    ] = []

    for start_rank, end_rank in RANK_BUCKETS:
        rows = bucket_values(
            full_rows=full_rows,
            start_rank=start_rank,
            end_rank=end_rank,
        )

        if not rows:
            continue

        semantic_scores = numeric_values(
            rows,
            "semantic_score",
        )

        keyword_scores = numeric_values(
            rows,
            "keyword_score",
        )

        semantic_ratios = numeric_values(
            rows,
            "semantic_ratio_to_best",
        )

        keyword_ratios = numeric_values(
            rows,
            "keyword_ratio_to_best",
        )

        bucket_row: dict[str, object] = {
            "query_type": (
                calibration_query.query_type
            ),
            "query": calibration_query.query,
            "rank_from": start_rank,
            "rank_to": end_rank,
            "results_in_bucket": len(rows),
            "semantic_score_max": max(
                semantic_scores
            ),
            "semantic_score_mean": mean(
                semantic_scores
            ),
            "semantic_score_median": median(
                semantic_scores
            ),
            "semantic_score_min": min(
                semantic_scores
            ),
            "semantic_ratio_mean": mean(
                semantic_ratios
            ),
            "semantic_ratio_median": median(
                semantic_ratios
            ),
            "keyword_score_max": max(
                keyword_scores
            ),
            "keyword_score_mean": mean(
                keyword_scores
            ),
            "keyword_score_median": median(
                keyword_scores
            ),
            "keyword_score_min": min(
                keyword_scores
            ),
            "keyword_ratio_mean": mean(
                keyword_ratios
            ),
            "keyword_ratio_median": median(
                keyword_ratios
            ),
        }

        for rule in RETENTION_RULES:
            field_name = (
                f"passes_{rule.name}_rule"
            )

            retained_count = sum(
                bool(row[field_name])
                for row in rows
            )

            bucket_row[
                f"{rule.name}_retained"
            ] = retained_count

        bucket_rows.append(
            bucket_row
        )

    return bucket_rows


def create_rule_summary_rows(
    calibration_query: CalibrationQuery,
    results: tuple[SearchResult, ...],
    best_semantic_score: float,
    best_keyword_score: float,
) -> list[dict[str, object]]:
    """Summarise how many results each rule retains."""

    summary_rows: list[
        dict[str, object]
    ] = []

    for rule in RETENTION_RULES:
        (
            semantic_floor,
            keyword_floor,
        ) = rule_thresholds(
            rule=rule,
            best_semantic_score=(
                best_semantic_score
            ),
            best_keyword_score=(
                best_keyword_score
            ),
        )

        retained_results = [
            result
            for result in results
            if result_passes_rule(
                result=result,
                rule=rule,
                best_semantic_score=(
                    best_semantic_score
                ),
                best_keyword_score=(
                    best_keyword_score
                ),
            )
        ]

        final_retained_rank = (
            max(
                (
                    rank
                    for rank, result in enumerate(
                        results,
                        start=1,
                    )
                    if result_passes_rule(
                        result=result,
                        rule=rule,
                        best_semantic_score=(
                            best_semantic_score
                        ),
                        best_keyword_score=(
                            best_keyword_score
                        ),
                    )
                ),
                default=0,
            )
        )

        summary_rows.append(
            {
                "query_type": (
                    calibration_query.query_type
                ),
                "query": (
                    calibration_query.query
                ),
                "rule": rule.name,
                "semantic_floor": semantic_floor,
                "keyword_floor": keyword_floor,
                "results_examined": len(results),
                "results_retained": len(
                    retained_results
                ),
                "retention_percentage": (
                    (
                        len(retained_results)
                        / len(results)
                        * 100
                    )
                    if results
                    else 0.0
                ),
                "last_retained_rank": (
                    final_retained_rank
                ),
            }
        )

    return summary_rows


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionary rows to a CSV file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            f"No rows were produced for {output_path}."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_query_report(
    calibration_query: CalibrationQuery,
    results: tuple[SearchResult, ...],
    full_rows: list[dict[str, object]],
    recognised_keyword_features: int,
    best_semantic_score: float,
    best_keyword_score: float,
) -> None:
    """Print checkpoints and candidate-rule counts."""

    print()
    print("=" * 100)
    print(
        f"Query type: "
        f"{calibration_query.query_type}"
    )
    print(
        f"Query: "
        f"{calibration_query.query!r}"
    )
    print(
        f"Results returned: "
        f"{len(results):,}"
    )
    print(
        "Recognised keyword features: "
        f"{recognised_keyword_features}"
    )
    print(
        "Best semantic score: "
        f"{best_semantic_score:.4f}"
    )
    print(
        "Best keyword score: "
        f"{best_keyword_score:.4f}"
    )
    print("-" * 100)

    for rule in RETENTION_RULES:
        (
            semantic_floor,
            keyword_floor,
        ) = rule_thresholds(
            rule=rule,
            best_semantic_score=(
                best_semantic_score
            ),
            best_keyword_score=(
                best_keyword_score
            ),
        )

        retained_count = sum(
            result_passes_rule(
                result=result,
                rule=rule,
                best_semantic_score=(
                    best_semantic_score
                ),
                best_keyword_score=(
                    best_keyword_score
                ),
            )
            for result in results
        )

        print(
            f"{rule.name.title()} rule: "
            f"{retained_count:,} retained "
            f"| semantic floor "
            f"{semantic_floor:.4f} "
            f"| keyword floor "
            f"{keyword_floor:.4f}"
        )

    print("-" * 100)
    print("Checkpoint results:")

    checkpoint_rows = create_checkpoint_rows(
        full_rows
    )

    for row in checkpoint_rows:
        title = str(
            row["title"]
        )

        if len(title) > 78:
            title = (
                title[:75].rstrip()
                + "…"
            )

        print(
            f"Rank {int(row['rank']):>4}: "
            f"semantic={float(row['semantic_score']):.4f} "
            f"({float(row['semantic_ratio_to_best']):.2%}) "
            f"| keyword={float(row['keyword_score']):.4f} "
            f"({float(row['keyword_ratio_to_best']):.2%}) "
            f"| {title}"
        )


def main() -> None:
    """Run result-level relevance calibration."""

    print(
        "Loading search engine..."
    )

    engine = SearchEngine()

    config = SearchConfig(
        top_k=MAX_RESULTS,
        candidate_pool=CANDIDATE_POOL,
        diversity_lambda=1.0,
        diversity_pool=MAX_RESULTS,
    ).validated()

    all_full_rows: list[
        dict[str, object]
    ] = []

    all_checkpoint_rows: list[
        dict[str, object]
    ] = []

    all_bucket_rows: list[
        dict[str, object]
    ] = []

    all_rule_summary_rows: list[
        dict[str, object]
    ] = []

    for calibration_query in CALIBRATION_QUERIES:
        response = engine.search(
            query=calibration_query.query,
            config=config,
        )

        results = response.results

        recognised_keyword_features = int(
            getattr(
                response,
                "keyword_query_feature_count",
                0,
            )
        )

        best_semantic_score = strongest_score(
            results=results,
            attribute_name="semantic_score",
        )

        best_keyword_score = strongest_score(
            results=results,
            attribute_name="keyword_score",
        )

        query_full_rows = [
            create_full_result_row(
                calibration_query=(
                    calibration_query
                ),
                result=result,
                rank=rank,
                best_semantic_score=(
                    best_semantic_score
                ),
                best_keyword_score=(
                    best_keyword_score
                ),
                recognised_keyword_features=(
                    recognised_keyword_features
                ),
            )
            for rank, result in enumerate(
                results,
                start=1,
            )
        ]

        query_checkpoint_rows = (
            create_checkpoint_rows(
                query_full_rows
            )
        )

        query_bucket_rows = (
            create_bucket_rows(
                calibration_query=(
                    calibration_query
                ),
                full_rows=query_full_rows,
            )
        )

        query_rule_summary_rows = (
            create_rule_summary_rows(
                calibration_query=(
                    calibration_query
                ),
                results=results,
                best_semantic_score=(
                    best_semantic_score
                ),
                best_keyword_score=(
                    best_keyword_score
                ),
            )
        )

        all_full_rows.extend(
            query_full_rows
        )

        all_checkpoint_rows.extend(
            query_checkpoint_rows
        )

        all_bucket_rows.extend(
            query_bucket_rows
        )

        all_rule_summary_rows.extend(
            query_rule_summary_rows
        )

        print_query_report(
            calibration_query=(
                calibration_query
            ),
            results=results,
            full_rows=query_full_rows,
            recognised_keyword_features=(
                recognised_keyword_features
            ),
            best_semantic_score=(
                best_semantic_score
            ),
            best_keyword_score=(
                best_keyword_score
            ),
        )

    write_csv(
        output_path=FULL_RESULTS_OUTPUT_PATH,
        rows=all_full_rows,
    )

    write_csv(
        output_path=CHECKPOINT_OUTPUT_PATH,
        rows=all_checkpoint_rows,
    )

    write_csv(
        output_path=BUCKET_OUTPUT_PATH,
        rows=all_bucket_rows,
    )

    write_csv(
        output_path=RULE_SUMMARY_OUTPUT_PATH,
        rows=all_rule_summary_rows,
    )

    print()
    print("=" * 100)
    print("Calibration complete.")
    print(
        "Full ranked results: "
        f"{FULL_RESULTS_OUTPUT_PATH}"
    )
    print(
        "Checkpoint results: "
        f"{CHECKPOINT_OUTPUT_PATH}"
    )
    print(
        "Rank-bucket summaries: "
        f"{BUCKET_OUTPUT_PATH}"
    )
    print(
        "Candidate-rule summaries: "
        f"{RULE_SUMMARY_OUTPUT_PATH}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()