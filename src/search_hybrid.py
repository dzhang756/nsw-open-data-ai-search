from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search_keyword import (
    load_keyword_records,
    load_matrix,
    load_vectorizer,
)
from src.search_semantic import (
    get_organisation_title,
    load_catalogue_metadata,
    load_embedding_records,
    load_embeddings,
    shorten_text,
)

EMBEDDING_MANIFEST_PATH = Path(
    "data/index/embedding_manifest.json"
)
KEYWORD_MANIFEST_PATH = Path(
    "data/index/keyword_manifest.json"
)

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_POOL = 200
DEFAULT_SEMANTIC_WEIGHT = 0.70
DEFAULT_KEYWORD_WEIGHT = 0.30
DEFAULT_RRF_K = 60
DEFAULT_DIVERSITY_LAMBDA = 0.85
DEFAULT_DIVERSITY_POOL = 100


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_indexes(
    embeddings: np.ndarray,
    embedding_records: list[dict[str, Any]],
    keyword_matrix: sparse.csr_matrix,
    keyword_records: list[dict[str, Any]],
    embedding_manifest: dict[str, Any],
    keyword_manifest: dict[str, Any],
) -> None:
    """Confirm that both indexes describe the same datasets."""

    embedding_count = embedding_manifest.get("dataset_count")
    keyword_count = keyword_manifest.get("dataset_count")

    if embeddings.shape[0] != embedding_count:
        raise RuntimeError(
            "Embedding matrix row count does not match its "
            f"manifest: {embeddings.shape[0]} != "
            f"{embedding_count}"
        )

    if keyword_matrix.shape[0] != keyword_count:
        raise RuntimeError(
            "Keyword matrix row count does not match its "
            f"manifest: {keyword_matrix.shape[0]} != "
            f"{keyword_count}"
        )

    if len(embedding_records) != embeddings.shape[0]:
        raise RuntimeError(
            "Embedding record count does not match the "
            "embedding matrix."
        )

    if len(keyword_records) != keyword_matrix.shape[0]:
        raise RuntimeError(
            "Keyword record count does not match the "
            "keyword matrix."
        )

    if embeddings.shape[0] != keyword_matrix.shape[0]:
        raise RuntimeError(
            "Semantic and keyword indexes contain different "
            "numbers of datasets."
        )

    embedding_ids = [
        record["dataset_id"]
        for record in embedding_records
    ]
    keyword_ids = [
        record["dataset_id"]
        for record in keyword_records
    ]

    if embedding_ids != keyword_ids:
        raise RuntimeError(
            "Semantic and keyword indexes do not use the same "
            "dataset row order."
        )


def build_rank_array(
    ordered_indices: np.ndarray,
    total_rows: int,
) -> np.ndarray:
    """Convert ordered row indices into one-based ranks."""

    ranks = np.zeros(total_rows, dtype=np.int32)

    ranks[ordered_indices] = np.arange(
        1,
        len(ordered_indices) + 1,
        dtype=np.int32,
    )

    return ranks


def calculate_semantic_scores(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Calculate semantic similarity against every dataset."""

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_embedding.shape != (embeddings.shape[1],):
        raise RuntimeError(
            "Unexpected query embedding shape: "
            f"{query_embedding.shape}"
        )

    return np.asarray(
        embeddings @ query_embedding,
        dtype=np.float32,
    )


def calculate_keyword_scores(
    query: str,
    vectorizer: TfidfVectorizer,
    keyword_matrix: sparse.csr_matrix,
) -> np.ndarray:
    """Calculate lexical similarity against every dataset."""

    query_vector = vectorizer.transform([query])
    query_vector = sparse.csr_matrix(
        query_vector,
        dtype=np.float32,
    )

    if query_vector.nnz == 0:
        return np.zeros(
            keyword_matrix.shape[0],
            dtype=np.float32,
        )

    scores = (
        keyword_matrix @ query_vector.transpose()
    ).toarray().ravel()

    return np.asarray(scores, dtype=np.float32)


def create_hybrid_ranking(
    semantic_scores: np.ndarray,
    keyword_scores: np.ndarray,
    semantic_weight: float,
    keyword_weight: float,
    rrf_k: int,
    candidate_pool: int,
) -> list[tuple[int, float, int, int]]:
    """
    Combine semantic and keyword rankings using weighted
    Reciprocal Rank Fusion.

    Returns:
        row index,
        hybrid score,
        semantic rank,
        keyword rank
    """

    total_rows = len(semantic_scores)
    pool_size = min(candidate_pool, total_rows)

    semantic_order = np.argsort(
        semantic_scores,
        kind="stable",
    )[::-1]

    semantic_ranks = build_rank_array(
        semantic_order,
        total_rows,
    )

    positive_keyword_indices = np.flatnonzero(
        keyword_scores > 0
    )

    keyword_order = positive_keyword_indices[
        np.argsort(
            keyword_scores[positive_keyword_indices],
            kind="stable",
        )[::-1]
    ]

    keyword_ranks = build_rank_array(
        keyword_order,
        total_rows,
    )

    candidate_indices = set(
        int(index)
        for index in semantic_order[:pool_size]
    )

    candidate_indices.update(
        int(index)
        for index in keyword_order[:pool_size]
    )

    ranked_candidates: list[
        tuple[int, float, int, int]
    ] = []

    for row_index in candidate_indices:
        semantic_rank = int(
            semantic_ranks[row_index]
        )
        keyword_rank = int(
            keyword_ranks[row_index]
        )

        hybrid_score = (
            semantic_weight
            / (rrf_k + semantic_rank)
        )

        if keyword_rank > 0:
            hybrid_score += (
                keyword_weight
                / (rrf_k + keyword_rank)
            )

        ranked_candidates.append(
            (
                row_index,
                hybrid_score,
                semantic_rank,
                keyword_rank,
            )
        )

    ranked_candidates.sort(
        key=lambda result: (
            result[1],
            semantic_scores[result[0]],
            keyword_scores[result[0]],
        ),
        reverse=True,
    )

    return ranked_candidates

def select_diverse_results(
    ranked_results: list[tuple[int, float, int, int]],
    embeddings: np.ndarray,
    result_count: int,
    diversity_pool: int,
    diversity_lambda: float,
) -> list[tuple[int, float, int, int]]:
    """
    Select relevant but non-repetitive results using Maximal
    Marginal Relevance.

    The first result is the strongest hybrid result. Each later
    result balances hybrid relevance against semantic similarity
    to results already selected.
    """

    if result_count <= 0 or not ranked_results:
        return []

    pool_size = min(
        max(diversity_pool, result_count),
        len(ranked_results),
    )

    candidates = ranked_results[:pool_size]

    if diversity_lambda >= 1.0:
        return candidates[:result_count]

    row_indices = np.asarray(
        [
            result[0]
            for result in candidates
        ],
        dtype=np.int64,
    )

    candidate_vectors = np.asarray(
        embeddings[row_indices],
        dtype=np.float32,
    )

    hybrid_scores = np.asarray(
        [
            result[1]
            for result in candidates
        ],
        dtype=np.float32,
    )

    minimum_score = float(hybrid_scores.min())
    maximum_score = float(hybrid_scores.max())
    score_range = maximum_score - minimum_score

    if score_range > 0:
        relevance_scores = (
            hybrid_scores - minimum_score
        ) / score_range
    else:
        relevance_scores = np.ones_like(
            hybrid_scores,
            dtype=np.float32,
        )

    # The first selection is always the strongest hybrid result.
    selected_local_indices: list[int] = [
        int(np.argmax(relevance_scores))
    ]

    available = np.ones(
        len(candidates),
        dtype=bool,
    )
    available[selected_local_indices[0]] = False

    target_count = min(
        result_count,
        len(candidates),
    )

    while len(selected_local_indices) < target_count:
        remaining_indices = np.flatnonzero(available)

        if len(remaining_indices) == 0:
            break

        selected_vectors = candidate_vectors[
            np.asarray(
                selected_local_indices,
                dtype=np.int64,
            )
        ]

        similarities = (
            candidate_vectors[remaining_indices]
            @ selected_vectors.transpose()
        )

        maximum_similarity = np.max(
            similarities,
            axis=1,
        )

        # Negative similarities should not create an artificial
        # diversity reward.
        maximum_similarity = np.clip(
            maximum_similarity,
            0.0,
            1.0,
        )

        mmr_scores = (
            diversity_lambda
            * relevance_scores[remaining_indices]
            - (
                1.0 - diversity_lambda
            )
            * maximum_similarity
        )

        best_remaining_position = int(
            np.argmax(mmr_scores)
        )

        selected_index = int(
            remaining_indices[
                best_remaining_position
            ]
        )

        selected_local_indices.append(selected_index)
        available[selected_index] = False

    return [
        candidates[index]
        for index in selected_local_indices
    ]

def search(
    query: str,
    top_k: int,
    candidate_pool: int,
    semantic_weight: float,
    keyword_weight: float,
    rrf_k: int,
    diversity_lambda: float,
    diversity_pool: int,
) -> None:
    """Run hybrid semantic and keyword search."""

    embedding_manifest = load_json(
        EMBEDDING_MANIFEST_PATH
    )
    keyword_manifest = load_json(
        KEYWORD_MANIFEST_PATH
    )

    embeddings = load_embeddings()
    embedding_records = load_embedding_records()

    keyword_matrix = load_matrix()
    keyword_records = load_keyword_records()
    vectorizer = load_vectorizer()

    metadata_by_id = load_catalogue_metadata()

    validate_indexes(
        embeddings=embeddings,
        embedding_records=embedding_records,
        keyword_matrix=keyword_matrix,
        keyword_records=keyword_records,
        embedding_manifest=embedding_manifest,
        keyword_manifest=keyword_manifest,
    )

    model_name = embedding_manifest.get("model_name")

    if not isinstance(model_name, str) or not model_name:
        raise RuntimeError(
            "The embedding manifest has no model name."
        )

    print(f"Loading query model: {model_name}")

    model = SentenceTransformer(model_name)

    semantic_scores = calculate_semantic_scores(
        query=query,
        model=model,
        embeddings=embeddings,
    )

    keyword_scores = calculate_keyword_scores(
        query=query,
        vectorizer=vectorizer,
        keyword_matrix=keyword_matrix,
    )

    ranked_results = create_hybrid_ranking(
        semantic_scores=semantic_scores,
        keyword_scores=keyword_scores,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        rrf_k=rrf_k,
        candidate_pool=candidate_pool,
    )

    selected_results = select_diverse_results(
        ranked_results=ranked_results,
        embeddings=embeddings,
        result_count=top_k,
        diversity_pool=diversity_pool,
        diversity_lambda=diversity_lambda,
    )

    print()
    print(f'Query: "{query}"')
    print(
        "Weights: "
        f"semantic={semantic_weight:.2f}, "
        f"keyword={keyword_weight:.2f}"
    )
    print(
        "Diversification: "
        f"lambda={diversity_lambda:.2f}, "
        f"candidate pool={diversity_pool}"
    )
    print(f"Results returned: {len(selected_results)}")
    print("=" * 80)

    for rank, (
        row_index,
        hybrid_score,
        semantic_rank,
        keyword_rank,
    ) in enumerate(
        selected_results,
        start=1,
    ):
        record = embedding_records[row_index]
        dataset_id = record["dataset_id"]

        dataset = metadata_by_id.get(dataset_id)

        if dataset is None:
            raise RuntimeError(
                "No cleaned catalogue metadata found for "
                f"dataset {dataset_id}."
            )

        title = dataset.get("title") or "Untitled dataset"
        organisation = get_organisation_title(dataset)
        formats = dataset.get("resource_formats", [])
        modified = dataset.get(
            "metadata_modified"
        ) or "Unknown"
        dataset_url = dataset.get("dataset_url") or ""
        description = shorten_text(
            dataset.get("description"),
        )

        format_text = (
            ", ".join(formats)
            if isinstance(formats, list) and formats
            else "No formats specified"
        )

        keyword_rank_text = (
            str(keyword_rank)
            if keyword_rank > 0
            else "No positive keyword match"
        )

        print()
        print(f"{rank}. {title}")
        print(f"   Hybrid score: {hybrid_score:.6f}")
        print(
            "   Semantic: "
            f"score={float(semantic_scores[row_index]):.4f}, "
            f"rank={semantic_rank}"
        )
        print(
            "   Keyword: "
            f"score={float(keyword_scores[row_index]):.4f}, "
            f"rank={keyword_rank_text}"
        )
        print(f"   Organisation: {organisation}")
        print(f"   Formats: {format_text}")
        print(f"   Modified: {modified}")

        if description:
            print(f"   Description: {description}")

        if dataset_url:
            print(f"   URL: {dataset_url}")

    print()
    print("=" * 80)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Search the Data.NSW catalogue using hybrid "
            "semantic and keyword retrieval."
        )
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of results to return "
            f"(default: {DEFAULT_TOP_K})."
        ),
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL,
        help=(
            "Number of leading results from each retrieval "
            "method considered for fusion "
            f"(default: {DEFAULT_CANDIDATE_POOL})."
        ),
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=DEFAULT_SEMANTIC_WEIGHT,
        help=(
            "Semantic-ranking weight "
            f"(default: {DEFAULT_SEMANTIC_WEIGHT})."
        ),
    )

    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=DEFAULT_KEYWORD_WEIGHT,
        help=(
            "Keyword-ranking weight "
            f"(default: {DEFAULT_KEYWORD_WEIGHT})."
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "Reciprocal Rank Fusion smoothing constant "
            f"(default: {DEFAULT_RRF_K})."
        ),
    )

    parser.add_argument(
        "--diversity-lambda",
        type=float,
        default=DEFAULT_DIVERSITY_LAMBDA,
        help=(
            "Balance between relevance and diversity. "
            "Use 1.0 for no diversification "
            f"(default: {DEFAULT_DIVERSITY_LAMBDA})."
        ),
    )

    parser.add_argument(
        "--diversity-pool",
        type=int,
        default=DEFAULT_DIVERSITY_POOL,
        help=(
            "Number of leading hybrid results considered during "
            "diversification "
            f"(default: {DEFAULT_DIVERSITY_POOL})."
        ),
    )

    arguments = parser.parse_args()

    if arguments.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    if arguments.candidate_pool <= 0:
        parser.error(
            "--candidate-pool must be greater than zero."
        )

    if arguments.rrf_k <= 0:
        parser.error("--rrf-k must be greater than zero.")

    if arguments.semantic_weight < 0:
        parser.error(
            "--semantic-weight cannot be negative."
        )

    if arguments.keyword_weight < 0:
        parser.error(
            "--keyword-weight cannot be negative."
        )

    if not 0.0 <= arguments.diversity_lambda <= 1.0:
        parser.error(
            "--diversity-lambda must be between 0 and 1."
        )

    if arguments.diversity_pool <= 0:
        parser.error(
        "--diversity-pool must be greater than zero."
    )

    total_weight = (
        arguments.semantic_weight
        + arguments.keyword_weight
    )

    if total_weight <= 0:
        parser.error(
            "At least one retrieval weight must be positive."
        )

    arguments.semantic_weight /= total_weight
    arguments.keyword_weight /= total_weight

    return arguments


def main() -> None:
    """Run hybrid search from the command line."""

    arguments = parse_arguments()
    query = " ".join(arguments.query).strip()

    if not query:
        raise SystemExit("A non-empty search query is required.")

    search(
        query=query,
        top_k=arguments.top_k,
        candidate_pool=arguments.candidate_pool,
        semantic_weight=arguments.semantic_weight,
        keyword_weight=arguments.keyword_weight,
        rrf_k=arguments.rrf_k,
        diversity_lambda=arguments.diversity_lambda,
        diversity_pool=arguments.diversity_pool,
    )


if __name__ == "__main__":
    main()