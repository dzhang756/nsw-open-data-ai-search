from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

CLEAN_CATALOGUE_PATH = Path(
    "data/processed/catalogue_clean.jsonl.gz"
)

EMBEDDINGS_PATH = Path("data/index/embeddings.npy")
EMBEDDING_RECORDS_PATH = Path(
    "data/index/embedding_records.jsonl.gz"
)
EMBEDDING_MANIFEST_PATH = Path(
    "data/index/embedding_manifest.json"
)

KEYWORD_MATRIX_PATH = Path(
    "data/index/keyword_matrix.npz"
)
KEYWORD_VECTORIZER_PATH = Path(
    "data/index/keyword_vectorizer.joblib"
)
KEYWORD_RECORDS_PATH = Path(
    "data/index/keyword_records.jsonl.gz"
)
KEYWORD_MANIFEST_PATH = Path(
    "data/index/keyword_manifest.json"
)


@dataclass(frozen=True)
class SearchConfig:
    """Settings controlling hybrid retrieval and diversification."""

    top_k: int = 10
    candidate_pool: int = 200
    semantic_weight: float = 0.70
    keyword_weight: float = 0.30
    rrf_k: int = 60
    diversity_lambda: float = 0.85
    diversity_pool: int = 100

    def validated(self) -> SearchConfig:
        """Validate settings and normalise retrieval weights."""

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.candidate_pool <= 0:
            raise ValueError(
                "candidate_pool must be greater than zero."
            )

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if self.diversity_pool <= 0:
            raise ValueError(
                "diversity_pool must be greater than zero."
            )

        if not 0.0 <= self.diversity_lambda <= 1.0:
            raise ValueError(
                "diversity_lambda must be between 0 and 1."
            )

        if self.semantic_weight < 0:
            raise ValueError(
                "semantic_weight cannot be negative."
            )

        if self.keyword_weight < 0:
            raise ValueError(
                "keyword_weight cannot be negative."
            )

        total_weight = (
            self.semantic_weight
            + self.keyword_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one retrieval weight must be positive."
            )

        return replace(
            self,
            semantic_weight=(
                self.semantic_weight / total_weight
            ),
            keyword_weight=(
                self.keyword_weight / total_weight
            ),
        )


@dataclass(frozen=True)
class SearchResult:
    """One structured hybrid-search result."""

    row_index: int
    dataset_id: str
    title: str
    description: str
    organisation: str
    resource_formats: tuple[str, ...]
    metadata_modified: str
    dataset_url: str

    hybrid_score: float
    semantic_score: float
    semantic_rank: int
    keyword_score: float
    keyword_rank: int | None


@dataclass(frozen=True)
class SearchResponse:
    """Complete structured response for one search query."""

    query: str
    config: SearchConfig
    results: tuple[SearchResult, ...]
    catalogue_size: int
    keyword_query_feature_count: int


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected a JSON object in {path}."
        )

    return value


def _load_row_records(
    path: Path,
    record_type: str,
) -> list[dict[str, Any]]:
    """Load and validate a row-to-dataset mapping file."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []
    seen_dataset_ids: set[str] = set()

    with gzip.open(
        path,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for expected_row_index, line in enumerate(file):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {record_type} records "
                    f"at row {expected_row_index}."
                ) from error

            row_index = record.get("row_index")
            dataset_id = record.get("dataset_id")

            if row_index != expected_row_index:
                raise RuntimeError(
                    f"{record_type} records are not in "
                    "contiguous row order. Expected "
                    f"{expected_row_index}, received "
                    f"{row_index}."
                )

            if not isinstance(dataset_id, str) or not dataset_id:
                raise RuntimeError(
                    f"{record_type} row {row_index} has no "
                    "valid dataset ID."
                )

            if dataset_id in seen_dataset_ids:
                raise RuntimeError(
                    f"Duplicate dataset ID in {record_type} "
                    f"records: {dataset_id}"
                )

            seen_dataset_ids.add(dataset_id)
            records.append(record)

    return records


def _load_catalogue_metadata() -> dict[str, dict[str, Any]]:
    """Load cleaned catalogue metadata by dataset ID."""

    if not CLEAN_CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {CLEAN_CATALOGUE_PATH}"
        )

    metadata_by_id: dict[str, dict[str, Any]] = {}

    with gzip.open(
        CLEAN_CATALOGUE_PATH,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            try:
                dataset = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "Invalid JSON in the cleaned catalogue "
                    f"on line {line_number}."
                ) from error

            dataset_id = dataset.get("dataset_id")

            if not isinstance(dataset_id, str) or not dataset_id:
                raise RuntimeError(
                    "A cleaned catalogue record has no valid "
                    f"dataset ID on line {line_number}."
                )

            if dataset_id in metadata_by_id:
                raise RuntimeError(
                    f"Duplicate cleaned dataset ID: {dataset_id}"
                )

            metadata_by_id[dataset_id] = dataset

    return metadata_by_id


def _organisation_title(dataset: dict[str, Any]) -> str:
    """Return a dataset organisation's display title."""

    organisation = dataset.get("organisation")

    if not isinstance(organisation, dict):
        return "Organisation not specified"

    title = organisation.get("title")

    if isinstance(title, str) and title.strip():
        return title.strip()

    return "Organisation not specified"


def _resource_formats(
    dataset: dict[str, Any],
) -> tuple[str, ...]:
    """Return resource formats as immutable display values."""

    formats = dataset.get("resource_formats")

    if not isinstance(formats, list):
        return ()

    return tuple(
        value
        for value in formats
        if isinstance(value, str) and value
    )


def _build_rank_array(
    ordered_indices: np.ndarray,
    total_rows: int,
) -> np.ndarray:
    """Convert ordered row indices to one-based ranks."""

    ranks = np.zeros(total_rows, dtype=np.int32)

    ranks[ordered_indices] = np.arange(
        1,
        len(ordered_indices) + 1,
        dtype=np.int32,
    )

    return ranks


class SearchEngine:
    """Reusable semantic, keyword and hybrid search engine."""

    def __init__(self) -> None:
        self.embedding_manifest = _load_json(
            EMBEDDING_MANIFEST_PATH
        )
        self.keyword_manifest = _load_json(
            KEYWORD_MANIFEST_PATH
        )

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"File not found: {EMBEDDINGS_PATH}"
            )

        self.embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r",
        )

        if self.embeddings.ndim != 2:
            raise RuntimeError(
                "Expected a two-dimensional embedding matrix, "
                f"received shape {self.embeddings.shape}."
            )

        self.embedding_records = _load_row_records(
            EMBEDDING_RECORDS_PATH,
            record_type="Embedding",
        )

        if not KEYWORD_MATRIX_PATH.exists():
            raise FileNotFoundError(
                f"File not found: {KEYWORD_MATRIX_PATH}"
            )

        self.keyword_matrix = sparse.csr_matrix(
            sparse.load_npz(KEYWORD_MATRIX_PATH),
            dtype=np.float32,
        )

        if not KEYWORD_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"File not found: {KEYWORD_VECTORIZER_PATH}"
            )

        vectorizer = joblib.load(
            KEYWORD_VECTORIZER_PATH
        )

        if not isinstance(vectorizer, TfidfVectorizer):
            raise RuntimeError(
                "The loaded keyword vectorizer has an "
                "unexpected object type."
            )

        self.keyword_vectorizer = vectorizer

        self.keyword_records = _load_row_records(
            KEYWORD_RECORDS_PATH,
            record_type="Keyword",
        )

        self.metadata_by_id = _load_catalogue_metadata()

        self._model: SentenceTransformer | None = None

        self._validate_loaded_indexes()

    @property
    def model_name(self) -> str:
        """Return the semantic embedding model name."""

        model_name = self.embedding_manifest.get(
            "model_name"
        )

        if not isinstance(model_name, str) or not model_name:
            raise RuntimeError(
                "The embedding manifest has no model name."
            )

        return model_name

    @property
    def model(self) -> SentenceTransformer:
        """Load the query model only when first required."""

        if self._model is None:
            print(
                f"Loading query model: {self.model_name}"
            )

            self._model = SentenceTransformer(
                self.model_name
            )

            expected_dimensions = (
                self.embeddings.shape[1]
            )

            actual_dimensions = (
                self._model.get_sentence_embedding_dimension()
            )

            if actual_dimensions != expected_dimensions:
                raise RuntimeError(
                    "The query model embedding dimensions do "
                    "not match the stored index: "
                    f"{actual_dimensions} != "
                    f"{expected_dimensions}"
                )

        return self._model

    def _validate_loaded_indexes(self) -> None:
        """Validate row counts, dimensions and ID alignment."""

        embedding_count = self.embedding_manifest.get(
            "dataset_count"
        )
        embedding_dimensions = (
            self.embedding_manifest.get(
                "embedding_dimensions"
            )
        )

        keyword_count = self.keyword_manifest.get(
            "dataset_count"
        )
        keyword_features = self.keyword_manifest.get(
            "feature_count"
        )

        if self.embeddings.shape[0] != embedding_count:
            raise RuntimeError(
                "Embedding matrix row count does not match "
                f"its manifest: {self.embeddings.shape[0]} "
                f"!= {embedding_count}"
            )

        if (
            self.embeddings.shape[1]
            != embedding_dimensions
        ):
            raise RuntimeError(
                "Embedding dimensions do not match their "
                f"manifest: {self.embeddings.shape[1]} "
                f"!= {embedding_dimensions}"
            )

        if (
            self.keyword_matrix.shape[0]
            != keyword_count
        ):
            raise RuntimeError(
                "Keyword matrix row count does not match its "
                f"manifest: {self.keyword_matrix.shape[0]} "
                f"!= {keyword_count}"
            )

        if (
            self.keyword_matrix.shape[1]
            != keyword_features
        ):
            raise RuntimeError(
                "Keyword feature count does not match its "
                f"manifest: {self.keyword_matrix.shape[1]} "
                f"!= {keyword_features}"
            )

        expected_rows = self.embeddings.shape[0]

        if len(self.embedding_records) != expected_rows:
            raise RuntimeError(
                "Embedding record count does not match the "
                "embedding matrix."
            )

        if len(self.keyword_records) != expected_rows:
            raise RuntimeError(
                "Keyword record count does not match the "
                "embedding matrix."
            )

        if self.keyword_matrix.shape[0] != expected_rows:
            raise RuntimeError(
                "The semantic and keyword indexes contain "
                "different numbers of datasets."
            )

        embedding_ids = [
            record["dataset_id"]
            for record in self.embedding_records
        ]

        keyword_ids = [
            record["dataset_id"]
            for record in self.keyword_records
        ]

        if embedding_ids != keyword_ids:
            raise RuntimeError(
                "The semantic and keyword indexes do not use "
                "the same dataset row order."
            )

        missing_metadata_ids = [
            dataset_id
            for dataset_id in embedding_ids
            if dataset_id not in self.metadata_by_id
        ]

        if missing_metadata_ids:
            raise RuntimeError(
                "Cleaned metadata is missing for "
                f"{len(missing_metadata_ids):,} indexed "
                "datasets."
            )

        if len(self.metadata_by_id) != expected_rows:
            raise RuntimeError(
                "The cleaned catalogue size does not match "
                f"the search indexes: "
                f"{len(self.metadata_by_id)} != "
                f"{expected_rows}"
            )

    def _calculate_semantic_scores(
        self,
        query: str,
    ) -> np.ndarray:
        """Calculate semantic similarity for every dataset."""

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        expected_shape = (
            self.embeddings.shape[1],
        )

        if query_embedding.shape != expected_shape:
            raise RuntimeError(
                "Unexpected query embedding shape: "
                f"{query_embedding.shape}"
            )

        return np.asarray(
            self.embeddings @ query_embedding,
            dtype=np.float32,
        )

    def _calculate_keyword_scores(
        self,
        query: str,
    ) -> tuple[np.ndarray, int]:
        """Calculate keyword similarity for every dataset."""

        query_vector = self.keyword_vectorizer.transform(
            [query]
        )

        query_vector = sparse.csr_matrix(
            query_vector,
            dtype=np.float32,
        )

        feature_count = int(query_vector.nnz)

        if feature_count == 0:
            return (
                np.zeros(
                    self.keyword_matrix.shape[0],
                    dtype=np.float32,
                ),
                0,
            )

        scores = (
            self.keyword_matrix
            @ query_vector.transpose()
        ).toarray().ravel()

        return (
            np.asarray(scores, dtype=np.float32),
            feature_count,
        )

    def _create_hybrid_ranking(
        self,
        semantic_scores: np.ndarray,
        keyword_scores: np.ndarray,
        config: SearchConfig,
    ) -> list[tuple[int, float, int, int]]:
        """Combine rankings using weighted reciprocal rank fusion."""

        total_rows = len(semantic_scores)

        if len(keyword_scores) != total_rows:
            raise RuntimeError(
                "Semantic and keyword score arrays have "
                "different lengths."
            )

        pool_size = min(
            config.candidate_pool,
            total_rows,
        )

        semantic_order = np.argsort(
            semantic_scores,
            kind="stable",
        )[::-1]

        semantic_ranks = _build_rank_array(
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

        keyword_ranks = _build_rank_array(
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
                config.semantic_weight
                / (config.rrf_k + semantic_rank)
            )

            if keyword_rank > 0:
                hybrid_score += (
                    config.keyword_weight
                    / (config.rrf_k + keyword_rank)
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
            key=lambda value: (
                value[1],
                semantic_scores[value[0]],
                keyword_scores[value[0]],
            ),
            reverse=True,
        )

        return ranked_candidates

    def _select_diverse_results(
        self,
        ranked_results: list[
            tuple[int, float, int, int]
        ],
        config: SearchConfig,
    ) -> list[tuple[int, float, int, int]]:
        """Apply Maximal Marginal Relevance diversification."""

        if not ranked_results:
            return []

        pool_size = min(
            max(
                config.diversity_pool,
                config.top_k,
            ),
            len(ranked_results),
        )

        candidates = ranked_results[:pool_size]

        if config.diversity_lambda >= 1.0:
            return candidates[: config.top_k]

        row_indices = np.asarray(
            [
                result[0]
                for result in candidates
            ],
            dtype=np.int64,
        )

        candidate_vectors = np.asarray(
            self.embeddings[row_indices],
            dtype=np.float32,
        )

        hybrid_scores = np.asarray(
            [
                result[1]
                for result in candidates
            ],
            dtype=np.float32,
        )

        minimum_score = float(
            hybrid_scores.min()
        )
        maximum_score = float(
            hybrid_scores.max()
        )

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

        selected_local_indices: list[int] = [
            int(np.argmax(relevance_scores))
        ]

        available = np.ones(
            len(candidates),
            dtype=bool,
        )
        available[selected_local_indices[0]] = False

        target_count = min(
            config.top_k,
            len(candidates),
        )

        while (
            len(selected_local_indices)
            < target_count
        ):
            remaining_indices = np.flatnonzero(
                available
            )

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

            maximum_similarity = np.clip(
                maximum_similarity,
                0.0,
                1.0,
            )

            mmr_scores = (
                config.diversity_lambda
                * relevance_scores[remaining_indices]
                - (
                    1.0
                    - config.diversity_lambda
                )
                * maximum_similarity
            )

            best_position = int(
                np.argmax(mmr_scores)
            )

            selected_index = int(
                remaining_indices[best_position]
            )

            selected_local_indices.append(
                selected_index
            )
            available[selected_index] = False

        return [
            candidates[index]
            for index in selected_local_indices
        ]

    def search(
        self,
        query: str,
        config: SearchConfig | None = None,
    ) -> SearchResponse:
        """Run hybrid search and return structured results."""

        cleaned_query = " ".join(
            query.split()
        )

        if not cleaned_query:
            raise ValueError(
                "A non-empty search query is required."
            )

        active_config = (
            config or SearchConfig()
        ).validated()

        semantic_scores = (
            self._calculate_semantic_scores(
                cleaned_query
            )
        )

        (
            keyword_scores,
            keyword_query_feature_count,
        ) = self._calculate_keyword_scores(
            cleaned_query
        )

        ranked_results = (
            self._create_hybrid_ranking(
                semantic_scores=semantic_scores,
                keyword_scores=keyword_scores,
                config=active_config,
            )
        )

        selected_results = (
            self._select_diverse_results(
                ranked_results=ranked_results,
                config=active_config,
            )
        )

        results: list[SearchResult] = []

        for (
            row_index,
            hybrid_score,
            semantic_rank,
            keyword_rank,
        ) in selected_results:
            dataset_id = self.embedding_records[
                row_index
            ]["dataset_id"]

            dataset = self.metadata_by_id[
                dataset_id
            ]

            results.append(
                SearchResult(
                    row_index=row_index,
                    dataset_id=dataset_id,
                    title=(
                        dataset.get("title")
                        or "Untitled dataset"
                    ),
                    description=(
                        dataset.get("description")
                        or ""
                    ),
                    organisation=(
                        _organisation_title(dataset)
                    ),
                    resource_formats=(
                        _resource_formats(dataset)
                    ),
                    metadata_modified=(
                        dataset.get(
                            "metadata_modified"
                        )
                        or ""
                    ),
                    dataset_url=(
                        dataset.get("dataset_url")
                        or ""
                    ),
                    hybrid_score=float(
                        hybrid_score
                    ),
                    semantic_score=float(
                        semantic_scores[row_index]
                    ),
                    semantic_rank=semantic_rank,
                    keyword_score=float(
                        keyword_scores[row_index]
                    ),
                    keyword_rank=(
                        keyword_rank
                        if keyword_rank > 0
                        else None
                    ),
                )
            )

        return SearchResponse(
            query=cleaned_query,
            config=active_config,
            results=tuple(results),
            catalogue_size=self.embeddings.shape[0],
            keyword_query_feature_count=(
                keyword_query_feature_count
            ),
        )