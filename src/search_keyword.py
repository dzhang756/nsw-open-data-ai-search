from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

CLEAN_CATALOGUE_PATH = Path(
    "data/processed/catalogue_clean.jsonl.gz"
)
MATRIX_PATH = Path("data/index/keyword_matrix.npz")
VECTORIZER_PATH = Path(
    "data/index/keyword_vectorizer.joblib"
)
RECORDS_PATH = Path(
    "data/index/keyword_records.jsonl.gz"
)
MANIFEST_PATH = Path(
    "data/index/keyword_manifest.json"
)

DEFAULT_TOP_K = 10


def load_manifest() -> dict[str, Any]:
    """Load the keyword-index manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Keyword manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_matrix() -> sparse.csr_matrix:
    """Load the sparse TF-IDF document matrix."""

    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Keyword matrix not found: {MATRIX_PATH}"
        )

    matrix = sparse.load_npz(MATRIX_PATH)
    matrix = sparse.csr_matrix(
        matrix,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise RuntimeError(
            f"Expected a two-dimensional matrix, received "
            f"shape {matrix.shape}."
        )

    return matrix


def load_vectorizer() -> TfidfVectorizer:
    """Load the fitted TF-IDF vectorizer."""

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Keyword vectorizer not found: "
            f"{VECTORIZER_PATH}"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)

    if not isinstance(vectorizer, TfidfVectorizer):
        raise RuntimeError(
            "The loaded keyword vectorizer has an "
            "unexpected object type."
        )

    return vectorizer


def load_keyword_records() -> list[dict[str, Any]]:
    """Load and validate keyword-index row mappings."""

    if not RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Keyword records not found: {RECORDS_PATH}"
        )

    records: list[dict[str, Any]] = []

    with gzip.open(
        RECORDS_PATH,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for expected_row_index, line in enumerate(file):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "Invalid JSON in the keyword records at "
                    f"row {expected_row_index}."
                ) from error

            row_index = record.get("row_index")
            dataset_id = record.get("dataset_id")

            if row_index != expected_row_index:
                raise RuntimeError(
                    "Keyword records are not in contiguous row "
                    f"order: expected {expected_row_index}, "
                    f"received {row_index}."
                )

            if not isinstance(dataset_id, str) or not dataset_id:
                raise RuntimeError(
                    f"Keyword row {row_index} has no dataset ID."
                )

            records.append(record)

    return records


def iter_clean_catalogue() -> Iterator[dict[str, Any]]:
    """Yield cleaned catalogue records."""

    if not CLEAN_CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned catalogue not found: "
            f"{CLEAN_CATALOGUE_PATH}"
        )

    with gzip.open(
        CLEAN_CATALOGUE_PATH,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of "
                    "the cleaned catalogue."
                ) from error


def load_catalogue_metadata() -> dict[str, dict[str, Any]]:
    """Load display metadata by dataset ID."""

    metadata_by_id: dict[str, dict[str, Any]] = {}

    for dataset in iter_clean_catalogue():
        dataset_id = dataset.get("dataset_id")

        if not isinstance(dataset_id, str) or not dataset_id:
            raise RuntimeError(
                "A cleaned catalogue record has no dataset ID."
            )

        if dataset_id in metadata_by_id:
            raise RuntimeError(
                f"Duplicate cleaned dataset ID: {dataset_id}"
            )

        metadata_by_id[dataset_id] = dataset

    return metadata_by_id


def shorten_text(
    value: Any,
    maximum_length: int = 300,
) -> str:
    """Create a compact, single-line description preview."""

    if not isinstance(value, str):
        return ""

    compact_value = " ".join(value.split())

    if len(compact_value) <= maximum_length:
        return compact_value

    shortened = compact_value[: maximum_length + 1]
    final_space = shortened.rfind(" ")

    if final_space >= maximum_length * 0.75:
        shortened = shortened[:final_space]
    else:
        shortened = shortened[:maximum_length]

    return shortened.rstrip(" ,.;:-") + "…"


def get_organisation_title(dataset: dict[str, Any]) -> str:
    """Return the organisation display title."""

    organisation = dataset.get("organisation")

    if not isinstance(organisation, dict):
        return "Organisation not specified"

    title = organisation.get("title")

    if isinstance(title, str) and title.strip():
        return title.strip()

    return "Organisation not specified"


def find_top_positive_indices(
    scores: np.ndarray,
    top_k: int,
) -> list[int]:
    """Return the highest positive-scoring row indices."""

    positive_indices = np.flatnonzero(scores > 0)

    if len(positive_indices) == 0:
        return []

    result_count = min(top_k, len(positive_indices))

    if result_count == len(positive_indices):
        candidate_indices = positive_indices
    else:
        positive_scores = scores[positive_indices]

        local_indices = np.argpartition(
            positive_scores,
            -result_count,
        )[-result_count:]

        candidate_indices = positive_indices[local_indices]

    ordered_indices = candidate_indices[
        np.argsort(scores[candidate_indices])[::-1]
    ]

    return [
        int(index)
        for index in ordered_indices
    ]


def search(
    query: str,
    top_k: int,
) -> None:
    """Search the catalogue using TF-IDF keyword similarity."""

    manifest = load_manifest()
    matrix = load_matrix()
    vectorizer = load_vectorizer()
    keyword_records = load_keyword_records()
    metadata_by_id = load_catalogue_metadata()

    expected_count = manifest.get("dataset_count")
    expected_features = manifest.get("feature_count")

    if matrix.shape[0] != expected_count:
        raise RuntimeError(
            "Keyword row count does not match the manifest: "
            f"{matrix.shape[0]} != {expected_count}"
        )

    if matrix.shape[1] != expected_features:
        raise RuntimeError(
            "Keyword feature count does not match the manifest: "
            f"{matrix.shape[1]} != {expected_features}"
        )

    if len(keyword_records) != matrix.shape[0]:
        raise RuntimeError(
            "Keyword record count does not match the matrix: "
            f"{len(keyword_records)} != {matrix.shape[0]}"
        )

    query_vector = vectorizer.transform([query])
    query_vector = sparse.csr_matrix(
        query_vector,
        dtype=np.float32,
    )

    if query_vector.nnz == 0:
        print()
        print(f'Query: "{query}"')
        print(
            "None of the query terms are present in the "
            "keyword-search vocabulary."
        )
        return

    # The document and query vectors are L2 normalised,
    # so this dot product is cosine similarity.
    scores = (
        matrix @ query_vector.transpose()
    ).toarray().ravel()

    top_indices = find_top_positive_indices(
        scores=scores,
        top_k=top_k,
    )

    print()
    print(f'Query: "{query}"')
    print(f"Results returned: {len(top_indices)}")
    print("=" * 80)

    for rank, row_index in enumerate(
        top_indices,
        start=1,
    ):
        record = keyword_records[row_index]
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
        modified = dataset.get("metadata_modified") or "Unknown"
        dataset_url = dataset.get("dataset_url") or ""
        description = shorten_text(
            dataset.get("description"),
        )

        format_text = (
            ", ".join(formats)
            if isinstance(formats, list) and formats
            else "No formats specified"
        )

        print()
        print(f"{rank}. {title}")
        print(
            f"   Keyword score: "
            f"{float(scores[row_index]):.4f}"
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
            "Search the Data.NSW catalogue using TF-IDF "
            "keyword similarity."
        )
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Keyword or phrase search query.",
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

    arguments = parser.parse_args()

    if arguments.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    return arguments


def main() -> None:
    """Run keyword search from the command line."""

    arguments = parse_arguments()
    query = " ".join(arguments.query).strip()

    if not query:
        raise SystemExit("A non-empty search query is required.")

    search(
        query=query,
        top_k=arguments.top_k,
    )


if __name__ == "__main__":
    main()