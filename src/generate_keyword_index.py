from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_PATH = Path("data/processed/search_documents.jsonl.gz")
OUTPUT_DIRECTORY = Path("data/index")

MATRIX_PATH = OUTPUT_DIRECTORY / "keyword_matrix.npz"
VECTORIZER_PATH = OUTPUT_DIRECTORY / "keyword_vectorizer.joblib"
RECORDS_PATH = OUTPUT_DIRECTORY / "keyword_records.jsonl.gz"
MANIFEST_PATH = OUTPUT_DIRECTORY / "keyword_manifest.json"

NGRAM_RANGE = (1, 2)
MAX_FEATURES = 200_000


def iter_search_documents() -> Iterator[dict[str, Any]]:
    """Yield generated search documents."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Search documents not found: {INPUT_PATH}"
        )

    with gzip.open(
        INPUT_PATH,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}."
                ) from error


def load_documents() -> tuple[
    list[str],
    list[str],
    list[str],
]:
    """Load dataset IDs, keyword hashes and keyword text."""

    dataset_ids: list[str] = []
    keyword_hashes: list[str] = []
    keyword_texts: list[str] = []

    seen_ids: set[str] = set()

    for document in iter_search_documents():
        dataset_id = document.get("dataset_id")
        keyword_hash = document.get("keyword_text_hash")
        keyword_text = document.get("keyword_text")

        if not isinstance(dataset_id, str) or not dataset_id:
            raise RuntimeError(
                "A search document is missing its dataset ID."
            )

        if dataset_id in seen_ids:
            raise RuntimeError(
                f"Duplicate dataset ID found: {dataset_id}"
            )

        if (
            not isinstance(keyword_hash, str)
            or not keyword_hash
        ):
            raise RuntimeError(
                f"Dataset {dataset_id} has no keyword-text hash."
            )

        if (
            not isinstance(keyword_text, str)
            or not keyword_text.strip()
        ):
            raise RuntimeError(
                f"Dataset {dataset_id} has no keyword text."
            )

        seen_ids.add(dataset_id)
        dataset_ids.append(dataset_id)
        keyword_hashes.append(keyword_hash)
        keyword_texts.append(keyword_text)

    return dataset_ids, keyword_hashes, keyword_texts


def build_vectorizer() -> TfidfVectorizer:
    """Create the initial lexical-search vectorizer."""

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=NGRAM_RANGE,
        min_df=1,
        max_df=0.98,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def validate_matrix(
    matrix: sparse.csr_matrix,
    expected_rows: int,
) -> None:
    """Validate the generated keyword-search matrix."""

    if not sparse.isspmatrix_csr(matrix):
        raise RuntimeError(
            "Expected a CSR sparse keyword matrix."
        )

    if matrix.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} rows, received "
            f"{matrix.shape[0]:,}."
        )

    if matrix.shape[1] <= 0:
        raise RuntimeError(
            "Keyword matrix contains no vocabulary features."
        )

    if matrix.dtype != np.float32:
        raise RuntimeError(
            f"Expected float32 values, received {matrix.dtype}."
        )

    if not np.isfinite(matrix.data).all():
        raise RuntimeError(
            "Keyword matrix contains NaN or infinite values."
        )

    empty_rows = int(
        np.count_nonzero(
            np.asarray(matrix.getnnz(axis=1)).ravel() == 0
        )
    )

    if empty_rows:
        raise RuntimeError(
            f"Keyword matrix contains {empty_rows:,} empty rows."
        )


def temporary_path(path: Path) -> Path:
    """Create a temporary sibling path retaining the file suffix."""

    return path.with_name(
        f"{path.stem}.tmp{path.suffix}"
    )


def save_records(
    dataset_ids: list[str],
    keyword_hashes: list[str],
    output_path: Path,
) -> None:
    """Save the row-to-dataset mapping for keyword search."""

    with gzip.open(
        output_path,
        mode="wt",
        encoding="utf-8",
    ) as file:
        for row_index, (
            dataset_id,
            keyword_hash,
        ) in enumerate(
            zip(
                dataset_ids,
                keyword_hashes,
                strict=True,
            )
        ):
            record = {
                "row_index": row_index,
                "dataset_id": dataset_id,
                "keyword_text_hash": keyword_hash,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def main() -> None:
    """Generate and save the TF-IDF keyword-search index."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    (
        dataset_ids,
        keyword_hashes,
        keyword_texts,
    ) = load_documents()

    print(f"Keyword documents loaded: {len(dataset_ids):,}")
    print("Building TF-IDF vocabulary and matrix...")

    vectorizer = build_vectorizer()

    matrix = vectorizer.fit_transform(keyword_texts)
    matrix = sparse.csr_matrix(
        matrix,
        dtype=np.float32,
    )

    validate_matrix(
        matrix=matrix,
        expected_rows=len(dataset_ids),
    )

    temporary_matrix_path = temporary_path(MATRIX_PATH)
    temporary_vectorizer_path = temporary_path(
        VECTORIZER_PATH
    )
    temporary_records_path = temporary_path(RECORDS_PATH)
    temporary_manifest_path = temporary_path(MANIFEST_PATH)

    with temporary_matrix_path.open("wb") as file:
        sparse.save_npz(
            file,
            matrix,
            compressed=True,
        )

    joblib.dump(
        vectorizer,
        temporary_vectorizer_path,
        compress=3,
    )

    save_records(
        dataset_ids=dataset_ids,
        keyword_hashes=keyword_hashes,
        output_path=temporary_records_path,
    )

    manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(INPUT_PATH),
        "matrix_file": MATRIX_PATH.name,
        "vectorizer_file": VECTORIZER_PATH.name,
        "records_file": RECORDS_PATH.name,
        "dataset_count": len(dataset_ids),
        "feature_count": int(matrix.shape[1]),
        "nonzero_values": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
        "matrix_format": "csr",
        "normalised": True,
        "ngram_range": list(NGRAM_RANGE),
        "maximum_features": MAX_FEATURES,
        "scikit_learn_version": sklearn.__version__,
    }

    with temporary_manifest_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    temporary_matrix_path.replace(MATRIX_PATH)
    temporary_vectorizer_path.replace(VECTORIZER_PATH)
    temporary_records_path.replace(RECORDS_PATH)
    temporary_manifest_path.replace(MANIFEST_PATH)

    matrix_size_mb = MATRIX_PATH.stat().st_size / (
        1024 * 1024
    )
    vectorizer_size_mb = VECTORIZER_PATH.stat().st_size / (
        1024 * 1024
    )

    print()
    print("Keyword-index generation completed successfully.")
    print(f"Dataset rows: {matrix.shape[0]:,}")
    print(f"Vocabulary features: {matrix.shape[1]:,}")
    print(f"Non-zero matrix values: {matrix.nnz:,}")
    print(f"Matrix file size: {matrix_size_mb:.2f} MB")
    print(
        "Vectorizer file size: "
        f"{vectorizer_size_mb:.2f} MB"
    )
    print(f"Saved keyword matrix: {MATRIX_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")
    print(f"Saved row records: {RECORDS_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()