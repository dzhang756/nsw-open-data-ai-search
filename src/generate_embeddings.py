from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/processed/search_documents.jsonl.gz")
OUTPUT_DIRECTORY = Path("data/index")

EMBEDDINGS_PATH = OUTPUT_DIRECTORY / "embeddings.npy"
RECORDS_PATH = OUTPUT_DIRECTORY / "embedding_records.jsonl.gz"
MANIFEST_PATH = OUTPUT_DIRECTORY / "embedding_manifest.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_MAX_TOKENS = 256
BATCH_SIZE = 64


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


def load_search_documents() -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """Load IDs, hashes and embedding text in stable row order."""

    dataset_ids: list[str] = []
    content_hashes: list[str] = []
    text_hashes: list[str] = []
    embedding_texts: list[str] = []

    seen_ids: set[str] = set()

    for document in iter_search_documents():
        dataset_id = document.get("dataset_id")
        embedding_text = document.get("embedding_text")
        content_hash = document.get("content_hash")
        text_hash = document.get("embedding_text_hash")
        token_count = document.get("embedding_token_count")

        if not isinstance(dataset_id, str) or not dataset_id:
            raise RuntimeError(
                "A search document is missing its dataset ID."
            )

        if dataset_id in seen_ids:
            raise RuntimeError(
                f"Duplicate dataset ID found: {dataset_id}"
            )

        if (
            not isinstance(embedding_text, str)
            or not embedding_text
        ):
            raise RuntimeError(
                f"Dataset {dataset_id} has no embedding text."
            )

        if (
            not isinstance(token_count, int)
            or token_count > EXPECTED_MAX_TOKENS
        ):
            raise RuntimeError(
                f"Dataset {dataset_id} has an invalid token count: "
                f"{token_count}"
            )

        if not isinstance(content_hash, str) or not content_hash:
            raise RuntimeError(
                f"Dataset {dataset_id} has no content hash."
            )

        if not isinstance(text_hash, str) or not text_hash:
            raise RuntimeError(
                f"Dataset {dataset_id} has no embedding text hash."
            )

        seen_ids.add(dataset_id)
        dataset_ids.append(dataset_id)
        content_hashes.append(content_hash)
        text_hashes.append(text_hash)
        embedding_texts.append(embedding_text)

    return (
        dataset_ids,
        content_hashes,
        text_hashes,
        embedding_texts,
    )


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> None:
    """Validate the generated embedding matrix."""

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected a 2D matrix, received shape "
            f"{embeddings.shape}."
        )

    if embeddings.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} embedding rows, "
            f"received {embeddings.shape[0]:,}."
        )

    if embeddings.shape[1] <= 0:
        raise RuntimeError(
            "Embedding matrix has no vector dimensions."
        )

    if embeddings.dtype != np.float32:
        raise RuntimeError(
            f"Expected float32 embeddings, received "
            f"{embeddings.dtype}."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embedding matrix contains NaN or infinite values."
        )

    row_norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(
        row_norms,
        1.0,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise RuntimeError(
            "One or more embeddings are not normalised."
        )


def save_records(
    dataset_ids: list[str],
    content_hashes: list[str],
    text_hashes: list[str],
    output_path: Path,
) -> None:
    """Save row-to-dataset mappings and hashes."""

    with gzip.open(
        output_path,
        mode="wt",
        encoding="utf-8",
    ) as file:
        for row_index, (
            dataset_id,
            content_hash,
            text_hash,
        ) in enumerate(
            zip(
                dataset_ids,
                content_hashes,
                text_hashes,
                strict=True,
            )
        ):
            record = {
                "row_index": row_index,
                "dataset_id": dataset_id,
                "content_hash": content_hash,
                "embedding_text_hash": text_hash,
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
    """Generate, validate and save the semantic-search index."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    (
        dataset_ids,
        content_hashes,
        text_hashes,
        embedding_texts,
    ) = load_search_documents()

    print(f"Search documents loaded: {len(dataset_ids):,}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    model_max_tokens = model.get_max_seq_length()

    if model_max_tokens != EXPECTED_MAX_TOKENS:
        raise RuntimeError(
            f"Expected a model limit of {EXPECTED_MAX_TOKENS}, "
            f"but the model reported {model_max_tokens}."
        )

    print("Generating embeddings...")

    embeddings = model.encode(
        embedding_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    validate_embeddings(
        embeddings=embeddings,
        expected_rows=len(dataset_ids),
    )

    temporary_embeddings_path = EMBEDDINGS_PATH.with_name(
        EMBEDDINGS_PATH.name + ".tmp"
    )
    temporary_records_path = RECORDS_PATH.with_name(
        RECORDS_PATH.name + ".tmp"
    )
    temporary_manifest_path = MANIFEST_PATH.with_name(
        MANIFEST_PATH.name + ".tmp"
    )

    with temporary_embeddings_path.open("wb") as file:
        np.save(file, embeddings)

    save_records(
        dataset_ids=dataset_ids,
        content_hashes=content_hashes,
        text_hashes=text_hashes,
        output_path=temporary_records_path,
    )

    manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": MODEL_NAME,
        "model_max_tokens": model_max_tokens,
        "input_file": str(INPUT_PATH),
        "embeddings_file": EMBEDDINGS_PATH.name,
        "records_file": RECORDS_PATH.name,
        "dataset_count": len(dataset_ids),
        "embedding_dimensions": embeddings.shape[1],
        "embedding_dtype": str(embeddings.dtype),
        "normalised": True,
        "batch_size": BATCH_SIZE,
    }

    with temporary_manifest_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    temporary_embeddings_path.replace(EMBEDDINGS_PATH)
    temporary_records_path.replace(RECORDS_PATH)
    temporary_manifest_path.replace(MANIFEST_PATH)

    file_size_mb = EMBEDDINGS_PATH.stat().st_size / (
        1024 * 1024
    )

    print()
    print("Embedding generation completed successfully.")
    print(f"Dataset embeddings: {embeddings.shape[0]:,}")
    print(f"Embedding dimensions: {embeddings.shape[1]:,}")
    print(f"Embedding data type: {embeddings.dtype}")
    print(f"Embedding file size: {file_size_mb:.2f} MB")
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Saved row records: {RECORDS_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()