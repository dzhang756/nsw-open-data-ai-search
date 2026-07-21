from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

CLEAN_CATALOGUE_PATH = Path(
    "data/processed/catalogue_clean.jsonl.gz"
)

INDEX_DIRECTORY = Path("data/index")

VECTORIZER_PATH = INDEX_DIRECTORY / (
    "keyword_vectorizer.joblib"
)

RECORDS_PATH = INDEX_DIRECTORY / (
    "keyword_records.jsonl.gz"
)

MANIFEST_PATH = INDEX_DIRECTORY / (
    "keyword_manifest.json"
)

FIELD_MATRIX_PATHS = {
    "title": INDEX_DIRECTORY / "keyword_title_matrix.npz",
    "subjects": INDEX_DIRECTORY / "keyword_subjects_matrix.npz",
    "organisation": (
        INDEX_DIRECTORY / "keyword_organisation_matrix.npz"
    ),
    "resources": (
        INDEX_DIRECTORY / "keyword_resources_matrix.npz"
    ),
    "description": (
        INDEX_DIRECTORY / "keyword_description_matrix.npz"
    ),
}

DESCRIPTION_CHARACTER_LIMIT = 12_000
MAX_TAGS = 20
MAX_GROUPS = 10
MAX_RESOURCE_NAMES = 8


def compact_text(value: Any) -> str:
    """Return a compact single-line string."""

    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def limited_text(
    value: Any,
    maximum_characters: int,
) -> str:
    """Compact and restrict text to a character limit."""

    text = compact_text(value)

    if len(text) <= maximum_characters:
        return text

    return text[:maximum_characters].rstrip()


def extract_named_values(
    values: Any,
    maximum_values: int,
) -> list[str]:
    """Extract strings or display names from a list."""

    if not isinstance(values, list):
        return []

    extracted: list[str] = []
    seen_values: set[str] = set()

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

        if not text or normalised in seen_values:
            continue

        seen_values.add(normalised)
        extracted.append(text)

        if len(extracted) >= maximum_values:
            break

    return extracted


def organisation_text(dataset: dict[str, Any]) -> str:
    """Extract searchable organisation text."""

    organisation = dataset.get("organisation")

    if isinstance(organisation, str):
        return compact_text(organisation)

    if not isinstance(organisation, dict):
        return ""

    values = [
        compact_text(organisation.get("title")),
        compact_text(organisation.get("name")),
    ]

    return " ".join(
        dict.fromkeys(
            value
            for value in values
            if value
        )
    )


def resource_text(dataset: dict[str, Any]) -> str:
    """Build searchable text for formats and resource names."""

    formats = extract_named_values(
        dataset.get("resource_formats"),
        maximum_values=50,
    )

    resources = dataset.get("resources")
    resource_names: list[str] = []

    if isinstance(resources, list):
        for resource in resources:
            if not isinstance(resource, dict):
                continue

            name = compact_text(resource.get("name"))

            if not name:
                continue

            resource_names.append(name)

            if len(resource_names) >= MAX_RESOURCE_NAMES:
                break

    return " ".join(
        formats + resource_names
    )


def build_field_texts(
    dataset: dict[str, Any],
) -> dict[str, str]:
    """Build independently weighted keyword-search fields."""

    tags = extract_named_values(
        dataset.get("tags"),
        maximum_values=MAX_TAGS,
    )

    groups = extract_named_values(
        dataset.get("groups"),
        maximum_values=MAX_GROUPS,
    )

    return {
        "title": limited_text(
            dataset.get("title"),
            maximum_characters=400,
        ),
        "subjects": " ".join(tags + groups),
        "organisation": organisation_text(dataset),
        "resources": resource_text(dataset),
        "description": limited_text(
            dataset.get("description"),
            maximum_characters=(
                DESCRIPTION_CHARACTER_LIMIT
            ),
        ),
    }


def calculate_keyword_hash(
    field_texts: dict[str, str],
) -> str:
    """Calculate a stable hash of all keyword fields."""

    serialised = json.dumps(
        field_texts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialised.encode("utf-8")
    ).hexdigest()


def load_catalogue() -> tuple[
    list[dict[str, Any]],
    dict[str, list[str]],
]:
    """Load catalogue records and their field texts."""

    if not CLEAN_CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {CLEAN_CATALOGUE_PATH}"
        )

    records: list[dict[str, Any]] = []

    field_documents = {
        field_name: []
        for field_name in FIELD_MATRIX_PATHS
    }

    seen_dataset_ids: set[str] = set()

    with gzip.open(
        CLEAN_CATALOGUE_PATH,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for row_index, line in enumerate(file):
            try:
                dataset = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "Invalid JSON in the cleaned catalogue "
                    f"at row {row_index}."
                ) from error

            dataset_id = dataset.get("dataset_id")

            if (
                not isinstance(dataset_id, str)
                or not dataset_id
            ):
                raise RuntimeError(
                    f"Row {row_index} has no dataset ID."
                )

            if dataset_id in seen_dataset_ids:
                raise RuntimeError(
                    f"Duplicate dataset ID: {dataset_id}"
                )

            seen_dataset_ids.add(dataset_id)

            field_texts = build_field_texts(dataset)

            records.append(
                {
                    "row_index": row_index,
                    "dataset_id": dataset_id,
                    "content_hash": dataset.get(
                        "content_hash",
                        "",
                    ),
                    "keyword_hash": (
                        calculate_keyword_hash(
                            field_texts
                        )
                    ),
                }
            )

            for field_name, text in field_texts.items():
                field_documents[field_name].append(text)

    return records, field_documents


def combined_documents(
    field_documents: dict[str, list[str]],
) -> list[str]:
    """Combine fields only for shared vocabulary fitting."""

    dataset_count = len(
        field_documents["title"]
    )

    documents: list[str] = []

    for row_index in range(dataset_count):
        document = " ".join(
            field_documents[field_name][row_index]
            for field_name in FIELD_MATRIX_PATHS
            if field_documents[field_name][row_index]
        )

        documents.append(document)

    return documents


def save_sparse_matrix(
    matrix: sparse.csr_matrix,
    path: Path,
) -> None:
    """Atomically save a compressed sparse matrix."""

    temporary_path = path.with_name(
        f"{path.stem}.tmp{path.suffix}"
    )

    sparse.save_npz(
        temporary_path,
        matrix,
        compressed=True,
    )

    os.replace(temporary_path, path)


def save_joblib(
    value: Any,
    path: Path,
) -> None:
    """Atomically save a Joblib object."""

    temporary_path = path.with_name(
        f"{path.stem}.tmp{path.suffix}"
    )

    joblib.dump(
        value,
        temporary_path,
        compress=3,
    )

    os.replace(temporary_path, path)


def save_records(
    records: list[dict[str, Any]],
) -> None:
    """Atomically save row-to-dataset records."""

    temporary_path = RECORDS_PATH.with_name(
        f"{RECORDS_PATH.stem}.tmp.gz"
    )

    with gzip.open(
        temporary_path,
        mode="wt",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")

    os.replace(temporary_path, RECORDS_PATH)


def save_manifest(
    manifest: dict[str, Any],
) -> None:
    """Atomically save the keyword-index manifest."""

    temporary_path = MANIFEST_PATH.with_name(
        f"{MANIFEST_PATH.stem}.tmp.json"
    )

    with temporary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    os.replace(temporary_path, MANIFEST_PATH)


def main() -> None:
    """Generate a shared-vocabulary field-aware keyword index."""

    INDEX_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading cleaned catalogue...")

    records, field_documents = load_catalogue()
    documents = combined_documents(field_documents)

    print(
        f"Datasets loaded: {len(records):,}"
    )
    print("Fitting shared TF-IDF vocabulary...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        max_features=200_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    vectorizer.fit(documents)

    feature_count = len(
        vectorizer.get_feature_names_out()
    )

    print(
        f"Vocabulary features: {feature_count:,}"
    )

    matrices: dict[str, sparse.csr_matrix] = {}

    for field_name, texts in field_documents.items():
        print(
            f"Transforming field: {field_name}"
        )

        matrix = sparse.csr_matrix(
            vectorizer.transform(texts),
            dtype=np.float32,
        )

        if matrix.shape != (
            len(records),
            feature_count,
        ):
            raise RuntimeError(
                f"Unexpected {field_name} matrix shape: "
                f"{matrix.shape}"
            )

        if not np.isfinite(matrix.data).all():
            raise RuntimeError(
                f"The {field_name} matrix contains "
                "non-finite values."
            )

        matrices[field_name] = matrix

    print("Saving field matrices...")

    for field_name, matrix in matrices.items():
        save_sparse_matrix(
            matrix,
            FIELD_MATRIX_PATHS[field_name],
        )

    save_joblib(
        vectorizer,
        VECTORIZER_PATH,
    )

    save_records(records)

    manifest = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "index_version": 2,
        "index_type": "field_aware_tfidf",
        "dataset_count": len(records),
        "feature_count": feature_count,
        "vectorizer_path": str(VECTORIZER_PATH),
        "records_path": str(RECORDS_PATH),
        "fields": {
            field_name: {
                "matrix_path": str(
                    FIELD_MATRIX_PATHS[field_name]
                ),
                "shape": list(matrix.shape),
                "nonzero_values": int(matrix.nnz),
            }
            for field_name, matrix in matrices.items()
        },
        "vectorizer_settings": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 0.98,
            "max_features": 200_000,
            "sublinear_tf": True,
            "norm": "l2",
            "dtype": "float32",
        },
    }

    save_manifest(manifest)

    print()
    print("Field-aware keyword index generated.")
    print(
        f"Datasets: {len(records):,}"
    )
    print(
        f"Features: {feature_count:,}"
    )

    for field_name, matrix in matrices.items():
        print(
            f"{field_name}: "
            f"shape={matrix.shape}, "
            f"nonzero={matrix.nnz:,}"
        )

    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()