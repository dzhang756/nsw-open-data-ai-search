from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer

INPUT_PATH = Path("data/processed/catalogue_clean.jsonl.gz")
OUTPUT_PATH = Path("data/processed/search_documents.jsonl.gz")
MANIFEST_PATH = Path(
    "data/processed/search_documents_manifest.json"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_MODEL_MAX_TOKENS = 256

MAX_EMBEDDING_DESCRIPTION_CHARACTERS = 1_200
MAX_KEYWORD_DESCRIPTION_CHARACTERS = 12_000

MAX_TITLE_CHARACTERS = 400
MAX_ORGANISATION_CHARACTERS = 180
MAX_TAG_CHARACTERS = 350
MAX_GROUP_CHARACTERS = 200
MAX_FORMAT_CHARACTERS = 200
MAX_RESOURCE_NAME_CHARACTERS = 350

MAX_TAGS = 20
MAX_GROUPS = 10
MAX_RESOURCE_NAMES = 8


def iter_clean_catalogue() -> Iterator[dict[str, Any]]:
    """Yield datasets from the cleaned catalogue."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned catalogue not found: {INPUT_PATH}"
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


def truncate_text(value: str, maximum_length: int) -> str:
    """Truncate text at a word boundary where possible."""

    if len(value) <= maximum_length:
        return value

    shortened = value[: maximum_length + 1]
    final_space = shortened.rfind(" ")

    if final_space >= maximum_length * 0.75:
        shortened = shortened[:final_space]
    else:
        shortened = shortened[:maximum_length]

    return shortened.rstrip(" ,.;:-") + "…"


def stable_unique(values: Any) -> list[str]:
    """Return unique non-empty strings in their original order."""

    if not isinstance(values, list):
        return []

    results: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            continue

        cleaned_value = value.strip()

        if not cleaned_value:
            continue

        comparison_value = cleaned_value.casefold()

        if comparison_value in seen:
            continue

        seen.add(comparison_value)
        results.append(cleaned_value)

    return results


def get_organisation_title(dataset: dict[str, Any]) -> str:
    """Return the dataset organisation's display title."""

    organisation = dataset.get("organisation")

    if not isinstance(organisation, dict):
        return ""

    title = organisation.get("title")

    return title.strip() if isinstance(title, str) else ""


def get_resource_names(dataset: dict[str, Any]) -> list[str]:
    """Extract unique populated resource names."""

    resources = dataset.get("resources")

    if not isinstance(resources, list):
        return []

    names: list[str] = []

    for resource in resources:
        if not isinstance(resource, dict):
            continue

        name = resource.get("name")

        if isinstance(name, str) and name.strip():
            names.append(name.strip())

    return stable_unique(names)


def append_field(
    fields: list[str],
    label: str,
    value: str,
) -> None:
    """Append a labelled field when it contains text."""

    if value:
        fields.append(f"{label}: {value}")


def build_embedding_candidate(
    dataset: dict[str, Any],
) -> str:
    """Build prioritised text before token-based truncation."""

    title = str(dataset.get("title", "")).strip()
    description = str(dataset.get("description", "")).strip()
    organisation = get_organisation_title(dataset)

    tags = stable_unique(dataset.get("tags"))[:MAX_TAGS]
    groups = stable_unique(dataset.get("groups"))[:MAX_GROUPS]
    formats = stable_unique(dataset.get("resource_formats"))
    resource_names = get_resource_names(dataset)[:MAX_RESOURCE_NAMES]

    fields: list[str] = []

    append_field(
        fields,
        "Title",
        truncate_text(
            title,
            MAX_TITLE_CHARACTERS,
        ),
    )
    append_field(
        fields,
        "Organisation",
        truncate_text(
            organisation,
            MAX_ORGANISATION_CHARACTERS,
        ),
    )
    append_field(
        fields,
        "Tags",
        truncate_text(
            ", ".join(tags),
            MAX_TAG_CHARACTERS,
        ),
    )
    append_field(
        fields,
        "Categories",
        truncate_text(
            ", ".join(groups),
            MAX_GROUP_CHARACTERS,
        ),
    )
    append_field(
        fields,
        "Formats",
        truncate_text(
            ", ".join(formats),
            MAX_FORMAT_CHARACTERS,
        ),
    )
    append_field(
        fields,
        "Resources",
        truncate_text(
            ", ".join(resource_names),
            MAX_RESOURCE_NAME_CHARACTERS,
        ),
    )
    append_field(
        fields,
        "Description",
        truncate_text(
            description,
            MAX_EMBEDDING_DESCRIPTION_CHARACTERS,
        ),
    )

    return "\n".join(fields)


def fit_text_to_token_limit(
    value: str,
    tokenizer: Any,
) -> tuple[str, int, int, bool]:
    """
    Truncate text to the model limit while preserving the exact
    original prefix and field order.

    Returns:
        final text,
        original token count,
        final token count,
        whether truncation occurred
    """

    full_encoding = tokenizer(
        value,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )

    original_token_count = len(full_encoding["input_ids"])

    if original_token_count <= EMBEDDING_MODEL_MAX_TOKENS:
        return (
            value,
            original_token_count,
            original_token_count,
            False,
        )

    limited_encoding = tokenizer(
        value,
        add_special_tokens=True,
        truncation=True,
        max_length=EMBEDDING_MODEL_MAX_TOKENS,
        return_offsets_mapping=True,
        verbose=False,
    )

    offsets = limited_encoding["offset_mapping"]

    usable_end_offsets = [
        end
        for start, end in offsets
        if end > start
    ]

    if not usable_end_offsets:
        raise RuntimeError(
            "The tokenizer returned no usable text offsets."
        )

    final_character_position = max(usable_end_offsets)
    truncated_text = value[:final_character_position].rstrip()

    final_token_count = len(
        tokenizer(
            truncated_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )

    # This should rarely be necessary, but protects against an
    # offset boundary re-tokenising to more than the model limit.
    while (
        final_token_count > EMBEDDING_MODEL_MAX_TOKENS
        and truncated_text
    ):
        final_space = truncated_text.rfind(" ")

        if final_space > 0:
            truncated_text = truncated_text[:final_space].rstrip()
        else:
            truncated_text = truncated_text[:-1].rstrip()

        final_token_count = len(
            tokenizer(
                truncated_text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )

    return (
        truncated_text,
        original_token_count,
        final_token_count,
        True,
    )


def build_keyword_text(dataset: dict[str, Any]) -> str:
    """Build broader text for lexical keyword retrieval."""

    title = str(dataset.get("title", "")).strip()
    description = str(dataset.get("description", "")).strip()
    organisation = get_organisation_title(dataset)

    tags = stable_unique(dataset.get("tags"))
    groups = stable_unique(dataset.get("groups"))
    formats = stable_unique(dataset.get("resource_formats"))
    resource_names = get_resource_names(dataset)

    fields = [
        title,
        organisation,
        " ".join(tags),
        " ".join(groups),
        " ".join(formats),
        " ".join(resource_names),
        truncate_text(
            description,
            MAX_KEYWORD_DESCRIPTION_CHARACTERS,
        ),
    ]

    return "\n".join(
        field for field in fields if field
    )


def calculate_text_hash(value: str) -> str:
    """Calculate a stable SHA-256 hash for generated text."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def percentile(values: list[int], proportion: float) -> int:
    """Calculate a nearest-rank percentile."""

    if not values:
        return 0

    ordered_values = sorted(values)
    index = round((len(ordered_values) - 1) * proportion)

    return ordered_values[index]


def build_length_summary(
    values: list[int],
) -> dict[str, int]:
    """Build common length statistics."""

    return {
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "maximum": max(values, default=0),
    }


def main() -> None:
    """Create and validate all search-document records."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading tokenizer: {EMBEDDING_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset-based "
            "token truncation."
        )

    temporary_output_path = OUTPUT_PATH.with_name(
        OUTPUT_PATH.name + ".tmp"
    )
    temporary_manifest_path = MANIFEST_PATH.with_name(
        MANIFEST_PATH.name + ".tmp"
    )

    dataset_ids: set[str] = set()

    embedding_candidate_character_lengths: list[int] = []
    embedding_final_character_lengths: list[int] = []
    embedding_original_token_lengths: list[int] = []
    embedding_final_token_lengths: list[int] = []
    keyword_lengths: list[int] = []

    record_count = 0
    embedding_texts_token_truncated = 0
    keyword_descriptions_truncated = 0

    with gzip.open(
        temporary_output_path,
        mode="wt",
        encoding="utf-8",
    ) as output_file:
        for dataset in iter_clean_catalogue():
            dataset_id = dataset.get("dataset_id")

            if not isinstance(dataset_id, str) or not dataset_id:
                raise RuntimeError(
                    "A cleaned dataset is missing its dataset ID."
                )

            if dataset_id in dataset_ids:
                raise RuntimeError(
                    f"Duplicate dataset ID found: {dataset_id}"
                )

            dataset_ids.add(dataset_id)

            embedding_candidate = build_embedding_candidate(
                dataset
            )

            (
                embedding_text,
                original_token_count,
                final_token_count,
                was_token_truncated,
            ) = fit_text_to_token_limit(
                embedding_candidate,
                tokenizer,
            )

            keyword_text = build_keyword_text(dataset)

            if not embedding_text:
                raise RuntimeError(
                    f"Dataset {dataset_id} has no embedding text."
                )

            if not keyword_text:
                raise RuntimeError(
                    f"Dataset {dataset_id} has no keyword text."
                )

            if final_token_count > EMBEDDING_MODEL_MAX_TOKENS:
                raise RuntimeError(
                    f"Dataset {dataset_id} still contains "
                    f"{final_token_count} tokens after truncation."
                )

            raw_description = str(
                dataset.get("description", "")
            ).strip()

            if was_token_truncated:
                embedding_texts_token_truncated += 1

            if (
                len(raw_description)
                > MAX_KEYWORD_DESCRIPTION_CHARACTERS
            ):
                keyword_descriptions_truncated += 1

            search_document = {
                "dataset_id": dataset_id,
                "content_hash": dataset.get(
                    "content_hash",
                    "",
                ),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_text": embedding_text,
                "embedding_text_hash": calculate_text_hash(
                    embedding_text
                ),
                "embedding_original_token_count": (
                    original_token_count
                ),
                "embedding_token_count": final_token_count,
                "embedding_was_truncated": (
                    was_token_truncated
                ),
                "keyword_text": keyword_text,
                "keyword_text_hash": calculate_text_hash(
                    keyword_text
                ),
            }

            output_file.write(
                json.dumps(
                    search_document,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            output_file.write("\n")

            record_count += 1

            embedding_candidate_character_lengths.append(
                len(embedding_candidate)
            )
            embedding_final_character_lengths.append(
                len(embedding_text)
            )
            embedding_original_token_lengths.append(
                original_token_count
            )
            embedding_final_token_lengths.append(
                final_token_count
            )
            keyword_lengths.append(len(keyword_text))

    generated_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "generated_at_utc": generated_at,
        "input_file": str(INPUT_PATH),
        "output_file": OUTPUT_PATH.name,
        "record_count": record_count,
        "unique_dataset_ids": len(dataset_ids),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_model_max_tokens": (
            EMBEDDING_MODEL_MAX_TOKENS
        ),
        "embedding_description_character_limit": (
            MAX_EMBEDDING_DESCRIPTION_CHARACTERS
        ),
        "keyword_description_character_limit": (
            MAX_KEYWORD_DESCRIPTION_CHARACTERS
        ),
        "embedding_texts_token_truncated": (
            embedding_texts_token_truncated
        ),
        "keyword_descriptions_truncated": (
            keyword_descriptions_truncated
        ),
        "embedding_candidate_character_length": (
            build_length_summary(
                embedding_candidate_character_lengths
            )
        ),
        "embedding_final_character_length": (
            build_length_summary(
                embedding_final_character_lengths
            )
        ),
        "embedding_original_token_length": (
            build_length_summary(
                embedding_original_token_lengths
            )
        ),
        "embedding_final_token_length": (
            build_length_summary(
                embedding_final_token_lengths
            )
        ),
        "keyword_text_length": build_length_summary(
            keyword_lengths
        ),
    }

    with temporary_manifest_path.open(
        mode="w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")

    temporary_output_path.replace(OUTPUT_PATH)
    temporary_manifest_path.replace(MANIFEST_PATH)

    print()
    print(f"Search documents created: {record_count:,}")
    print(f"Unique dataset IDs: {len(dataset_ids):,}")
    print(
        "Embedding texts truncated by token limit: "
        f"{embedding_texts_token_truncated:,}"
    )
    print(
        "Keyword descriptions truncated: "
        f"{keyword_descriptions_truncated:,}"
    )
    print()
    print("Final embedding token counts:")
    print(
        "  Median:  "
        f"{percentile(embedding_final_token_lengths, 0.50):,}"
    )
    print(
        "  90th:    "
        f"{percentile(embedding_final_token_lengths, 0.90):,}"
    )
    print(
        "  99th:    "
        f"{percentile(embedding_final_token_lengths, 0.99):,}"
    )
    print(
        "  Maximum: "
        f"{max(embedding_final_token_lengths):,}"
    )
    print()
    print("Search-document generation completed successfully.")
    print(f"Saved search documents: {OUTPUT_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()