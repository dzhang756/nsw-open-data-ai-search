from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from src.clean_catalogue import (
    MACHINE_READABLE_FORMATS,
    build_dataset_url,
    calculate_content_hash,
)

CATALOGUE_PATH = Path("data/processed/catalogue_clean.jsonl.gz")
MANIFEST_PATH = Path(
    "data/processed/catalogue_clean_manifest.json"
)

HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def iter_catalogue() -> Iterator[dict[str, Any]]:
    """Yield records from the cleaned compressed catalogue."""

    if not CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned catalogue not found: {CATALOGUE_PATH}"
        )

    with gzip.open(
        CATALOGUE_PATH,
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


def load_manifest() -> dict[str, Any]:
    """Load the cleaned catalogue manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def has_duplicate_strings(values: list[Any]) -> bool:
    """Return whether a list contains duplicate text values."""

    text_values = [
        value.casefold()
        for value in values
        if isinstance(value, str)
    ]

    return len(text_values) != len(set(text_values))


def validate_record(
    dataset: dict[str, Any],
    line_number: int,
) -> list[str]:
    """Return validation errors for one cleaned dataset."""

    errors: list[str] = []

    dataset_id = dataset.get("dataset_id")
    dataset_name = dataset.get("name")
    title = dataset.get("title")
    description = dataset.get("description")
    organisation = dataset.get("organisation")
    tags = dataset.get("tags")
    groups = dataset.get("groups")
    resources = dataset.get("resources")
    resource_formats = dataset.get("resource_formats")
    content_hash = dataset.get("content_hash")

    record_label = (
        str(dataset_id)
        if dataset_id
        else f"line {line_number}"
    )

    for field_name, value in (
        ("dataset_id", dataset_id),
        ("name", dataset_name),
        ("title", title),
    ):
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{record_label}: missing or invalid {field_name}"
            )

    if not isinstance(description, str):
        errors.append(
            f"{record_label}: description must be a string"
        )

    if not isinstance(organisation, dict):
        errors.append(
            f"{record_label}: organisation must be an object"
        )
        organisation = {
            "id": "",
            "name": "",
            "title": "",
        }

    for field_name in ("id", "name", "title"):
        if not isinstance(
            organisation.get(field_name),
            str,
        ):
            errors.append(
                f"{record_label}: organisation.{field_name} "
                "must be a string"
            )

    for field_name, values in (
        ("tags", tags),
        ("groups", groups),
        ("resource_formats", resource_formats),
    ):
        if not isinstance(values, list):
            errors.append(
                f"{record_label}: {field_name} must be a list"
            )
            continue

        if not all(isinstance(value, str) for value in values):
            errors.append(
                f"{record_label}: {field_name} must contain "
                "only strings"
            )

        if has_duplicate_strings(values):
            errors.append(
                f"{record_label}: {field_name} contains duplicates"
            )

    if not isinstance(resources, list):
        errors.append(
            f"{record_label}: resources must be a list"
        )
        resources = []

    expected_resource_count = len(resources)

    if dataset.get("resource_count") != expected_resource_count:
        errors.append(
            f"{record_label}: resource_count does not match "
            "the resources list"
        )

    formats_from_resources: set[str] = set()

    for resource_number, resource in enumerate(
        resources,
        start=1,
    ):
        if not isinstance(resource, dict):
            errors.append(
                f"{record_label}: resource {resource_number} "
                "must be an object"
            )
            continue

        for field_name in (
            "id",
            "name",
            "description",
            "format",
            "url",
            "created",
            "last_modified",
        ):
            if not isinstance(resource.get(field_name), str):
                errors.append(
                    f"{record_label}: resource {resource_number} "
                    f"{field_name} must be a string"
                )

        resource_format = resource.get("format")

        if isinstance(resource_format, str):
            formats_from_resources.add(resource_format)

    if isinstance(resource_formats, list):
        expected_formats = sorted(formats_from_resources)

        if resource_formats != expected_formats:
            errors.append(
                f"{record_label}: resource_formats does not "
                "match the resource records"
            )

        expected_machine_readable = any(
            resource_format in MACHINE_READABLE_FORMATS
            for resource_format in resource_formats
        )

        if (
            dataset.get("has_machine_readable_resource")
            is not expected_machine_readable
        ):
            errors.append(
                f"{record_label}: machine-readable flag is "
                "inconsistent"
            )

    if isinstance(dataset_name, str) and dataset_name:
        expected_url = build_dataset_url(dataset_name)

        if dataset.get("dataset_url") != expected_url:
            errors.append(
                f"{record_label}: dataset_url is inconsistent"
            )

    if (
        not isinstance(content_hash, str)
        or HASH_PATTERN.fullmatch(content_hash) is None
    ):
        errors.append(
            f"{record_label}: invalid content_hash"
        )
    elif (
        isinstance(title, str)
        and isinstance(description, str)
        and isinstance(tags, list)
        and isinstance(groups, list)
        and isinstance(resources, list)
        and isinstance(organisation, dict)
    ):
        expected_hash = calculate_content_hash(
            title=title,
            description=description,
            organisation=organisation,
            tags=tags,
            groups=groups,
            resources=[
                resource
                for resource in resources
                if isinstance(resource, dict)
            ],
        )

        if content_hash != expected_hash:
            errors.append(
                f"{record_label}: content_hash does not match "
                "the cleaned searchable content"
            )

    return errors


def main() -> None:
    """Validate the complete cleaned catalogue."""

    manifest = load_manifest()

    dataset_ids: set[str] = set()
    errors: list[str] = []

    statistics: Counter[str] = Counter()

    record_count = 0

    for line_number, dataset in enumerate(
        iter_catalogue(),
        start=1,
    ):
        record_count += 1

        record_errors = validate_record(
            dataset=dataset,
            line_number=line_number,
        )
        errors.extend(record_errors)

        dataset_id = dataset.get("dataset_id")

        if isinstance(dataset_id, str):
            if dataset_id in dataset_ids:
                errors.append(
                    f"{dataset_id}: duplicate dataset ID"
                )

            dataset_ids.add(dataset_id)

        if not dataset.get("description"):
            statistics["blank_descriptions"] += 1

        organisation = dataset.get("organisation", {})

        if not organisation.get("title"):
            statistics["missing_organisations"] += 1

        if not dataset.get("tags"):
            statistics["datasets_without_tags"] += 1

        if not dataset.get("groups"):
            statistics["datasets_without_groups"] += 1

        if not dataset.get("resources"):
            statistics["datasets_without_resources"] += 1

        if "UNSPECIFIED" in dataset.get(
            "resource_formats",
            [],
        ):
            statistics[
                "datasets_with_unspecified_format"
            ] += 1

        resource_urls = [
            resource.get("url")
            for resource in dataset.get("resources", [])
            if isinstance(resource, dict)
        ]

        statistics["resources_without_urls"] += sum(
            1 for url in resource_urls if not url
        )

    manifest_count = manifest.get("dataset_count")

    if manifest_count != record_count:
        errors.append(
            "Manifest dataset count does not match the "
            f"catalogue: {manifest_count} != {record_count}"
        )

    manifest_unique_count = manifest.get(
        "unique_dataset_ids"
    )

    if manifest_unique_count != len(dataset_ids):
        errors.append(
            "Manifest unique ID count does not match the "
            f"catalogue: {manifest_unique_count} != "
            f"{len(dataset_ids)}"
        )

    print(f"Records validated: {record_count:,}")
    print(f"Unique dataset IDs: {len(dataset_ids):,}")
    print()

    print("Non-fatal data-quality observations:")
    print(
        "  Blank descriptions: "
        f"{statistics['blank_descriptions']:,}"
    )
    print(
        "  Missing organisations: "
        f"{statistics['missing_organisations']:,}"
    )
    print(
        "  Datasets without tags: "
        f"{statistics['datasets_without_tags']:,}"
    )
    print(
        "  Datasets without groups: "
        f"{statistics['datasets_without_groups']:,}"
    )
    print(
        "  Datasets without resources: "
        f"{statistics['datasets_without_resources']:,}"
    )
    print(
        "  Datasets containing an unspecified format: "
        f"{statistics['datasets_with_unspecified_format']:,}"
    )
    print(
        "  Resource records without URLs: "
        f"{statistics['resources_without_urls']:,}"
    )
    print()

    if errors:
        print(f"Validation errors found: {len(errors):,}")

        for error in errors[:20]:
            print(f"  - {error}")

        if len(errors) > 20:
            print(
                f"  ...and {len(errors) - 20:,} more errors"
            )

        raise SystemExit(1)

    print("No validation errors found.")
    print("Cleaned catalogue validation completed successfully.")


if __name__ == "__main__":
    main()