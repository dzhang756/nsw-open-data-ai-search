from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

CATALOGUE_PATH = Path("data/raw/catalogue.jsonl.gz")

SEARCH_FIELDS = (
    "id",
    "name",
    "title",
    "notes",
    "metadata_created",
    "metadata_modified",
    "license_id",
    "license_title",
    "organization",
    "tags",
    "resources",
    "url",
)


def has_value(value: Any) -> bool:
    """Return whether a metadata value contains usable information."""

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)

    return True


def normalise_format(value: Any) -> str:
    """Standardise a resource-format value for profiling."""

    if not isinstance(value, str) or not value.strip():
        return "UNSPECIFIED"

    return value.strip().upper()


def load_catalogue() -> list[dict[str, Any]]:
    """Load every dataset from the compressed JSON Lines catalogue."""

    if not CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Catalogue file not found: {CATALOGUE_PATH}. "
            "Run src.fetch_catalogue first."
        )

    datasets: list[dict[str, Any]] = []

    with gzip.open(
        CATALOGUE_PATH,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            try:
                dataset = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}."
                ) from error

            datasets.append(dataset)

    return datasets


def profile_catalogue(
    datasets: list[dict[str, Any]],
) -> None:
    """Print a summary of metadata coverage and resource formats."""

    total_datasets = len(datasets)

    field_coverage: Counter[str] = Counter()
    resource_formats: Counter[str] = Counter()
    organisation_names: Counter[str] = Counter()

    datasets_with_resources = 0
    datasets_with_tags = 0
    total_resources = 0
    total_tags = 0

    for dataset in datasets:
        for field in SEARCH_FIELDS:
            if has_value(dataset.get(field)):
                field_coverage[field] += 1

        resources = dataset.get("resources")

        if isinstance(resources, list) and resources:
            datasets_with_resources += 1
            total_resources += len(resources)

            for resource in resources:
                if isinstance(resource, dict):
                    resource_formats[
                        normalise_format(resource.get("format"))
                    ] += 1

        tags = dataset.get("tags")

        if isinstance(tags, list) and tags:
            datasets_with_tags += 1
            total_tags += len(tags)

        organisation = dataset.get("organization")

        if isinstance(organisation, dict):
            organisation_name = (
                organisation.get("title")
                or organisation.get("name")
            )

            if has_value(organisation_name):
                organisation_names[str(organisation_name).strip()] += 1

    print(f"Total datasets: {total_datasets:,}")
    print()

    print("Search field coverage:")
    for field in SEARCH_FIELDS:
        populated = field_coverage[field]
        percentage = (
            populated / total_datasets * 100
            if total_datasets
            else 0
        )

        print(
            f"  {field:<20} "
            f"{populated:>6,} "
            f"({percentage:>6.2f}%)"
        )

    print()
    print(f"Datasets with resources: {datasets_with_resources:,}")
    print(f"Total resources: {total_resources:,}")
    print(f"Datasets with tags: {datasets_with_tags:,}")
    print(f"Total tags: {total_tags:,}")
    print(
        f"Unique organisations: "
        f"{len(organisation_names):,}"
    )

    print()
    print("Top 20 resource formats:")
    for resource_format, count in resource_formats.most_common(20):
        print(f"  {resource_format:<25} {count:>7,}")

    print()
    print("Top 10 organisations by dataset count:")
    for organisation, count in organisation_names.most_common(10):
        print(f"  {organisation:<60} {count:>6,}")


def main() -> None:
    """Load and profile the downloaded catalogue."""

    datasets = load_catalogue()
    profile_catalogue(datasets)


if __name__ == "__main__":
    main()