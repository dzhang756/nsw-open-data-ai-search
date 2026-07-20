from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

CATALOGUE_PATH = Path("data/raw/catalogue.jsonl.gz")

ADDITIONAL_DATASET_FIELDS = (
    "author",
    "author_email",
    "maintainer",
    "maintainer_email",
    "version",
    "state",
    "type",
    "private",
    "groups",
    "extras",
)

ORGANISATION_FIELDS = (
    "id",
    "name",
    "title",
    "description",
)

TAG_FIELDS = (
    "id",
    "name",
    "display_name",
)

RESOURCE_FIELDS = (
    "id",
    "name",
    "description",
    "format",
    "mimetype",
    "mimetype_inner",
    "url",
    "url_type",
    "created",
    "last_modified",
    "size",
)


def has_value(value: Any) -> bool:
    """Return whether a value contains usable information."""

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)

    return True


def iter_catalogue() -> Iterator[dict[str, Any]]:
    """Yield datasets from the compressed catalogue."""

    if not CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Catalogue file not found: {CATALOGUE_PATH}"
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


def percentile(values: list[int], percentage: float) -> int:
    """Calculate a simple nearest-rank percentile."""

    if not values:
        return 0

    ordered_values = sorted(values)
    index = round((len(ordered_values) - 1) * percentage)

    return ordered_values[index]


def print_coverage(
    heading: str,
    coverage: Counter[str],
    fields: tuple[str, ...],
    total: int,
) -> None:
    """Print field coverage for one metadata object type."""

    print(heading)

    for field in fields:
        populated = coverage[field]
        percentage = populated / total * 100 if total else 0

        print(
            f"  {field:<22} "
            f"{populated:>7,} "
            f"({percentage:>6.2f}%)"
        )

    print()


def print_length_summary(
    heading: str,
    values: list[int],
) -> None:
    """Print useful text-length statistics."""

    print(heading)
    print(f"  Populated values: {len(values):,}")
    print(f"  Median length:    {percentile(values, 0.50):,}")
    print(f"  90th percentile:  {percentile(values, 0.90):,}")
    print(f"  99th percentile:  {percentile(values, 0.99):,}")
    print(f"  Maximum length:   {max(values, default=0):,}")
    print()


def shorten(value: str, maximum_length: int = 100) -> str:
    """Return a compact single-line example value."""

    compact_value = " ".join(value.split())

    if len(compact_value) <= maximum_length:
        return compact_value

    return compact_value[: maximum_length - 3] + "..."


def main() -> None:
    """Inspect nested catalogue metadata used by search."""

    dataset_count = 0
    organisation_count = 0
    tag_count = 0
    resource_count = 0

    additional_dataset_coverage: Counter[str] = Counter()
    organisation_coverage: Counter[str] = Counter()
    tag_coverage: Counter[str] = Counter()
    resource_coverage: Counter[str] = Counter()
    extra_key_counts: Counter[str] = Counter()

    extra_examples: dict[str, str] = {}

    title_lengths: list[int] = []
    description_lengths: list[int] = []
    resource_name_lengths: list[int] = []
    resource_description_lengths: list[int] = []

    for dataset in iter_catalogue():
        dataset_count += 1

        for field in ADDITIONAL_DATASET_FIELDS:
            if has_value(dataset.get(field)):
                additional_dataset_coverage[field] += 1

        title = dataset.get("title")

        if isinstance(title, str) and title.strip():
            title_lengths.append(len(title.strip()))

        description = dataset.get("notes")

        if isinstance(description, str) and description.strip():
            description_lengths.append(len(description.strip()))

        organisation = dataset.get("organization")

        if isinstance(organisation, dict) and organisation:
            organisation_count += 1

            for field in ORGANISATION_FIELDS:
                if has_value(organisation.get(field)):
                    organisation_coverage[field] += 1

        tags = dataset.get("tags")

        if isinstance(tags, list):
            for tag in tags:
                if not isinstance(tag, dict):
                    continue

                tag_count += 1

                for field in TAG_FIELDS:
                    if has_value(tag.get(field)):
                        tag_coverage[field] += 1

        resources = dataset.get("resources")

        if isinstance(resources, list):
            for resource in resources:
                if not isinstance(resource, dict):
                    continue

                resource_count += 1

                for field in RESOURCE_FIELDS:
                    if has_value(resource.get(field)):
                        resource_coverage[field] += 1

                resource_name = resource.get("name")

                if isinstance(resource_name, str) and resource_name.strip():
                    resource_name_lengths.append(
                        len(resource_name.strip())
                    )

                resource_description = resource.get("description")

                if (
                    isinstance(resource_description, str)
                    and resource_description.strip()
                ):
                    resource_description_lengths.append(
                        len(resource_description.strip())
                    )

        extras = dataset.get("extras")

        if isinstance(extras, list):
            seen_extra_keys: set[str] = set()

            for extra in extras:
                if not isinstance(extra, dict):
                    continue

                key = extra.get("key")
                value = extra.get("value")

                if not isinstance(key, str) or not key.strip():
                    continue

                normalised_key = key.strip()

                if normalised_key not in seen_extra_keys:
                    extra_key_counts[normalised_key] += 1
                    seen_extra_keys.add(normalised_key)

                if (
                    normalised_key not in extra_examples
                    and isinstance(value, str)
                    and value.strip()
                ):
                    extra_examples[normalised_key] = shorten(value)

    print(f"Total datasets inspected: {dataset_count:,}")
    print(f"Organisation objects: {organisation_count:,}")
    print(f"Tag objects: {tag_count:,}")
    print(f"Resource objects: {resource_count:,}")
    print()

    print_coverage(
        heading="Additional dataset field coverage:",
        coverage=additional_dataset_coverage,
        fields=ADDITIONAL_DATASET_FIELDS,
        total=dataset_count,
    )

    print_coverage(
        heading="Organisation field coverage:",
        coverage=organisation_coverage,
        fields=ORGANISATION_FIELDS,
        total=organisation_count,
    )

    print_coverage(
        heading="Tag field coverage:",
        coverage=tag_coverage,
        fields=TAG_FIELDS,
        total=tag_count,
    )

    print_coverage(
        heading="Resource field coverage:",
        coverage=resource_coverage,
        fields=RESOURCE_FIELDS,
        total=resource_count,
    )

    print("Top 25 extra metadata fields:")
    for key, count in extra_key_counts.most_common(25):
        example = extra_examples.get(key, "")
        percentage = count / dataset_count * 100 if dataset_count else 0

        print(
            f"  {key:<35} "
            f"{count:>6,} "
            f"({percentage:>6.2f}%)"
        )

        if example:
            print(f"    Example: {example}")

    print()

    print_length_summary(
        heading="Dataset title lengths:",
        values=title_lengths,
    )

    print_length_summary(
        heading="Dataset description lengths:",
        values=description_lengths,
    )

    print_length_summary(
        heading="Resource name lengths:",
        values=resource_name_lengths,
    )

    print_length_summary(
        heading="Resource description lengths:",
        values=resource_description_lengths,
    )


if __name__ == "__main__":
    main()