from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote, unquote, urlparse

INPUT_PATH = Path("data/raw/catalogue.jsonl.gz")
OUTPUT_DIRECTORY = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIRECTORY / "catalogue_clean.jsonl.gz"
MANIFEST_PATH = OUTPUT_DIRECTORY / "catalogue_clean_manifest.json"

DATASET_PAGE_BASE_URL = "https://data.nsw.gov.au/data/dataset"

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

FORMAT_ALIASES = {
    "ARCGIS REST": "ARCGIS REST SERVICE",
    "ESRI REST": "ARCGIS REST SERVICE",
    "ESRI REST SERVICE": "ARCGIS REST SERVICE",
    "REST API": "API",
    "GEO TIFF": "GEOTIFF",
    "GEO-TIFF": "GEOTIFF",
    "GEO JSON": "GEOJSON",
    "GEO-JSON": "GEOJSON",
    "MS EXCEL": "XLS",
    "MICROSOFT EXCEL": "EXCEL",
    "TEXT": "TXT",
}

MIMETYPE_FORMATS = {
    "text/csv": "CSV",
    "application/csv": "CSV",
    "application/json": "JSON",
    "application/geo+json": "GEOJSON",
    "application/vnd.geo+json": "GEOJSON",
    "application/pdf": "PDF",
    "application/zip": "ZIP",
    "application/x-zip-compressed": "ZIP",
    "application/xml": "XML",
    "text/xml": "XML",
    "text/plain": "TXT",
    "text/html": "HTML",
    "application/vnd.ms-excel": "XLS",
    (
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ): "XLSX",
    "image/tiff": "TIFF",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
}

EXTENSION_FORMATS = {
    ".csv": "CSV",
    ".json": "JSON",
    ".geojson": "GEOJSON",
    ".pdf": "PDF",
    ".zip": "ZIP",
    ".xml": "XML",
    ".txt": "TXT",
    ".xls": "XLS",
    ".xlsx": "XLSX",
    ".shp": "SHP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".kml": "KML",
    ".kmz": "KMZ",
    ".html": "HTML",
    ".htm": "HTML",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
}

MACHINE_READABLE_FORMATS = {
    "API",
    "ARCGIS REST SERVICE",
    "CSV",
    "EXCEL",
    "GEOJSON",
    "GEOTIFF",
    "JSON",
    "KML",
    "KMZ",
    "SHP",
    "TIFF",
    "TXT",
    "WFS",
    "WMS",
    "XLS",
    "XLSX",
    "XML",
}


def clean_text(value: Any) -> str:
    """Convert a metadata value into clean, single-line text."""

    if not isinstance(value, str):
        return ""

    text = html.unescape(value)

    text = re.sub(
        r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>",
        " ",
        text,
    )
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def clean_named_objects(
    values: Any,
    preferred_fields: tuple[str, ...],
) -> list[str]:
    """Extract unique names from nested CKAN objects."""

    if not isinstance(values, list):
        return []

    cleaned_values: list[str] = []
    seen_values: set[str] = set()

    for value in values:
        if not isinstance(value, dict):
            continue

        cleaned_name = ""

        for field in preferred_fields:
            cleaned_name = clean_text(value.get(field))

            if cleaned_name:
                break

        if not cleaned_name:
            continue

        comparison_value = cleaned_name.casefold()

        if comparison_value in seen_values:
            continue

        seen_values.add(comparison_value)
        cleaned_values.append(cleaned_name)

    return cleaned_values


def normalise_format_label(value: Any) -> str:
    """Standardise an explicitly supplied resource-format label."""

    cleaned_value = clean_text(value)

    if not cleaned_value:
        return ""

    normalised_value = cleaned_value.upper()
    normalised_value = WHITESPACE_PATTERN.sub(" ", normalised_value)

    return FORMAT_ALIASES.get(normalised_value, normalised_value)


def infer_format_from_mimetype(value: Any) -> str:
    """Infer a format from a MIME type when possible."""

    cleaned_value = clean_text(value).lower()

    if not cleaned_value:
        return ""

    mimetype = cleaned_value.split(";", maxsplit=1)[0].strip()

    return MIMETYPE_FORMATS.get(mimetype, "")


def infer_format_from_url(value: Any) -> str:
    """Infer a resource format from its URL extension."""

    cleaned_url = clean_text(value)

    if not cleaned_url:
        return ""

    parsed_url = urlparse(cleaned_url)
    decoded_path = unquote(parsed_url.path).lower()

    for extension, resource_format in EXTENSION_FORMATS.items():
        if decoded_path.endswith(extension):
            return resource_format

    return ""


def determine_resource_format(resource: dict[str, Any]) -> str:
    """Determine the best available normalised resource format."""

    explicit_format = normalise_format_label(resource.get("format"))

    if explicit_format:
        return explicit_format

    mimetype_format = infer_format_from_mimetype(
        resource.get("mimetype")
    )

    if mimetype_format:
        return mimetype_format

    url_format = infer_format_from_url(resource.get("url"))

    if url_format:
        return url_format

    return "UNSPECIFIED"


def extract_extras(dataset: dict[str, Any]) -> dict[str, str]:
    """Convert CKAN extras into a simple key-value mapping."""

    extras = dataset.get("extras")

    if not isinstance(extras, list):
        return {}

    cleaned_extras: dict[str, str] = {}

    for extra in extras:
        if not isinstance(extra, dict):
            continue

        key = clean_text(extra.get("key"))
        value = clean_text(extra.get("value"))

        if key and value:
            cleaned_extras[key] = value

    return cleaned_extras


def clean_organisation(value: Any) -> dict[str, str]:
    """Extract the useful organisation fields."""

    if not isinstance(value, dict):
        return {
            "id": "",
            "name": "",
            "title": "",
        }

    return {
        "id": clean_text(value.get("id")),
        "name": clean_text(value.get("name")),
        "title": (
            clean_text(value.get("title"))
            or clean_text(value.get("name"))
        ),
    }


def clean_resources(
    values: Any,
) -> tuple[list[dict[str, str]], list[str]]:
    """Clean resource records and return their unique formats."""

    if not isinstance(values, list):
        return [], []

    cleaned_resources: list[dict[str, str]] = []
    formats: set[str] = set()

    for resource in values:
        if not isinstance(resource, dict):
            continue

        resource_format = determine_resource_format(resource)
        formats.add(resource_format)

        cleaned_resources.append(
            {
                "id": clean_text(resource.get("id")),
                "name": clean_text(resource.get("name")),
                "description": clean_text(
                    resource.get("description")
                ),
                "format": resource_format,
                "url": clean_text(resource.get("url")),
                "created": clean_text(resource.get("created")),
                "last_modified": clean_text(
                    resource.get("last_modified")
                ),
            }
        )

    return cleaned_resources, sorted(formats)


def build_dataset_url(dataset_name: str) -> str:
    """Construct the authoritative Data.NSW dataset page URL."""

    if not dataset_name:
        return ""

    encoded_name = quote(dataset_name, safe="-._~")

    return f"{DATASET_PAGE_BASE_URL}/{encoded_name}"


def calculate_content_hash(
    *,
    title: str,
    description: str,
    organisation: dict[str, str],
    tags: list[str],
    groups: list[str],
    resources: list[dict[str, str]],
) -> str:
    """Hash the fields that may later affect search embeddings."""

    hash_content = {
        "title": title,
        "description": description,
        "organisation": organisation,
        "tags": tags,
        "groups": groups,
        "resources": [
            {
                "name": resource["name"],
                "description": resource["description"],
                "format": resource["format"],
            }
            for resource in resources
        ],
    }

    encoded_content = json.dumps(
        hash_content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded_content).hexdigest()


def clean_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Transform one raw CKAN dataset into the search-ready schema."""

    dataset_id = clean_text(dataset.get("id"))
    dataset_name = clean_text(dataset.get("name"))
    title = clean_text(dataset.get("title"))
    description = clean_text(dataset.get("notes"))

    organisation = clean_organisation(
        dataset.get("organization")
    )

    tags = clean_named_objects(
        dataset.get("tags"),
        preferred_fields=("display_name", "name"),
    )

    groups = clean_named_objects(
        dataset.get("groups"),
        preferred_fields=("display_name", "title", "name"),
    )

    resources, resource_formats = clean_resources(
        dataset.get("resources")
    )

    extras = extract_extras(dataset)

    content_hash = calculate_content_hash(
        title=title,
        description=description,
        organisation=organisation,
        tags=tags,
        groups=groups,
        resources=resources,
    )

    return {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "title": title,
        "description": description,
        "organisation": organisation,
        "tags": tags,
        "groups": groups,
        "license_id": clean_text(dataset.get("license_id")),
        "license_title": clean_text(
            dataset.get("license_title")
        ),
        "metadata_created": clean_text(
            dataset.get("metadata_created")
        ),
        "metadata_modified": clean_text(
            dataset.get("metadata_modified")
        ),
        "state": clean_text(dataset.get("state")),
        "private": bool(dataset.get("private", False)),
        "dataset_url": build_dataset_url(dataset_name),
        "source_portal": extras.get(
            "harvest_portal",
            "Data.NSW",
        ),
        "resource_count": len(resources),
        "resource_formats": resource_formats,
        "has_machine_readable_resource": any(
            resource_format in MACHINE_READABLE_FORMATS
            for resource_format in resource_formats
        ),
        "resources": resources,
        "content_hash": content_hash,
    }


def iter_raw_catalogue() -> Iterator[dict[str, Any]]:
    """Yield raw datasets from the downloaded catalogue."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Raw catalogue not found: {INPUT_PATH}"
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


def main() -> None:
    """Clean and validate the complete catalogue."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    temporary_output_path = OUTPUT_PATH.with_name(
        OUTPUT_PATH.name + ".tmp"
    )
    temporary_manifest_path = MANIFEST_PATH.with_name(
        MANIFEST_PATH.name + ".tmp"
    )

    dataset_count = 0
    machine_readable_count = 0
    datasets_with_resources = 0
    datasets_with_tags = 0

    dataset_ids: set[str] = set()
    format_counts: Counter[str] = Counter()

    with gzip.open(
        temporary_output_path,
        mode="wt",
        encoding="utf-8",
    ) as output_file:
        for raw_dataset in iter_raw_catalogue():
            cleaned_dataset = clean_dataset(raw_dataset)

            dataset_id = cleaned_dataset["dataset_id"]

            if not dataset_id:
                raise RuntimeError(
                    "A cleaned dataset is missing its dataset ID."
                )

            if dataset_id in dataset_ids:
                raise RuntimeError(
                    f"Duplicate dataset ID found: {dataset_id}"
                )

            dataset_ids.add(dataset_id)
            dataset_count += 1

            if cleaned_dataset["resource_count"] > 0:
                datasets_with_resources += 1

            if cleaned_dataset["tags"]:
                datasets_with_tags += 1

            if cleaned_dataset[
                "has_machine_readable_resource"
            ]:
                machine_readable_count += 1

            for resource_format in cleaned_dataset[
                "resource_formats"
            ]:
                format_counts[resource_format] += 1

            output_file.write(
                json.dumps(
                    cleaned_dataset,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            output_file.write("\n")

    generated_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "generated_at_utc": generated_at,
        "input_file": str(INPUT_PATH),
        "output_file": OUTPUT_PATH.name,
        "dataset_count": dataset_count,
        "unique_dataset_ids": len(dataset_ids),
        "datasets_with_resources": datasets_with_resources,
        "datasets_with_tags": datasets_with_tags,
        "datasets_with_machine_readable_resources": (
            machine_readable_count
        ),
        "unique_resource_formats": len(format_counts),
        "top_resource_formats_by_dataset": dict(
            format_counts.most_common(20)
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

    print(f"Datasets cleaned: {dataset_count:,}")
    print(f"Unique dataset IDs: {len(dataset_ids):,}")
    print(
        "Datasets with resources: "
        f"{datasets_with_resources:,}"
    )
    print(
        "Datasets with tags: "
        f"{datasets_with_tags:,}"
    )
    print(
        "Datasets with machine-readable resources: "
        f"{machine_readable_count:,}"
    )
    print(f"Unique resource formats: {len(format_counts):,}")
    print()
    print("Catalogue cleaning completed successfully.")
    print(f"Saved cleaned catalogue: {OUTPUT_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()