from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://data.nsw.gov.au/data/api/3/action/package_search"
PAGE_SIZE = 500

OUTPUT_DIRECTORY = Path("data/raw")
CATALOGUE_PATH = OUTPUT_DIRECTORY / "catalogue.jsonl.gz"
MANIFEST_PATH = OUTPUT_DIRECTORY / "catalogue_manifest.json"


def create_session() -> requests.Session:
    """Create an HTTP session with automatic retry handling."""

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "nsw-open-data-ai-search/0.1",
            "Accept": "application/json",
        }
    )

    return session


def fetch_page(
    session: requests.Session,
    start: int,
) -> dict[str, Any]:
    """Retrieve one page of datasets from the CKAN API."""

    response = session.get(
        API_URL,
        params={
            "rows": PAGE_SIZE,
            "start": start,
        },
        timeout=(10, 120),
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("success") is not True:
        raise RuntimeError(
            f"CKAN API returned an error: {payload.get('error')}"
        )

    return payload["result"]


def fetch_catalogue() -> tuple[list[dict[str, Any]], int]:
    """Retrieve every dataset currently reported by the catalogue."""

    datasets: list[dict[str, Any]] = []
    start = 0
    reported_count: int | None = None

    with create_session() as session:
        while reported_count is None or len(datasets) < reported_count:
            result = fetch_page(session, start)

            if reported_count is None:
                reported_count = int(result["count"])
                print(f"Datasets reported by API: {reported_count:,}")

            page_datasets = result.get("results", [])

            if not page_datasets:
                raise RuntimeError(
                    "The API returned an empty page before all datasets "
                    "were retrieved."
                )

            datasets.extend(page_datasets)
            start += len(page_datasets)

            print(
                f"Retrieved {len(datasets):,} of "
                f"{reported_count:,} datasets"
            )

    if reported_count is None:
        raise RuntimeError("The API did not report a dataset count.")

    return datasets, reported_count


def validate_catalogue(
    datasets: list[dict[str, Any]],
    reported_count: int,
) -> None:
    """Check that the catalogue is complete and contains unique IDs."""

    dataset_ids = [dataset.get("id") for dataset in datasets]

    if any(dataset_id is None for dataset_id in dataset_ids):
        raise RuntimeError("At least one dataset does not contain an ID.")

    unique_ids = set(dataset_ids)

    if len(datasets) != reported_count:
        raise RuntimeError(
            f"Retrieved {len(datasets):,} records, but the API reported "
            f"{reported_count:,}."
        )

    if len(unique_ids) != len(datasets):
        raise RuntimeError(
            f"Retrieved {len(datasets):,} records but only "
            f"{len(unique_ids):,} unique dataset IDs."
        )


def save_catalogue(
    datasets: list[dict[str, Any]],
    reported_count: int,
) -> None:
    """Save the catalogue and its retrieval metadata atomically."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    temporary_catalogue_path = CATALOGUE_PATH.with_suffix(
        CATALOGUE_PATH.suffix + ".tmp"
    )
    temporary_manifest_path = MANIFEST_PATH.with_suffix(
        MANIFEST_PATH.suffix + ".tmp"
    )

    with gzip.open(
        temporary_catalogue_path,
        mode="wt",
        encoding="utf-8",
    ) as file:
        for dataset in datasets:
            file.write(
                json.dumps(
                    dataset,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    fetched_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "api_url": API_URL,
        "fetched_at_utc": fetched_at,
        "reported_dataset_count": reported_count,
        "retrieved_dataset_count": len(datasets),
        "unique_dataset_ids": len(
            {dataset["id"] for dataset in datasets}
        ),
        "page_size": PAGE_SIZE,
        "catalogue_file": CATALOGUE_PATH.name,
    }

    with temporary_manifest_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    temporary_catalogue_path.replace(CATALOGUE_PATH)
    temporary_manifest_path.replace(MANIFEST_PATH)


def main() -> None:
    """Download, validate and save the complete catalogue."""

    datasets, reported_count = fetch_catalogue()

    validate_catalogue(
        datasets=datasets,
        reported_count=reported_count,
    )

    save_catalogue(
        datasets=datasets,
        reported_count=reported_count,
    )

    print()
    print("Catalogue download completed successfully.")
    print(f"Saved catalogue: {CATALOGUE_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()