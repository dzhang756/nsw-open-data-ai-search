from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INDEX_DIRECTORY = Path("data/index")
PROCESSED_DIRECTORY = Path("data/processed")
OUTPUT_DIRECTORY = Path("dist")

BUNDLE_FILENAME = "search-index.tar.gz"
MANIFEST_FILENAME = "search-index-manifest.json"

BUNDLE_PATH = OUTPUT_DIRECTORY / BUNDLE_FILENAME
MANIFEST_PATH = OUTPUT_DIRECTORY / MANIFEST_FILENAME

REQUIRED_INDEX_FILES = (
    "embedding_manifest.json",
    "embedding_records.jsonl.gz",
    "embeddings.npy",
    "keyword_description_matrix.npz",
    "keyword_manifest.json",
    "keyword_organisation_matrix.npz",
    "keyword_records.jsonl.gz",
    "keyword_resources_matrix.npz",
    "keyword_subjects_matrix.npz",
    "keyword_title_matrix.npz",
    "keyword_vectorizer.joblib",
)

REQUIRED_PROCESSED_FILES = (
    "catalogue_clean.jsonl.gz",
    "catalogue_clean_manifest.json",
)

REQUIRED_RELEASE_FILES = (
    tuple(
        INDEX_DIRECTORY / filename
        for filename in REQUIRED_INDEX_FILES
    )
    + tuple(
        PROCESSED_DIRECTORY / filename
        for filename in REQUIRED_PROCESSED_FILES
    )
)


def sha256_file(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Read and validate a JSON object from disk."""

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            value = json.load(input_file)

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            f"Could not read JSON file: {file_path}"
        ) from error

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected a JSON object in: {file_path}"
        )

    return value


def validate_release_files() -> tuple[Path, ...]:
    """Confirm that all files required by the hosted app exist."""

    missing_files = [
        file_path
        for file_path in REQUIRED_RELEASE_FILES
        if not file_path.is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {file_path.as_posix()}"
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "The search-index release is incomplete. "
            "Missing required files:\n"
            f"{missing_text}"
        )

    return tuple(
        sorted(
            REQUIRED_RELEASE_FILES,
            key=lambda path: path.as_posix().casefold(),
        )
    )


def create_bundle(
    release_files: tuple[Path, ...],
) -> None:
    """Create a compressed archive containing the release files."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    BUNDLE_PATH.unlink(
        missing_ok=True,
    )

    with tarfile.open(
        BUNDLE_PATH,
        mode="w:gz",
    ) as archive:
        for file_path in release_files:
            archive.add(
                file_path,
                arcname=file_path.as_posix(),
                recursive=False,
            )


def create_file_records(
    release_files: tuple[Path, ...],
) -> list[dict[str, object]]:
    """Create size and checksum records for packaged files."""

    records: list[dict[str, object]] = []

    for file_path in release_files:
        records.append(
            {
                "path": file_path.as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )

    return records


def create_release_manifest(
    release_files: tuple[Path, ...],
) -> dict[str, object]:
    """Create metadata describing the packaged search index."""

    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(
            f"Bundle was not created: {BUNDLE_PATH}"
        )

    catalogue_clean_manifest = load_json_file(
        PROCESSED_DIRECTORY
        / "catalogue_clean_manifest.json"
    )

    embedding_manifest = load_json_file(
        INDEX_DIRECTORY
        / "embedding_manifest.json"
    )

    keyword_manifest = load_json_file(
        INDEX_DIRECTORY
        / "keyword_manifest.json"
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat().replace(
        "+00:00",
        "Z",
    )

    file_records = create_file_records(
        release_files
    )

    total_uncompressed_bytes = sum(
        file_path.stat().st_size
        for file_path in release_files
    )

    return {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "bundle": {
            "filename": BUNDLE_FILENAME,
            "size_bytes": BUNDLE_PATH.stat().st_size,
            "sha256": sha256_file(BUNDLE_PATH),
        },
        "release": {
            "file_count": len(release_files),
            "total_uncompressed_bytes": (
                total_uncompressed_bytes
            ),
            "files": file_records,
        },
        "source_manifests": {
            "catalogue_clean": (
                catalogue_clean_manifest
            ),
            "embedding": embedding_manifest,
            "keyword": keyword_manifest,
        },
    }


def write_release_manifest(
    manifest: dict[str, object],
) -> None:
    """Write the release manifest to disk."""

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            manifest,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        output_file.write("\n")


def verify_bundle_members(
    release_files: tuple[Path, ...],
) -> None:
    """Confirm that the archive contains exactly the expected files."""

    expected_members = {
        file_path.as_posix()
        for file_path in release_files
    }

    try:
        with tarfile.open(
            BUNDLE_PATH,
            mode="r:gz",
        ) as archive:
            actual_members = {
                member.name
                for member in archive.getmembers()
                if member.isfile()
            }

    except (
        OSError,
        tarfile.TarError,
    ) as error:
        raise RuntimeError(
            "The generated release bundle could not "
            "be reopened."
        ) from error

    if actual_members != expected_members:
        missing_members = sorted(
            expected_members - actual_members
        )

        unexpected_members = sorted(
            actual_members - expected_members
        )

        raise RuntimeError(
            "The generated archive has unexpected contents.\n"
            f"Missing: {missing_members}\n"
            f"Unexpected: {unexpected_members}"
        )


def format_megabytes(size_bytes: int) -> str:
    """Convert bytes to a readable megabyte value."""

    return f"{size_bytes / 1024 / 1024:.2f} MB"


def main() -> None:
    """Package the files required by the hosted search app."""

    print(
        "Validating search-index release files..."
    )

    release_files = validate_release_files()

    print(
        f"Release files found: {len(release_files):,}"
    )

    print(
        "Creating compressed search-index bundle..."
    )

    create_bundle(
        release_files
    )

    verify_bundle_members(
        release_files
    )

    manifest = create_release_manifest(
        release_files
    )

    write_release_manifest(
        manifest
    )

    bundle_size = BUNDLE_PATH.stat().st_size

    uncompressed_size = sum(
        file_path.stat().st_size
        for file_path in release_files
    )

    bundle_information = manifest.get("bundle")

    if not isinstance(bundle_information, dict):
        raise RuntimeError(
            "Generated manifest has no valid bundle section."
        )

    bundle_checksum = bundle_information.get(
        "sha256"
    )

    if not isinstance(bundle_checksum, str):
        raise RuntimeError(
            "Generated manifest has no valid bundle checksum."
        )

    print()
    print(
        "Search-index packaging completed successfully."
    )

    print(
        f"Files packaged: {len(release_files):,}"
    )

    print(
        "Uncompressed release size: "
        f"{format_megabytes(uncompressed_size)}"
    )

    print(
        "Compressed bundle size: "
        f"{format_megabytes(bundle_size)}"
    )

    print(
        f"Bundle: {BUNDLE_PATH}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print(
        f"Bundle SHA-256: {bundle_checksum}"
    )


if __name__ == "__main__":
    main()