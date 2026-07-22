from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


REPOSITORY = "dzhang756/nsw-open-data-ai-search"
RELEASE_TAG = "search-index-latest"

RELEASE_BASE_URL = (
    f"https://github.com/{REPOSITORY}/releases/download/"
    f"{RELEASE_TAG}"
)

MANIFEST_FILENAME = "search-index-manifest.json"
BUNDLE_FILENAME = "search-index.tar.gz"

MANIFEST_URL = (
    f"{RELEASE_BASE_URL}/{MANIFEST_FILENAME}"
)

LOCAL_RELEASE_MANIFEST_PATH = Path(
    "data/index/search-index-release-manifest.json"
)

LOCAL_CHECK_STATE_PATH = Path(
    "data/index/search-index-check-state.json"
)

DEFAULT_CHECK_INTERVAL_SECONDS = 60 * 60

CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 600
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

REQUIRED_RELEASE_PATHS = (
    Path("data/index/embedding_manifest.json"),
    Path("data/index/embedding_records.jsonl.gz"),
    Path("data/index/embeddings.npy"),
    Path("data/index/keyword_description_matrix.npz"),
    Path("data/index/keyword_manifest.json"),
    Path("data/index/keyword_organisation_matrix.npz"),
    Path("data/index/keyword_records.jsonl.gz"),
    Path("data/index/keyword_resources_matrix.npz"),
    Path("data/index/keyword_subjects_matrix.npz"),
    Path("data/index/keyword_title_matrix.npz"),
    Path("data/index/keyword_vectorizer.joblib"),
    Path("data/processed/catalogue_clean.jsonl.gz"),
    Path("data/processed/catalogue_clean_manifest.json"),
)

INDEX_UPDATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReleaseFileRecord:
    """One file contained in the search-index release."""

    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class IndexReleaseStatus:
    """Result of checking or installing the search index."""

    version: str
    generated_at_utc: str | None
    updated: bool
    warning: str | None = None


def sha256_file(
    file_path: Path,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def is_valid_sha256(value: object) -> bool:
    """Return whether a value is a hexadecimal SHA-256 hash."""

    if not isinstance(value, str):
        return False

    if len(value) != 64:
        return False

    try:
        int(value, 16)
    except ValueError:
        return False

    return True


def read_json_file(
    file_path: Path,
) -> dict[str, Any] | None:
    """Read a local JSON object, returning None if invalid."""

    if not file_path.is_file():
        return None

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            value = json.load(input_file)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(value, dict):
        return None

    return value


def write_json_file_atomic(
    file_path: Path,
    value: dict[str, Any],
) -> None:
    """Write JSON without leaving a partially written file."""

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = file_path.with_name(
        f".{file_path.name}.{os.getpid()}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                value,
                output_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

            output_file.write("\n")

        os.replace(
            temporary_path,
            file_path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True,
        )


def required_release_files_exist() -> bool:
    """Return whether all files needed by the app are present."""

    return all(
        file_path.is_file()
        for file_path in REQUIRED_RELEASE_PATHS
    )


def request_headers() -> dict[str, str]:
    """Return headers for GitHub release downloads."""

    return {
        "Accept": "application/octet-stream",
        "Cache-Control": "no-cache",
        "User-Agent": (
            "nsw-open-data-ai-search/1.0"
        ),
    }


def download_release_manifest() -> dict[str, Any]:
    """Download the current GitHub release manifest."""

    response = requests.get(
        MANIFEST_URL,
        headers=request_headers(),
        params={
            "cache_bust": time.time_ns(),
        },
        timeout=(
            CONNECT_TIMEOUT_SECONDS,
            60,
        ),
    )

    response.raise_for_status()

    try:
        value = response.json()
    except requests.JSONDecodeError as error:
        raise RuntimeError(
            "The downloaded release manifest is not "
            "valid JSON."
        ) from error

    if not isinstance(value, dict):
        raise RuntimeError(
            "The downloaded release manifest is not "
            "a JSON object."
        )

    return value


def validate_release_manifest(
    manifest: dict[str, Any],
) -> tuple[ReleaseFileRecord, ...]:
    """Validate and normalise release-manifest contents."""

    if manifest.get("schema_version") != 1:
        raise RuntimeError(
            "The release manifest has an unsupported "
            "schema version."
        )

    bundle = manifest.get("bundle")

    if not isinstance(bundle, dict):
        raise RuntimeError(
            "The release manifest has no valid bundle section."
        )

    if bundle.get("filename") != BUNDLE_FILENAME:
        raise RuntimeError(
            "The release manifest references an unexpected "
            "bundle filename."
        )

    if not is_valid_sha256(
        bundle.get("sha256")
    ):
        raise RuntimeError(
            "The release manifest contains an invalid "
            "bundle checksum."
        )

    bundle_size = bundle.get("size_bytes")

    if (
        not isinstance(bundle_size, int)
        or isinstance(bundle_size, bool)
        or bundle_size <= 0
    ):
        raise RuntimeError(
            "The release manifest contains an invalid "
            "bundle size."
        )

    release = manifest.get("release")

    if not isinstance(release, dict):
        raise RuntimeError(
            "The release manifest has no valid release section."
        )

    raw_files = release.get("files")

    if not isinstance(raw_files, list):
        raise RuntimeError(
            "The release manifest contains no valid file list."
        )

    records: list[ReleaseFileRecord] = []

    for raw_record in raw_files:
        if not isinstance(raw_record, dict):
            raise RuntimeError(
                "The release manifest contains an invalid "
                "file record."
            )

        raw_path = raw_record.get("path")
        size_bytes = raw_record.get("size_bytes")
        checksum = raw_record.get("sha256")

        if not isinstance(raw_path, str):
            raise RuntimeError(
                "A release file has no valid path."
            )

        file_path = Path(raw_path)

        if (
            file_path.is_absolute()
            or ".." in file_path.parts
        ):
            raise RuntimeError(
                "The release manifest contains an unsafe "
                f"path: {raw_path}"
            )

        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise RuntimeError(
                "A release file has an invalid size: "
                f"{raw_path}"
            )

        if not is_valid_sha256(checksum):
            raise RuntimeError(
                "A release file has an invalid checksum: "
                f"{raw_path}"
            )

        records.append(
            ReleaseFileRecord(
                path=file_path,
                size_bytes=size_bytes,
                sha256=checksum,
            )
        )

    actual_paths = {
        record.path
        for record in records
    }

    expected_paths = set(
        REQUIRED_RELEASE_PATHS
    )

    if actual_paths != expected_paths:
        missing_paths = sorted(
            expected_paths - actual_paths,
            key=lambda path: path.as_posix(),
        )

        unexpected_paths = sorted(
            actual_paths - expected_paths,
            key=lambda path: path.as_posix(),
        )

        raise RuntimeError(
            "The release manifest contains unexpected files.\n"
            f"Missing: {[path.as_posix() for path in missing_paths]}\n"
            f"Unexpected: "
            f"{[path.as_posix() for path in unexpected_paths]}"
        )

    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.path.as_posix().casefold()
            ),
        )
    )


def download_bundle(
    manifest: dict[str, Any],
    destination: Path,
) -> None:
    """Download and verify the compressed release bundle."""

    bundle = manifest["bundle"]

    expected_size = int(
        bundle["size_bytes"]
    )

    expected_checksum = str(
        bundle["sha256"]
    )

    bundle_url = (
        f"{RELEASE_BASE_URL}/{BUNDLE_FILENAME}"
    )

    with requests.get(
        bundle_url,
        headers=request_headers(),
        params={
            "cache_bust": expected_checksum[:16],
        },
        stream=True,
        timeout=(
            CONNECT_TIMEOUT_SECONDS,
            READ_TIMEOUT_SECONDS,
        ),
    ) as response:
        response.raise_for_status()

        with destination.open(
            "wb"
        ) as output_file:
            for chunk in response.iter_content(
                chunk_size=DOWNLOAD_CHUNK_SIZE
            ):
                if chunk:
                    output_file.write(chunk)

    actual_size = destination.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            "The downloaded search-index bundle has an "
            "unexpected size. "
            f"Expected {expected_size:,} bytes, "
            f"received {actual_size:,} bytes."
        )

    actual_checksum = sha256_file(
        destination
    )

    if actual_checksum != expected_checksum:
        raise RuntimeError(
            "The downloaded search-index bundle failed "
            "SHA-256 verification."
        )


def extract_and_verify_bundle(
    bundle_path: Path,
    extraction_directory: Path,
    records: tuple[ReleaseFileRecord, ...],
) -> None:
    """Extract only expected files and verify each checksum."""

    expected_names = {
        record.path.as_posix()
        for record in records
    }

    try:
        with tarfile.open(
            bundle_path,
            mode="r:gz",
        ) as archive:
            members = archive.getmembers()

            file_members = {
                member.name: member
                for member in members
                if member.isfile()
            }

            non_regular_members = [
                member.name
                for member in members
                if (
                    not member.isfile()
                    and not member.isdir()
                )
            ]

            if non_regular_members:
                raise RuntimeError(
                    "The release bundle contains unsupported "
                    "archive members."
                )

            actual_names = set(
                file_members
            )

            if actual_names != expected_names:
                raise RuntimeError(
                    "The release bundle contents do not match "
                    "the release manifest."
                )

            for record in records:
                member_name = (
                    record.path.as_posix()
                )

                member = file_members[
                    member_name
                ]

                source_file = archive.extractfile(
                    member
                )

                if source_file is None:
                    raise RuntimeError(
                        "Could not read archive member: "
                        f"{member_name}"
                    )

                destination = (
                    extraction_directory
                    / record.path
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with (
                    source_file,
                    destination.open("wb")
                    as output_file,
                ):
                    shutil.copyfileobj(
                        source_file,
                        output_file,
                    )

    except tarfile.TarError as error:
        raise RuntimeError(
            "The downloaded search-index bundle is not "
            "a valid archive."
        ) from error

    for record in records:
        extracted_file = (
            extraction_directory
            / record.path
        )

        if not extracted_file.is_file():
            raise RuntimeError(
                "An expected release file was not extracted: "
                f"{record.path}"
            )

        actual_size = (
            extracted_file.stat().st_size
        )

        if actual_size != record.size_bytes:
            raise RuntimeError(
                "An extracted release file has an unexpected "
                f"size: {record.path}"
            )

        actual_checksum = sha256_file(
            extracted_file
        )

        if actual_checksum != record.sha256:
            raise RuntimeError(
                "An extracted release file failed checksum "
                f"verification: {record.path}"
            )


def install_release_files(
    extraction_directory: Path,
    records: tuple[ReleaseFileRecord, ...],
) -> None:
    """Move verified files into their application locations."""

    for record in records:
        source_path = (
            extraction_directory
            / record.path
        )

        destination_path = record.path

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.replace(
            source_path,
            destination_path,
        )


def manifest_bundle_checksum(
    manifest: dict[str, Any] | None,
) -> str | None:
    """Read a bundle checksum from a release manifest."""

    if manifest is None:
        return None

    bundle = manifest.get("bundle")

    if not isinstance(bundle, dict):
        return None

    checksum = bundle.get("sha256")

    if not is_valid_sha256(checksum):
        return None

    return checksum


def status_from_manifest(
    manifest: dict[str, Any],
    *,
    updated: bool,
    warning: str | None = None,
) -> IndexReleaseStatus:
    """Create an application status from a manifest."""

    checksum = manifest_bundle_checksum(
        manifest
    )

    if checksum is None:
        raise RuntimeError(
            "The local release manifest has no valid checksum."
        )

    generated_at = manifest.get(
        "generated_at_utc"
    )

    if not isinstance(generated_at, str):
        generated_at = None

    return IndexReleaseStatus(
        version=checksum,
        generated_at_utc=generated_at,
        updated=updated,
        warning=warning,
    )


def local_unversioned_status() -> IndexReleaseStatus:
    """Create a stable version for a locally generated index."""

    digest = hashlib.sha256()

    manifest_paths = (
        Path("data/index/embedding_manifest.json"),
        Path("data/index/keyword_manifest.json"),
        Path(
            "data/processed/"
            "catalogue_clean_manifest.json"
        ),
    )

    for manifest_path in manifest_paths:
        digest.update(
            manifest_path.as_posix().encode(
                "utf-8"
            )
        )

        if manifest_path.is_file():
            digest.update(
                sha256_file(
                    manifest_path
                ).encode("ascii")
            )

    return IndexReleaseStatus(
        version=(
            "local-unversioned-"
            f"{digest.hexdigest()}"
        ),
        generated_at_utc=None,
        updated=False,
    )


def check_was_recent(
    bundle_checksum: str,
    check_interval_seconds: int,
) -> bool:
    """Return whether GitHub was checked recently."""

    if check_interval_seconds <= 0:
        return False

    state = read_json_file(
        LOCAL_CHECK_STATE_PATH
    )

    if state is None:
        return False

    if (
        state.get("bundle_sha256")
        != bundle_checksum
    ):
        return False

    checked_at = state.get(
        "checked_at_epoch"
    )

    if not isinstance(
        checked_at,
        (int, float),
    ):
        return False

    elapsed_seconds = (
        time.time() - float(checked_at)
    )

    return (
        0 <= elapsed_seconds
        < check_interval_seconds
    )


def record_successful_check(
    bundle_checksum: str,
) -> None:
    """Record the most recent successful GitHub check."""

    write_json_file_atomic(
        LOCAL_CHECK_STATE_PATH,
        {
            "bundle_sha256": bundle_checksum,
            "checked_at_epoch": time.time(),
        },
    )


def download_and_install_release(
    manifest: dict[str, Any],
    records: tuple[ReleaseFileRecord, ...],
) -> None:
    """Download, verify and install a release."""

    Path("data").mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".search-index-release-",
        dir="data",
    ) as temporary_directory:
        temporary_root = Path(
            temporary_directory
        )

        bundle_path = (
            temporary_root
            / BUNDLE_FILENAME
        )

        extraction_directory = (
            temporary_root
            / "extracted"
        )

        extraction_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        download_bundle(
            manifest=manifest,
            destination=bundle_path,
        )

        extract_and_verify_bundle(
            bundle_path=bundle_path,
            extraction_directory=(
                extraction_directory
            ),
            records=records,
        )

        install_release_files(
            extraction_directory=(
                extraction_directory
            ),
            records=records,
        )

    write_json_file_atomic(
        LOCAL_RELEASE_MANIFEST_PATH,
        manifest,
    )


def ensure_search_index(
    check_interval_seconds: int = (
        DEFAULT_CHECK_INTERVAL_SECONDS
    ),
) -> IndexReleaseStatus:
    """
    Ensure the application has a complete current search index.

    A locally generated unversioned index is preserved for
    development. A release-managed index is checked against
    GitHub at most once per configured interval.
    """

    with INDEX_UPDATE_LOCK:
        local_files_complete = (
            required_release_files_exist()
        )

        local_manifest = read_json_file(
            LOCAL_RELEASE_MANIFEST_PATH
        )

        local_checksum = (
            manifest_bundle_checksum(
                local_manifest
            )
        )

        if (
            local_files_complete
            and local_manifest is None
        ):
            return local_unversioned_status()

        if (
            local_files_complete
            and local_manifest is not None
            and local_checksum is not None
            and check_was_recent(
                bundle_checksum=local_checksum,
                check_interval_seconds=(
                    check_interval_seconds
                ),
            )
        ):
            return status_from_manifest(
                local_manifest,
                updated=False,
            )

        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                remote_manifest = (
                    download_release_manifest()
                )

                records = (
                    validate_release_manifest(
                        remote_manifest
                    )
                )

                remote_checksum = (
                    manifest_bundle_checksum(
                        remote_manifest
                    )
                )

                if remote_checksum is None:
                    raise RuntimeError(
                        "The remote manifest has no valid "
                        "bundle checksum."
                    )

                if (
                    local_files_complete
                    and local_checksum
                    == remote_checksum
                ):
                    record_successful_check(
                        remote_checksum
                    )

                    return status_from_manifest(
                        remote_manifest,
                        updated=False,
                    )

                download_and_install_release(
                    manifest=remote_manifest,
                    records=records,
                )

                if not required_release_files_exist():
                    raise RuntimeError(
                        "The installed search-index release "
                        "is incomplete."
                    )

                record_successful_check(
                    remote_checksum
                )

                return status_from_manifest(
                    remote_manifest,
                    updated=True,
                )

            except (
                OSError,
                RuntimeError,
                ValueError,
                requests.RequestException,
            ) as error:
                last_error = error

                if attempt < 3:
                    time.sleep(
                        attempt * 5
                    )

        if (
            local_files_complete
            and local_manifest is not None
        ):
            warning = (
                "The latest search index could not be "
                "checked, so the previously downloaded "
                "index is being used."
            )

            if last_error is not None:
                warning += (
                    f" Details: {last_error}"
                )

            return status_from_manifest(
                local_manifest,
                updated=False,
                warning=warning,
            )

        raise RuntimeError(
            "The search index is unavailable and the latest "
            "release could not be downloaded."
        ) from last_error


def main() -> None:
    """Check the search-index release from the command line."""

    status = ensure_search_index(
        check_interval_seconds=0
    )

    print(
        "Search index is ready."
    )

    print(
        f"Version: {status.version}"
    )

    print(
        "Updated during this check: "
        f"{status.updated}"
    )

    if status.generated_at_utc:
        print(
            "Generated: "
            f"{status.generated_at_utc}"
        )

    if status.warning:
        print(
            f"Warning: {status.warning}"
        )


if __name__ == "__main__":
    main()