from __future__ import annotations

import requests

API_URL = "https://data.nsw.gov.au/data/api/3/action/package_search"


def main() -> None:
    """Confirm that the Data.NSW CKAN catalogue API is accessible."""

    response = requests.get(
        API_URL,
        params={
            "rows": 1,
            "start": 0,
        },
        headers={
            "User-Agent": "nsw-open-data-ai-search/0.1",
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("success") is not True:
        raise RuntimeError(
            f"CKAN API returned an error: {payload.get('error')}"
        )

    result = payload["result"]
    datasets = result.get("results", [])

    print(f"Total datasets reported by API: {result['count']}")

    if not datasets:
        raise RuntimeError("The API returned no dataset records.")

    first_dataset = datasets[0]

    print(f"First dataset title: {first_dataset.get('title')}")
    print(f"First dataset ID: {first_dataset.get('id')}")
    print("API connection test completed successfully.")


if __name__ == "__main__":
    main()