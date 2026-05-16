from typing import Optional

import requests


def fetch_image_url(query: str, access_key: str) -> Optional[str]:
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "per_page": 1, "client_id": access_key}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return None
        return results[0]["urls"]["regular"]
    except Exception:
        return None
