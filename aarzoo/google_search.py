import requests
import os
from typing import List

class GoogleSerperSearch:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.endpoint = "https://google.serper.dev/search"

    def search(self, query: str, num_results: int = 3) -> List[str]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query}
        resp = requests.post(self.endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", [])[:num_results]:
            snippet = item.get("snippet")
            if snippet:
                results.append(snippet)
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Serper AI Google Search")
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--api_key', type=str, default=None)
    args = parser.parse_args()
    searcher = GoogleSerperSearch(api_key=args.api_key)
    results = searcher.search(args.query)
    for i, snippet in enumerate(results):
        print(f"Result {i+1}: {snippet}")
