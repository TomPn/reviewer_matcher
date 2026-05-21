"""Semantic Scholar API client — author search, paper fetch, retry logic."""

from __future__ import annotations
import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

BASE_URL     = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,abstract,year,venue,citationCount,authors"

_session: Optional[requests.Session] = None
_request_delay: float = 1.5


def configure(api_key: str = "") -> None:
    global _session, _request_delay
    # Remove 429 from status_forcelist — we handle it manually with backoff
    retries = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],   # ← 429 removed
    )
    _session = requests.Session()
    _session.mount("https://", HTTPAdapter(max_retries=retries))
    if api_key:
        _session.headers.update({"x-api-key": api_key})
    _request_delay = 0.5 if api_key else 3.0    # ← unauthenticated: 3s between requests
    log.info(f"S2 client ready. Authenticated: {bool(api_key)}. Delay: {_request_delay}s")


def _get_session():
    global _session
    if _session is None:
        configure()
    return _session


def _get(url, params):
    time.sleep(_request_delay)
    resp = _get_session().get(url, params=params, timeout=15)

    # Manual 429 handling with exponential backoff
    if resp.status_code == 429:
        for wait in [30, 60, 120]:
            log.warning(f"Rate limited (429). Sleeping {wait}s before retry...")
            time.sleep(wait)
            resp = _get_session().get(url, params=params, timeout=15)
            if resp.status_code != 429:
                break
        else:
            # All retries exhausted
            resp.raise_for_status()

    resp.raise_for_status()
    return resp.json()


def _name_matches(candidate_name: str, full_name: str) -> bool:
    """
    Return True if candidate_name is a plausible match for full_name.

    Handles two common Semantic Scholar name variants:
      1. Exact match:        "Theodora Chaspari" == "Theodora Chaspari"
      2. Abbreviated first:  "T. Chaspari"       matches "Theodora Chaspari"
    """
    c = candidate_name.strip().lower()
    f = full_name.strip().lower()

    if c == f:
        return True

    parts = f.split()
    if len(parts) < 2:
        return False

    first, last = parts[0], parts[-1]

    # Abbreviated first name: "t. chaspari" or "t chaspari"
    abbreviated = f"{first[0]}. {last}"
    abbreviated_no_dot = f"{first[0]} {last}"
    if c in (abbreviated, abbreviated_no_dot):
        return True

    return False

def _affiliation_matches(candidate: dict, affiliation: str) -> bool:
    """
    Check whether any of the candidate's affiliations contain
    the reviewer's institution as a substring (case-insensitive).
    """
    if not affiliation:
        return False
    affil_lower = affiliation.lower()
    for a in candidate.get("affiliations", []):
        if affil_lower in a.lower() or a.lower() in affil_lower:
            return True
    return False


def _best_match(results: list[dict], name: str, affiliation: str = "") -> Optional[dict]:
    """
    From a list of search results, pick the best match for (name, affiliation).

    Priority:
      1. Name match (exact / abbreviated / reversed) + affiliation match
      2. Name match only — highest paperCount wins (handles duplicates/stubs)
      3. None — caller will try next strategy
    """
    name_matched = [r for r in results if _name_matches(r.get("name", ""), name)]

    if not name_matched:
        return None

    # Priority 1: also matches affiliation
    if affiliation:
        affil_matched = [r for r in name_matched if _affiliation_matches(r, affiliation)]
        if affil_matched:
            return max(affil_matched, key=lambda r: r.get("paperCount") or 0)

    # Priority 2: name match only — highest paperCount
    return max(name_matched, key=lambda r: r.get("paperCount") or 0)


def search_author(name: str, affiliation: str = "") -> Optional[dict]:
    """
    Find the best-matching Semantic Scholar author for (name, affiliation).

    Search strategy:
      1. Query "name + affiliation" → best name+affiliation match
      2. Query "name + affiliation" → best name match by paperCount
      3. Query "name only"          → best name+affiliation match
      4. Query "name only"          → best name match by paperCount
      5. Top result (last resort, logged as warning)
    """
    author_fields = "authorId,name,paperCount,affiliations"

    def _search(query: str) -> list[dict]:
        data = _get(
            f"{BASE_URL}/author/search",
            {"query": query, "fields": author_fields, "limit": 10},
        )
        return data.get("data", [])

    queries = []
    if affiliation:
        queries.append(f"{name} {affiliation}")
    queries.append(name)

    for query in queries:
        results = _search(query)
        if not results:
            continue

        best = _best_match(results, name, affiliation)
        if best:
            log.info(
                f"'{name}': matched as '{best['name']}' "
                f"(id={best['authorId']}, papers={best.get('paperCount')}, "
                f"affiliations={best.get('affiliations')})."
            )
            return best

    # Last resort: top result regardless of name
    all_results = _search(name)
    if all_results:
        top = all_results[0]
        log.warning(
            f"'{name}': no name match found via any strategy. "
            f"Falling back to top result: '{top.get('name')}' "
            f"(id={top.get('authorId')}). Verify this is correct."
        )
        return top

    log.warning(f"'{name}': not found on Semantic Scholar.")
    return None


def get_papers_for_author(author_id: str, max_papers: int = 30) -> list[dict]:
    """Fetch up to `max_papers` papers with non-empty abstracts for an author."""
    collected, offset, page_size = [], 0, min(100, max_papers)

    while len(collected) < max_papers:
        data = _get(
            f"{BASE_URL}/author/{author_id}/papers",
            {"fields": PAPER_FIELDS, "limit": page_size, "offset": offset},
        )
        batch = data.get("data", [])
        collected.extend([p for p in batch if (p.get("abstract") or "").strip()])
        if data.get("next") is None or len(batch) < page_size:
            break
        offset += page_size

    return collected[:max_papers]


def fetch_papers_for_reviewer(reviewer: dict, max_papers: int = 30) -> list[dict]:
    """
    Given a reviewer dict (name, affiliation), return their papers
    with reviewer metadata attached.
    """
    name        = reviewer["name"]
    s2_id     = reviewer["s2_id"]
    affiliation = reviewer.get("affiliation", "")

    papers    = get_papers_for_author(s2_id, max_papers=max_papers)

    for p in papers:
        p["reviewer_name"]        = name
        p["reviewer_affiliation"] = affiliation
        p["author_id"]            = s2_id

    return papers