"""ChromaDB persistent vector store — no server, all data lives on disk."""

from __future__ import annotations
import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_client = None
_collection = None


def _get_client(db_path: str):
    global _client
    if _client is None:
        import chromadb
        Path(db_path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=db_path)
        log.info(f"ChromaDB initialised at: '{db_path}'")
    return _client


def get_collection(db_path: str, collection_name: str, distance_metric: str = "cosine"):
    global _collection
    if _collection is not None:
        return _collection
    client = _get_client(db_path)
    _collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric},
    )
    log.info(f"Collection '{collection_name}' ready. Size: {_collection.count()}")
    return _collection


def _make_doc_id(paper: dict) -> str:
    """Unique ID = hash(paperId + reviewer_name), so co-authored papers index per-reviewer."""
    paper_id = paper.get("paperId") or paper.get("title", "unknown")
    reviewer = paper.get("reviewer_name", "unknown")
    return hashlib.md5(f"{paper_id}::{reviewer}".encode()).hexdigest()


def upsert_papers(papers: list[dict], embeddings: list[list[float]], collection) -> int:
    ids, docs, metas, vecs = [], [], [], []
    for paper, vec in zip(papers, embeddings):
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        ids.append(_make_doc_id(paper))
        docs.append(abstract)
        vecs.append(vec)
        metas.append({
            "reviewer_name":        paper.get("reviewer_name", ""),
            "reviewer_affiliation": paper.get("reviewer_affiliation", ""),
            "author_id":            paper.get("author_id", ""),
            "paper_id":             paper.get("paperId", ""),
            "title":                (paper.get("title") or "")[:200],
            "year":                 str(paper.get("year") or ""),
            "venue":                (paper.get("venue") or "")[:100],
            "citation_count":       str(paper.get("citationCount") or 0),
        })
    if not ids:
        return 0
    collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
    return len(ids)


def query(query_vector: list[float], n_results: int, collection) -> list[dict]:
    actual_n = min(n_results, collection.count())
    if actual_n == 0:
        return []
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=actual_n,
        include=["metadatas", "distances", "documents"],
    )
    results = []
    for doc_id, doc, meta, dist in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append({
            "id": doc_id, "document": doc, "metadata": meta,
            "distance": dist, "similarity": 1.0 - dist,
        })
    return results


def count(collection) -> int:
    return collection.count()


def reset_collection(db_path: str, collection_name: str) -> None:
    global _collection
    client = _get_client(db_path)
    try:
        client.delete_collection(collection_name)
        log.warning(f"Collection '{collection_name}' deleted.")
    except Exception:
        pass
    _collection = None