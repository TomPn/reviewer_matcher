"""Reviewer ranking pipeline — embed query abstract, search ChromaDB, aggregate scores."""

from __future__ import annotations
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

from . import embedder, vector_store

log = logging.getLogger(__name__)


def _semantic_scholar_url(kind: str, identifier: str) -> str:
    if not identifier:
        return ""
    return f"https://www.semanticscholar.org/{kind}/{identifier}"


def _score_reviewer(similarities, strategy):
    if not similarities:
        return 0.0
    s = sorted(similarities, reverse=True)
    if strategy == "max":
        return s[0]
    elif strategy == "mean_top3":
        top = s[:3]
        return sum(top) / len(top)
    else:  # mean_all
        return sum(similarities) / len(similarities)


def _csv_flag_is_set(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "x"}


def _load_excluded_author_ids(
    csv_path: str,
    *,
    exclude_low_volume: bool,
    exclude_special_issue: bool,
) -> set[str]:
    if not exclude_low_volume and not exclude_special_issue:
        return set()

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Reviewer CSV not found: {csv_path}")

    excluded = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            author_id = row.get("S2 Author ID", "").strip()
            if not author_id:
                continue
            low_volume = _csv_flag_is_set(row.get("Low Volume", ""))
            special_issue = _csv_flag_is_set(row.get("Special Issue", ""))
            if (exclude_low_volume and low_volume) or (exclude_special_issue and special_issue):
                excluded.add(author_id)
    return excluded


def find_reviewers(
    abstract: str, *,
    db_path="data/chroma", collection_name="reviewer_abstracts",
    distance_metric="cosine", model_name="all-MiniLM-L6-v2",
    offline_mode=False, candidate_pool=200, top_n=10,
    scoring_strategy: Literal["mean_top3", "max", "mean_all"] = "mean_top3",
    reviewer_csv_path="reviewer_id_matches.csv",
    exclude_low_volume=False,
    exclude_special_issue=False,
) -> list[dict]:
    if embedder._model is None:
        embedder.load_model(model_name, offline=offline_mode)

    log.info("Embedding query abstract...")
    query_vec = embedder.embed_one(abstract)

    collection = vector_store.get_collection(db_path, collection_name, distance_metric)
    db_size = vector_store.count(collection)
    if db_size == 0:
        log.error("DB is empty. Run `python ingest.py` first.")
        return []

    n_fetch = min(candidate_pool, db_size)
    log.info(f"Querying top {n_fetch} candidates from {db_size} documents...")
    candidates = vector_store.query(query_vec, n_results=n_fetch, collection=collection)
    excluded_author_ids = _load_excluded_author_ids(
        reviewer_csv_path,
        exclude_low_volume=exclude_low_volume,
        exclude_special_issue=exclude_special_issue,
    )

    reviewer_sims   = defaultdict(list)
    reviewer_papers = defaultdict(list)
    reviewer_meta   = {}

    for hit in candidates:
        meta = hit["metadata"]
        author_id = meta.get("author_id", "")
        if author_id in excluded_author_ids:
            continue
        name = meta.get("reviewer_name", "")
        sim  = hit["similarity"]
        reviewer_sims[name].append(sim)
        reviewer_papers[name].append({
            "title":      meta.get("title", ""),
            "similarity": sim,
            "year":       meta.get("year", ""),
            "venue":      meta.get("venue", ""),
            "paper_id":   meta.get("paper_id", ""),
            "paper_url":  _semantic_scholar_url("paper", meta.get("paper_id", "")),
        })
        if name not in reviewer_meta:
            reviewer_meta[name] = {
                "affiliation": meta.get("reviewer_affiliation", ""),
                "author_id": author_id,
                "author_url": _semantic_scholar_url("author", author_id),
            }

    ranked = []
    for name, sims in reviewer_sims.items():
        score = _score_reviewer(sims, scoring_strategy)
        top_paper = max(reviewer_papers[name], key=lambda p: p["similarity"])
        ranked.append({
            "reviewer":           name,
            "affiliation":        reviewer_meta[name]["affiliation"],
            "author_id":          reviewer_meta[name]["author_id"],
            "author_url":         reviewer_meta[name]["author_url"],
            "score":              round(score, 4),
            "top_paper_score":    round(max(sims), 4),
            "matching_papers":    len(sims),
            "top_matching_paper": top_paper,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    return ranked[:top_n]
