"""Full ingestion pipeline: Semantic Scholar -> embed -> ChromaDB."""

from __future__ import annotations
import logging
import sys

from tqdm import tqdm

from . import embedder, vector_store
from .semantic_scholar import configure as s2_configure, fetch_papers_for_reviewer
import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)

def load_reviewers_from_csv(csv_path: str) -> list[dict]:
    """
    Read reviewer_id_matches.csv (manually verified).

    Expected columns:
        Input Name, Input Affiliation, Matched Name, S2 Author ID

    Rows with a blank S2 Author ID are skipped (not found / not verified).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Reviewer CSV not found: {csv_path}")

    reviewers = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            author_id = row.get("S2 Author ID", "").strip()
            if not author_id:
                log.warning(f"Skipping '{row.get('Input Name')}' — no S2 Author ID.")
                continue
            reviewers.append({
                "name":        row["Input Name"].strip(),
                "affiliation": row.get("Input Affiliation", "").strip(),
                "s2_id":       author_id,
            })

    log.info(f"Loaded {len(reviewers)} reviewers from {csv_path}.")
    return reviewers

def run(
    *,
    csv_path,
    # Semantic Scholar
    s2_api_key="",
    max_papers=30,
    # Embedding
    model_name="all-MiniLM-L6-v2",
    offline_mode=False,
    # Vector DB
    db_path="data/chroma",
    collection_name="reviewer_abstracts",
    distance_metric="cosine",
    # Options
    reset=False,
):
    # Configure S2 API
    s2_configure(s2_api_key)

    # Load embedding model
    embedder.load_model(model_name, offline=offline_mode)

    # Open / reset vector DB
    if reset:
        log.warning("--reset: clearing existing collection.")
        vector_store.reset_collection(db_path, collection_name)

    collection = vector_store.get_collection(db_path, collection_name, distance_metric)
    initial_count = vector_store.count(collection)
    log.info(f"Vector DB currently holds {initial_count} documents.")

    # Load reviewers from CSV
    reviewers = load_reviewers_from_csv(csv_path)

    if not reviewers:
        log.error("No reviewers loaded. Check your csv path.")
        sys.exit(1)

    log.info(f"Loaded {len(reviewers)} reviewers.")

    # Fetch papers from Semantic Scholar
    all_papers, failed = [], []

    for reviewer in tqdm(reviewers, desc="Fetching from Semantic Scholar", unit="reviewer"):
        try:
            papers = fetch_papers_for_reviewer(reviewer, max_papers=max_papers)
            all_papers.extend(papers)
        except Exception as exc:
            log.warning(f"Error fetching papers for '{reviewer['name']}': {exc}")
            failed.append(reviewer["name"])

    log.info(f"Total papers with abstracts fetched: {len(all_papers)}")
    if failed:
        log.warning(f"Skipped {len(failed)} reviewers due to errors: {failed}")

    if not all_papers:
        log.error(
            "No papers collected. "
            "Verify reviewer names are correct and covered by Semantic Scholar."
        )
        sys.exit(1)

    # Embed abstracts locally
    log.info("Generating embeddings (local inference)...")
    abstracts = [p["abstract"] for p in all_papers]
    vectors   = embedder.embed(abstracts, show_progress=True)

    # Store in ChromaDB
    log.info("Storing in ChromaDB...")
    inserted    = vector_store.upsert_papers(all_papers, vectors, collection)
    final_count = vector_store.count(collection)

    log.info(
        f"Ingestion complete. "
        f"Upserted: {inserted}. "
        f"DB size: {initial_count} → {final_count}."
    )

    # Summary
    reviewers_indexed = {p["reviewer_name"] for p in all_papers}
    not_indexed       = {r["name"] for r in reviewers} - reviewers_indexed

    print(f"\n✓ Indexed papers for {len(reviewers_indexed)} / {len(reviewers)} reviewers.")
    if not_indexed:
        print("  Reviewers with no papers found on Semantic Scholar:")
        for name in sorted(not_indexed):
            print(f"    • {name}")
    print(f"  Total documents in DB: {final_count}\n")