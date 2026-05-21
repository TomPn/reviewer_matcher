#!/usr/bin/env python3
"""
ingest.py — Populate ChromaDB from reviewer_id_matches.csv.

Usage:
    python3 ingest.py                                       # Populate DB using config defaults (see config.py)
    python3 ingest.py --csv path/to/reviewer_id_matches.csv # Custom CSV path
    python3 ingest.py --s2-key YOUR_KEY --verbose           # More logging
    python3 ingest.py --reset                               # Clear existing DB collection before ingesting
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from src.ingestion import run


def main():
    parser = argparse.ArgumentParser(description="Ingest reviewer papers into ChromaDB.")
    parser.add_argument("--csv",     default=config.REVIEWER_ID_MATCHES_CSV,
                        help="Path to reviewer_id_matches.csv")
    parser.add_argument("--s2-key", default=config.S2_API_KEY,
                        help="Semantic Scholar API key")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing ChromaDB collection before ingesting")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    run(
        csv_path = args.csv,
        s2_api_key = args.s2_key,
        max_papers = config.S2_MAX_PAPERS_PER_REVIEWER,
        model_name = config.EMBEDDING_MODEL,
        offline_mode = config.OFFLINE_MODE,
        db_path = config.CHROMA_DB_PATH,
        collection_name = config.CHROMA_COLLECTION_NAME,
        distance_metric = config.CHROMA_DISTANCE_METRIC,
        reset = args.reset,
    )


if __name__ == "__main__":
    main()