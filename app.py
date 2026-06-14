#!/usr/bin/env python3
"""
Find reviewers whose expertise matches a new paper's abstract.

Usage:
    python3 app.py                             # Interactive prompt
    python3 app.py --abstract "We propose..."  # Inline text
    python3 app.py --file my_abstract.txt      # From file
    python3 app.py --top-n 15 --show-papers 2  # More results
    python3 app.py --strategy max              # Different scoring
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.query import find_reviewers

BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _bar(score, width=20):
    filled = max(0, min(width, round(score * width)))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _print_header(total_docs):
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}  Reviewer Matcher{RESET}  —  {total_docs:,} abstracts indexed")
    print(f"{BOLD}{'═' * 70}{RESET}\n")


def _print_results(results, show_papers=1, scoring_strategy="mean_top3"):
    if not results:
        print(f"{YELLOW}No matching reviewers found. Is the vector DB populated?{RESET}\n")
        return

    strategy_labels = {
        "mean_top3": "mean of top-3 paper similarities",
        "max":       "best single-paper similarity",
        "mean_all":  "mean of all matched paper similarities",
    }
    print(f"{DIM}Scoring: {strategy_labels.get(scoring_strategy, scoring_strategy)}{RESET}\n")

    for r in results:
        score       = r["score"]
        rank        = r["rank"]
        reviewer    = r["reviewer"]
        affiliation = r["affiliation"] or "—"
        n_papers    = r["matching_papers"]
        author_url  = r.get("author_url", "")

        print(
            f"  {BOLD}{rank:>2}.{RESET} {CYAN}{reviewer}{RESET}\n"
            f"      {DIM}{affiliation}{RESET}\n"
            f"      Score: {GREEN}{score:.4f}{RESET}  {_bar(score)}  "
            f"({n_papers} matching paper{'s' if n_papers != 1 else ''})"
        )
        if author_url:
            print(f"      {DIM}AE Semantic Scholar: {author_url}{RESET}")
        if show_papers > 0:
            p = r["top_matching_paper"]
            title = p["title"] or "(untitled)"
            year_str = f"({p.get('year', '')})" if p.get("year") else ""
            print(f"         {DIM} [{p['similarity']:.3f}] {title[:70]} {year_str}{RESET}")
            if p.get("paper_url"):
                print(f"         {DIM}Document: {p['paper_url']}{RESET}")
        print()

    print(f"  {DIM}Tip: run `python app.py --help` for all options.{RESET}\n")


def _read_abstract(args):
    if args.abstract:
        return args.abstract.strip()
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8").strip()
    # Interactive
    print(f"{BOLD}Paste the paper abstract below.{RESET}")
    print(f"{DIM}When done, press Enter twice.{RESET}\n")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _build_parser():
    p = argparse.ArgumentParser(description="Rank reviewers by expertise match to a paper abstract.")
    p.add_argument("--abstract",    "-a", metavar="TEXT")
    p.add_argument("--file",        "-f", metavar="PATH")
    p.add_argument("--top-n",       type=int, default=config.TOP_N_REVIEWERS)
    p.add_argument("--candidates",  type=int, default=config.QUERY_CANDIDATE_POOL)
    p.add_argument("--strategy",    default=config.SCORING_STRATEGY,
                   choices=["mean_top3", "max", "mean_all"])
    p.add_argument("--show-papers", type=int, default=1)
    p.add_argument("--exclude-low-volume", action="store_true",
                   help="Exclude reviewers marked Low Volume in reviewer_id_matches.csv")
    p.add_argument("--exclude-special-issue", action="store_true",
                   help="Exclude reviewers marked Special Issue in reviewer_id_matches.csv")
    p.add_argument("--offline",     action="store_true", default=config.OFFLINE_MODE)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    from src import vector_store
    collection = vector_store.get_collection(
        config.CHROMA_DB_PATH, config.CHROMA_COLLECTION_NAME, config.CHROMA_DISTANCE_METRIC
    )
    total_docs = vector_store.count(collection)
    _print_header(total_docs)

    if total_docs == 0:
        print(f"{YELLOW}  Vector database is empty. Run `python ingest.py` first.{RESET}\n")
        sys.exit(1)

    abstract = _read_abstract(args)
    if not abstract:
        print("No abstract provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{DIM}Abstract received ({len(abstract.split())} words). Searching...{RESET}\n")
    results = find_reviewers(
        abstract,
        db_path=config.CHROMA_DB_PATH,
        collection_name=config.CHROMA_COLLECTION_NAME,
        distance_metric=config.CHROMA_DISTANCE_METRIC,
        model_name=config.EMBEDDING_MODEL,
        offline_mode=args.offline,
        candidate_pool=args.candidates,
        top_n=args.top_n,
        scoring_strategy=args.strategy,
        reviewer_csv_path=config.REVIEWER_ID_MATCHES_CSV,
        exclude_low_volume=args.exclude_low_volume,
        exclude_special_issue=args.exclude_special_issue,
    )
    _print_results(results, show_papers=args.show_papers, scoring_strategy=args.strategy)


if __name__ == "__main__":
    main()
