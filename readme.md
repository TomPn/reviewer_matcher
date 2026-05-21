# Reviewer Matcher

A tool for recommending paper reviewers based on semantic similarity between a submitted paper's abstract and a pool of reviewers' publication histories.

***

## How It Works

1. Reviewer papers and abstracts are fetched from Semantic Scholar and embedded into a local vector database (ChromaDB).
2. At query time, a submitted paper's abstract is embedded and matched against the stored reviewer embeddings.
3. The top-N most relevant reviewers are returned ranked by similarity.

***

## Project Structure

```
reviewer_matcher/
├── app.py                    # Query entry point
├── ingest.py                 # Ingestion entry point
├── config.py                 # All configurable settings
├── reviewer_id_matches.csv   # Manually verified reviewer -> S2 Author ID mapping
├── data/chroma/              # ChromaDB vector database
└── src/
    ├── ingestion.py          # Ingestion pipeline logic
    ├── semantic_scholar.py   # Semantic Scholar API client
    ├── embedder.py           # Sentence embedding model
    ├── vector_store.py       # ChromaDB interface
    └── query.py              # Query and ranking logic
```

***

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

***

## reviewer_id_matches.csv

This file maps each reviewer to their verified Semantic Scholar Author ID. It must be in the **project root**.

Required columns:

| Column | Description |
|---|---|
| `Input Name` | Reviewer's full name |
| `Input Affiliation` | Reviewer's institution |
| `Matched Name` | Name as it appears on Semantic Scholar |
| `S2 Author ID` | Semantic Scholar author ID (from their profile URL) |

Rows with a blank `S2 Author ID` are skipped during ingestion.

### Finding an Author's S2 Author ID

1. Search one of the author's known paper titles on [semanticscholar.org](https://www.semanticscholar.org).
2. Open the paper's page and click the author's name.
3. On their author profile page, copy the numeric ID from the URL:

```
https://www.semanticscholar.org/author/Sharifa-Alghowinem/145001530
                                                            ^ this is the ID
```
***

## Ingesting Reviewers

Populate the vector database from `reviewer_id_matches.csv`:

```bash
python3 ingest.py
```

To add a new reviewer, add their row to `reviewer_id_matches.csv` then rebuild the database:

```bash
python3 ingest.py --reset
```


***

## Querying

Find the best reviewers for a submitted paper abstract:

```bash
python3 app.py --abstract "Your paper abstract here"
```