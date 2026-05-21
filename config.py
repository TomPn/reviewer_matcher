import os

# Reviewer ID matches CSV
REVIEWER_ID_MATCHES_CSV = "reviewer_id_matches.csv"

# Semantic Scholar API
# Optional — register free at semanticscholar.org/product/api for 10x rate limit
S2_API_KEY                 = "s2k-DZUfEpbDRRTwJSI4w5TjS1geiAZZk6V6xm0xDpMo"
S2_MAX_PAPERS_PER_REVIEWER = 30

# Embedding Model (local after first download, ~90 MB)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OFFLINE_MODE    = False

# Vector Database (offline)
CHROMA_DB_PATH         = "data/chroma"
CHROMA_COLLECTION_NAME = "reviewer_abstracts"
CHROMA_DISTANCE_METRIC = "cosine"

# Query / Ranking
QUERY_CANDIDATE_POOL = 200
TOP_N_REVIEWERS      = 10
SCORING_STRATEGY     = "mean_top3"  # "mean_top3" | "max" | "mean_all"