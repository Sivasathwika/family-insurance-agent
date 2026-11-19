"""Central configuration. Everything is overridable via environment variables."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader: KEY=VALUE lines, '#' comments; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("FIA_DATA_DIR", ROOT / "data"))
POLICY_DIR = DATA_DIR / "policies"
FAMILY_JSON = DATA_DIR / "family.json"
GOLD_PATH = DATA_DIR / "eval" / "gold.jsonl"
STORE_DIR = Path(os.getenv("FIA_STORE_DIR", ROOT / ".store"))
# The retrieval index is derived data (rebuilt from data/policies) and may live in the image;
# STORE_DIR holds state that must survive restarts (audit log + review queue) -> mount a disk there.
INDEX_DIR = Path(os.getenv("FIA_INDEX_DIR", STORE_DIR))
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
MODEL_CACHE = os.getenv("FIA_MODEL_CACHE", "") or None
AUDIT_DB = Path(os.getenv("FIA_AUDIT_DB", STORE_DIR / "audit.sqlite"))

# Graph backend: "neo4j" or "memory". Default: neo4j if NEO4J_URI is set.
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GRAPH_BACKEND = os.getenv("FIA_GRAPH_BACKEND", "neo4j" if NEO4J_URI else "memory")

# Retrieval
EMBEDDER = os.getenv("FIA_EMBEDDER", "auto")  # auto | st | fastembed | hash
EMBED_MODEL = os.getenv("FIA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("FIA_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FASTEMBED_RERANK_MODEL = os.getenv("FIA_FASTEMBED_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANKER = os.getenv("FIA_RERANKER", "auto")  # auto | cross | fastembed | lexical
RRF_K = 60
CANDIDATES = 20  # fused candidates passed to the reranker
TOP_K = 5

# LLM
LLM_BACKEND = os.getenv("FIA_LLM", "auto")  # auto | anthropic | openai | extractive
LLM_BASE_URL = os.getenv("FIA_LLM_BASE_URL", "")   # any OpenAI-compatible endpoint (Ollama, Groq, Gemini, ...)
LLM_API_KEY = os.getenv("FIA_LLM_API_KEY", "")
LLM_MODEL = os.getenv("FIA_LLM_MODEL", "claude-sonnet-5-5")

# Governance
CONFIDENCE_THRESHOLD = float(os.getenv("FIA_CONFIDENCE_THRESHOLD", "0.7"))
AMOUNT_THRESHOLD_INR = int(os.getenv("FIA_AMOUNT_THRESHOLD_INR", "100000"))


def today() -> date:
    """Overridable so evaluation is deterministic (FIA_TODAY=2026-11-07)."""
    override = os.getenv("FIA_TODAY")
    return date.fromisoformat(override) if override else date.today()


# Access control for the web app. Unset = open (local dev only). Set both on any public deployment.
ACCESS_KEY = os.getenv("FIA_ACCESS_KEY", "")      # needed to ask questions / read answers
REVIEWER_KEY = os.getenv("FIA_REVIEWER_KEY", "")  # additionally needed for review + audit endpoints
SEED_ON_START = os.getenv("FIA_SEED_ON_START", "true").lower() == "true"
