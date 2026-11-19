import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Tests must ignore your personal .env (keys, Neo4j, LLM). Setting a variable - even to "" - stops the .env loader
# from overriding it, because the loader only fills in variables that are not already set.
for _k in ("FIA_ACCESS_KEY", "FIA_REVIEWER_KEY", "NEO4J_URI", "FIA_LLM_BASE_URL", "FIA_LLM_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ[_k] = ""
os.environ["FIA_GRAPH_BACKEND"] = "memory"
os.environ.setdefault("FIA_TODAY", "2026-11-07")
os.environ.setdefault("FIA_EMBEDDER", "hash")      # tests never download models
os.environ.setdefault("FIA_RERANKER", "lexical")
os.environ.setdefault("FIA_LLM", "extractive")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def svc(tmp_path_factory):
    store_dir = tmp_path_factory.mktemp("store")
    from fia import config
    config.STORE_DIR = store_dir
    config.CHROMA_DIR = store_dir / "chroma"
    config.CHUNKS_PATH = store_dir / "chunks.json"
    from fia.governance import AuditStore
    from fia.ingest import ingest
    from fia.service import FamilyInsuranceService
    ingest()
    return FamilyInsuranceService(store=AuditStore(":memory:"))