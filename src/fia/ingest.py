"""Policy ingestion: markdown -> clause-level chunks -> Chroma (dense) + chunks.json (BM25)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from . import config
from .embedders import Embedder, get_embedder

MAX_WORDS = 220


@dataclass
class Chunk:
    id: str          # e.g. "P1§2.2"
    policy_id: str
    policy_name: str
    policy_type: str
    clause: str      # "2.2" or "S"
    title: str
    text: str        # body only
    page: int = 1

    @property
    def indexed_text(self) -> str:
        """Contextual chunk text: clause header is part of what we index/embed."""
        return f"{self.policy_name} | Clause {self.clause} {self.title}\n{self.text}"

    @property
    def citation(self) -> str:
        return f"[{self.policy_id}§{self.clause}]"


def _front_matter(raw: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines()) if m else {}
    return {k.strip(): v.strip() for k, v in meta.items()}, (m.group(2) if m else raw)


def chunk_policy(raw: str) -> list[Chunk]:
    meta, body = _front_matter(raw)
    chunks: list[Chunk] = []
    for sec in re.split(r"^## ", body, flags=re.M)[1:]:
        head, _, text = sec.partition("\n")
        clause, _, title = head.partition(" ")
        clause = clause.rstrip(".")
        words = text.split()
        parts = [text.strip()] if len(words) <= MAX_WORDS else [
            " ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS - 30)]
        for i, part in enumerate(parts):
            cid = f'{meta["policy_id"]}§{clause}' + (f"#{i}" if len(parts) > 1 else "")
            chunks.append(Chunk(cid, meta["policy_id"], meta["name"], meta["type"], clause, title.strip(), part))
    return chunks


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in sorted(config.POLICY_DIR.glob("*.md")):
        chunks.extend(chunk_policy(f.read_text(encoding="utf-8")))
    return chunks


def ingest(embedder: Embedder | None = None) -> dict:
    import chromadb

    embedder = embedder or get_embedder()
    chunks = load_chunks()
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection("policies")
    except Exception:  # noqa: BLE001
        pass
    col = client.create_collection("policies", metadata={"hnsw:space": "cosine", "embedder": embedder.name})
    vecs = embedder.encode([c.indexed_text for c in chunks])
    col.add(ids=[c.id for c in chunks], embeddings=vecs.tolist(),
            documents=[c.indexed_text for c in chunks],
            metadatas=[{"policy_id": c.policy_id, "policy_type": c.policy_type, "clause": c.clause} for c in chunks])
    return {"chunks": len(chunks), "policies": len({c.policy_id for c in chunks}), "embedder": embedder.name}


def load_chunk_index() -> dict[str, Chunk]:
    return {d["id"]: Chunk(**d) for d in json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))}


def ensure_index(embedder: Embedder | None = None) -> dict:
    """Build the index if it is missing or was built with a different embedder (e.g. after a config change)."""
    import chromadb

    embedder = embedder or get_embedder()
    try:
        col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection("policies")
        ok = config.CHUNKS_PATH.exists() and (col.metadata or {}).get("embedder") == embedder.name \
            and col.count() == len(load_chunks())
    except Exception:  # noqa: BLE001
        ok = False
    return {"rebuilt": False, "embedder": embedder.name} if ok else {"rebuilt": True, **ingest(embedder)}
