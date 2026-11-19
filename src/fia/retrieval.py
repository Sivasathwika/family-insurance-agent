"""Hybrid retrieval: dense (Chroma) + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from .embedders import get_embedder, get_reranker, tokenize
from .ingest import Chunk, load_chunk_index

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Retrieved:
    chunk: Chunk
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf: float = 0.0
    rerank: float | None = None

    def audit(self) -> dict:
        return {"chunk": self.chunk.id, "policy": self.chunk.policy_name, "clause": self.chunk.clause,
                "dense_rank": self.dense_rank, "bm25_rank": self.bm25_rank,
                "rrf": round(self.rrf, 5), "rerank": None if self.rerank is None else round(self.rerank, 4)}


def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    """score(d) = sum_i 1 / (k + rank_i(d)), ranks are 1-based. Needs no score normalisation."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return scores


class HybridRetriever:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        import chromadb

        self.chunks = load_chunk_index()
        self.ids = list(self.chunks)
        self.tokens = [tokenize(self.chunks[i].indexed_text) for i in self.ids]
        self.bm25 = BM25Okapi(self.tokens)
        self.embedder = get_embedder()
        self.reranker = get_reranker(self.tokens)
        self.col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection("policies")
        built_with = (self.col.metadata or {}).get("embedder")
        if built_with and built_with != self.embedder.name:
            raise RuntimeError(f"Index was built with '{built_with}' but runtime embedder is "
                               f"'{self.embedder.name}'. Re-run `python -m fia.cli ingest`.")

    @property
    def backends(self) -> dict:
        return {"embedder": self.embedder.name, "reranker": self.reranker.name}

    def _allowed(self, policy_ids, ptypes) -> set[str] | None:
        if not policy_ids and not ptypes:
            return None
        return {i for i, c in self.chunks.items()
                if (not policy_ids or c.policy_id in policy_ids) and (not ptypes or c.policy_type in ptypes)}

    def _dense(self, query: str, allowed: set[str] | None, n: int) -> list[str]:
        qv = self.embedder.encode([query])[0].tolist()
        res = self.col.query(query_embeddings=[qv], n_results=min(len(self.ids), max(n * 3, 60)))
        ids = res["ids"][0]
        return [i for i in ids if allowed is None or i in allowed][:n]

    def _bm25(self, query: str, allowed: set[str] | None, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = []
        for idx in order:
            cid = self.ids[idx]
            if scores[idx] <= 0:
                break
            if allowed is None or cid in allowed:
                out.append(cid)
            if len(out) >= n:
                break
        return out

    def search(self, query: str, mode: str = "hybrid_rerank", top_k: int = config.TOP_K,
               policy_ids: list[str] | None = None, ptypes: list[str] | None = None) -> tuple[list[Retrieved], dict]:
        assert mode in MODES
        t0 = time.perf_counter()
        allowed = self._allowed(policy_ids, ptypes)
        n = config.CANDIDATES
        dense = self._dense(query, allowed, n) if mode != "bm25" else []
        sparse = self._bm25(query, allowed, n) if mode != "dense" else []
        d_rank = {c: r for r, c in enumerate(dense, 1)}
        b_rank = {c: r for r, c in enumerate(sparse, 1)}
        fused = rrf_fuse([r for r in (dense, sparse) if r])
        order = sorted(fused, key=lambda c: -fused[c])
        cands = [Retrieved(self.chunks[c], d_rank.get(c), b_rank.get(c), fused[c]) for c in order[:n]]
        timing = {"candidates_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if mode == "hybrid_rerank" and cands:
            t1 = time.perf_counter()
            scores = self.reranker.score(query, [r.chunk.indexed_text for r in cands])
            for r, s in zip(cands, scores):
                r.rerank = s
            cands.sort(key=lambda r: -r.rerank)
            timing["rerank_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        return cands[:top_k], timing
