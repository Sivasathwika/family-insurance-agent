"""Embedding + reranking backends.

Production: sentence-transformers (bi-encoder + cross-encoder), downloaded from HuggingFace.
Offline fallback: deterministic hashing embedder + lexical reranker, so tests/CI run with no
model downloads. The eval report prints which backend was used - never quote a hit-rate
measured with the fallbacks as your headline number.
"""
from __future__ import annotations

import logging
import math
import re
import zlib
from typing import Protocol

import numpy as np

from . import config

log = logging.getLogger("fia.embed")
_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")
STOP = set("a an the of in on for to and or is are was be by with at as it its this that these those from "
           "under any all our we i my do does did can will what which how when who whom if not no".split())


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    out = []
    for t in toks:
        out.append(t)
        if re.search(r"[./-]", t):  # also index the parts: "shi/ff/2025/004512" -> shi, ff, ...
            out.extend(re.split(r"[./-]", t))
    return [t for t in out if t not in STOP]


class Embedder(Protocol):
    name: str
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    name = "hash-1024 (offline fallback)"
    dim = 1024

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = list(toks) + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for w in toks:
            w = f"#{w}#"
            feats += [w[i:i + 3] for i in range(len(w) - 2)]
        for f in feats:
            v[zlib.crc32(f.encode()) % self.dim] += 1.0
        v = np.sqrt(v)  # sublinear tf
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class STEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.name = f"sentence-transformers:{model}"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class FastEmbedEmbedder:
    """ONNX runtime, no PyTorch: ~10x smaller install and far less RAM. Used for Render deployments."""

    def __init__(self, model: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model, cache_dir=config.MODEL_CACHE)
        self.name = f"fastembed:{model}"

    def encode(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def get_embedder() -> Embedder:
    """Explicit settings (st / fastembed) fail loudly; only 'auto' may degrade to a weaker backend."""
    if config.EMBEDDER == "fastembed":
        return FastEmbedEmbedder(config.EMBED_MODEL)
    if config.EMBEDDER in ("auto", "st"):
        for make in (lambda: STEmbedder(config.EMBED_MODEL), lambda: FastEmbedEmbedder(config.EMBED_MODEL)):
            try:
                return make()
            except Exception as e:  # noqa: BLE001
                if config.EMBEDDER == "st":
                    raise
                log.warning("embedder unavailable (%s)", e)
        log.warning("falling back to hash embedder (offline fallback)")
    return HashEmbedder()


# ----------------------------------------------------------------------------- rerankers
class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)
        self.name = f"cross-encoder:{model}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, d) for d in docs])]


class FastEmbedReranker:
    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=model, cache_dir=config.MODEL_CACHE)
        self.name = f"fastembed-cross-encoder:{model}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(x) for x in self.model.rerank(query, docs)]


class LexicalReranker:
    """IDF-weighted unigram + bigram overlap. Weak stand-in for a cross-encoder."""
    name = "lexical (offline fallback)"

    def __init__(self, corpus_tokens: list[list[str]] | None = None):
        df: dict[str, int] = {}
        for toks in corpus_tokens or []:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.n = max(len(corpus_tokens or []), 1)
        self.df = df

    def _idf(self, t: str) -> float:
        return math.log(1 + self.n / (1 + self.df.get(t, 0)))

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = tokenize(query)
        qb = set(zip(q, q[1:]))
        out = []
        for d in docs:
            dt = tokenize(d)
            ds, db = set(dt), set(zip(dt, dt[1:]))
            uni = sum(self._idf(t) for t in set(q) if t in ds) / (sum(self._idf(t) for t in set(q)) or 1)
            bi = len(qb & db) / (len(qb) or 1)
            out.append(uni + 0.5 * bi)
        return out


def get_reranker(corpus_tokens: list[list[str]]):
    if config.RERANKER == "fastembed":
        return FastEmbedReranker(config.FASTEMBED_RERANK_MODEL)
    if config.RERANKER in ("auto", "cross"):
        for make in (lambda: CrossEncoderReranker(config.RERANK_MODEL),
                     lambda: FastEmbedReranker(config.FASTEMBED_RERANK_MODEL)):
            try:
                return make()
            except Exception as e:  # noqa: BLE001
                if config.RERANKER == "cross":
                    raise
                log.warning("reranker unavailable (%s)", e)
        log.warning("falling back to lexical reranker (offline fallback)")
    return LexicalReranker(corpus_tokens)
