"""Facade used by the API, CLI, UI and eval: wires graph + retriever + LLM + audit + agent."""
from __future__ import annotations

from . import config
from .agents import build_agent
from .governance import AuditStore
from .graph_store import BaseGraph, get_graph
from .ingest import ensure_index
from .llm import get_llm
from .retrieval import HybridRetriever


class FamilyInsuranceService:
    def __init__(self, graph: BaseGraph | None = None, store: AuditStore | None = None, llm=None,
                 retriever: HybridRetriever | None = None):
        if retriever is None:
            ensure_index()  # first boot / config change -> build index (fast: ~80 chunks)
        self.graph = graph or get_graph()
        self.store = store or AuditStore()
        self.llm = llm or get_llm()
        self.retriever = retriever or HybridRetriever()
        self.chunks = self.retriever.chunks
        self._ask = build_agent(self.graph, self.retriever, self.llm, self.store)

    def citation_details(self, cites: list[str]) -> list[dict]:
        out = []
        for c in cites:
            ch = self.chunks.get(c)
            if ch:
                out.append({"id": c, "policy_id": ch.policy_id, "policy": ch.policy_name, "clause": ch.clause,
                            "title": ch.title, "text": ch.text})
        return out

    def ask(self, user: str, question: str) -> dict:
        st = self._ask(user, question)
        pending = st["needs_review"]
        return {
            "query_id": st["query_id"], "status": st["status"], "route": st["route"],
            "answer": None if pending else st["answer"],
            "message": ("A human reviewer must approve this answer before you see it. "
                        "Reasons: " + "; ".join(st["review_reasons"])) if pending else None,
            "confidence": st["confidence"], "needs_review": pending, "review_reasons": st["review_reasons"],
            "citations": self.citation_details(st["citations"]), "timings_ms": st["timings"], "trace": st["trace"],
            "_state": st,
        }

    def get_answer(self, query_id: str) -> dict | None:
        rec = self.store.get_query(query_id)
        if rec is None:
            return None
        done = rec["status"] in ("auto_approved", "approved", "approved_with_edits")
        return {"query_id": query_id, "status": rec["status"], "route": rec["route"],
                "answer": rec["final_answer"] if done else None,
                "citations": self.citation_details(rec["citations"]) if done else [],
                "confidence": rec["confidence"]}

    def pending_reviews(self) -> list[dict]:
        return self.store.pending()

    def decide(self, review_id: int, reviewer: str, decision: str, edited_answer: str | None = None, note: str = ""):
        return self.store.decide(review_id, reviewer, decision, edited_answer, note)
