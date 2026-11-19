"""FastAPI service + built-in web UI.  Run:  uvicorn fia.api:app --port 8000   (UI at /, API docs at /docs)

Auth (set both on any public deployment; unset = open, for local dev):
  X-Access-Key    -> FIA_ACCESS_KEY    required for /ask and /queries/*
  X-Reviewer-Key  -> FIA_REVIEWER_KEY  additionally required for /reviews/* and /audit*
"""
from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .service import FamilyInsuranceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fia.api")
STATIC = Path(__file__).parent / "static"
svc: FamilyInsuranceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svc
    svc = svc or FamilyInsuranceService()
    log.info("ready | graph=%s llm=%s embedder=%s reranker=%s | access_key=%s reviewer_key=%s",
             svc.graph.backend, svc.llm.name, svc.retriever.backends["embedder"], svc.retriever.backends["reranker"],
             bool(config.ACCESS_KEY), bool(config.REVIEWER_KEY))
    if not config.ACCESS_KEY:
        log.warning("FIA_ACCESS_KEY is not set: the API is OPEN. Fine locally, never on the public internet.")
    yield


app = FastAPI(title="Family Insurance Agent", version="0.2.0", lifespan=lifespan)


def _same(given: str | None, expected: str) -> bool:
    return hmac.compare_digest((given or "").encode(), expected.encode())


def require_access(x_access_key: str | None = Header(default=None)):
    if config.ACCESS_KEY and not _same(x_access_key, config.ACCESS_KEY):
        raise HTTPException(401, "invalid or missing access key")


def require_reviewer(x_access_key: str | None = Header(default=None), x_reviewer_key: str | None = Header(default=None)):
    require_access(x_access_key)
    if config.REVIEWER_KEY and not _same(x_reviewer_key, config.REVIEWER_KEY):
        raise HTTPException(401, "invalid or missing reviewer key")


class AskRequest(BaseModel):
    user: str = Field(..., min_length=1, max_length=60, examples=["Sathwika"])
    question: str = Field(..., min_length=3, max_length=500, examples=["Which policies cover Mom?"])


class DecisionRequest(BaseModel):
    reviewer: str = Field(..., min_length=1, max_length=60)
    decision: Literal["approve", "edit", "reject"]
    edited_answer: str | None = Field(default=None, max_length=4000)
    note: str = Field(default="", max_length=500)


def _svc() -> FamilyInsuranceService:
    if svc is None:
        raise HTTPException(503, "service not ready")
    return svc


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    s = _svc()
    return {"ok": True, "graph": s.graph.backend, "llm": s.llm.name, **s.retriever.backends,
            "auth": {"access_key": bool(config.ACCESS_KEY), "reviewer_key": bool(config.REVIEWER_KEY)}}


@app.post("/ask", dependencies=[Depends(require_access)])
def ask(req: AskRequest):
    r = _svc().ask(req.user, req.question)
    r.pop("_state")
    return r


@app.get("/queries/{query_id}", dependencies=[Depends(require_access)])
def get_query(query_id: str):
    r = _svc().get_answer(query_id)
    if r is None:
        raise HTTPException(404, "unknown query_id")
    return r


@app.get("/reviews/pending", dependencies=[Depends(require_reviewer)])
def pending():
    return _svc().pending_reviews()


@app.post("/reviews/{review_id}/decision", dependencies=[Depends(require_reviewer)])
def decide(review_id: int, req: DecisionRequest):
    try:
        return _svc().decide(review_id, req.reviewer, req.decision, req.edited_answer, req.note)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/audit", dependencies=[Depends(require_reviewer)])
def audit(limit: int = 50):
    return _svc().store.recent(min(max(limit, 1), 500))


@app.get("/audit/{query_id}", dependencies=[Depends(require_reviewer)])
def audit_detail(query_id: str):
    r = _svc().store.get_query(query_id)
    if r is None:
        raise HTTPException(404, "unknown query_id")
    return r


@app.get("/audit-verify", dependencies=[Depends(require_reviewer)])
def audit_verify():
    return _svc().store.verify_chain()
