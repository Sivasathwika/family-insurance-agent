"""Audit log (tamper-evident hash chain) + human review queue, backed by SQLite."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT UNIQUE, ts TEXT, user TEXT, query TEXT,
  route TEXT, status TEXT, needs_review INTEGER, record TEXT, prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS audit_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT, ts TEXT, event TEXT, actor TEXT, payload TEXT,
  prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS review_queue (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT UNIQUE, ts TEXT, status TEXT,
  reasons TEXT, draft_answer TEXT, final_answer TEXT, reviewer TEXT, note TEXT, decided_at TEXT);
"""
now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731


def _h(prev: str, payload: str) -> str:
    return hashlib.sha256((prev + payload).encode()).hexdigest()


class AuditStore:
    def __init__(self, path=config.AUDIT_DB):
        path = str(path)
        if path != ":memory:":
            config.STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.db.executescript(SCHEMA)

    # ---- writes
    def _last_hash(self, table: str) -> str:
        r = self.db.execute(f"SELECT hash FROM {table} ORDER BY seq DESC LIMIT 1").fetchone()
        return r["hash"] if r else "GENESIS"

    def log_query(self, rec: dict) -> None:
        payload = json.dumps(rec, ensure_ascii=False, default=str, sort_keys=True)
        with self.lock, self.db:
            prev = self._last_hash("audit_log")
            self.db.execute(
                "INSERT INTO audit_log(query_id,ts,user,query,route,status,needs_review,record,prev_hash,hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (rec["query_id"], rec["ts"], rec["user"], rec["query"], rec["route"], rec["status"],
                 int(rec["needs_review"]), payload, prev, _h(prev, payload)))
            if rec["needs_review"]:
                self.db.execute(
                    "INSERT INTO review_queue(query_id,ts,status,reasons,draft_answer) VALUES(?,?,?,?,?)",
                    (rec["query_id"], rec["ts"], "pending", json.dumps(rec["review_reasons"]), rec["answer"]))

    def _event(self, query_id: str, event: str, actor: str, payload: dict) -> None:
        body = json.dumps({"query_id": query_id, "event": event, "actor": actor, **payload},
                          ensure_ascii=False, sort_keys=True)
        prev = self._last_hash("audit_events")
        self.db.execute("INSERT INTO audit_events(query_id,ts,event,actor,payload,prev_hash,hash) VALUES(?,?,?,?,?,?,?)",
                        (query_id, now(), event, actor, body, prev, _h(prev, body)))

    def decide(self, review_id: int, reviewer: str, decision: str, edited_answer: str | None = None,
               note: str = "") -> dict:
        assert decision in ("approve", "edit", "reject")
        with self.lock, self.db:
            r = self.db.execute("SELECT * FROM review_queue WHERE review_id=?", (review_id,)).fetchone()
            if r is None:
                raise KeyError(f"review {review_id} not found")
            if r["status"] != "pending":
                raise ValueError(f"review {review_id} already {r['status']}")
            if decision == "edit" and not edited_answer:
                raise ValueError("edited_answer required for decision=edit")
            final = {"approve": r["draft_answer"], "edit": edited_answer, "reject": None}[decision]
            status = {"approve": "approved", "edit": "edited", "reject": "rejected"}[decision]
            self.db.execute("UPDATE review_queue SET status=?, final_answer=?, reviewer=?, note=?, decided_at=? "
                            "WHERE review_id=?", (status, final, reviewer, note, now(), review_id))
            self._event(r["query_id"], "review_decision", reviewer,
                        {"decision": decision, "final_answer": final, "draft_answer": r["draft_answer"], "note": note})
        return {"review_id": review_id, "query_id": r["query_id"], "status": status}

    # ---- reads
    def pending(self) -> list[dict]:
        rows = self.db.execute("SELECT r.*, a.query, a.user, a.record FROM review_queue r JOIN audit_log a "
                               "USING(query_id) WHERE r.status='pending' ORDER BY r.review_id").fetchall()
        out = []
        for r in rows:
            rec = json.loads(r["record"])
            out.append({"review_id": r["review_id"], "query_id": r["query_id"], "user": r["user"], "query": r["query"],
                        "reasons": json.loads(r["reasons"]), "draft_answer": r["draft_answer"],
                        "confidence": rec["confidence"], "citations": rec["citations"],
                        "retrieved": rec["retrieved"], "graph": rec["graph"]})
        return out

    def get_query(self, query_id: str) -> dict | None:
        a = self.db.execute("SELECT record FROM audit_log WHERE query_id=?", (query_id,)).fetchone()
        if a is None:
            return None
        rec = json.loads(a["record"])
        r = self.db.execute("SELECT * FROM review_queue WHERE query_id=?", (query_id,)).fetchone()
        rec["review"] = dict(r) if r else None
        if r:
            rec["status"] = {"pending": "pending_review", "approved": "approved", "edited": "approved_with_edits",
                             "rejected": "rejected"}[r["status"]]
            rec["final_answer"] = r["final_answer"]
        else:
            rec["final_answer"] = rec["answer"]
        return rec

    def recent(self, limit: int = 50) -> list[dict]:
        """Newest first; `status` reflects the reviewer's decision once one exists."""
        rows = self.db.execute(
            "SELECT a.query_id, a.ts, a.user, a.query, a.route, a.status AS logged_status, a.needs_review, a.hash, "
            "r.status AS review_status FROM audit_log a LEFT JOIN review_queue r USING(query_id) "
            "ORDER BY a.seq DESC LIMIT ?", (limit,)).fetchall()
        names = {"pending": "pending_review", "approved": "approved", "edited": "approved_with_edits", "rejected": "rejected"}
        out = []
        for r in rows:
            d = dict(r)
            review, logged = d.pop("review_status"), d.pop("logged_status")
            d["status"] = names[review] if review else logged
            out.append(d)
        return out

    def verify_chain(self) -> dict:
        """Recompute both hash chains; any edited/deleted row breaks verification."""
        ok, bad = True, []
        for table, col in (("audit_log", "record"), ("audit_events", "payload")):
            prev = "GENESIS"
            for r in self.db.execute(f"SELECT seq, {col} AS body, prev_hash, hash FROM {table} ORDER BY seq"):
                if r["prev_hash"] != prev or r["hash"] != _h(prev, r["body"]):
                    ok = False
                    bad.append((table, r["seq"]))
                prev = r["hash"]
        return {"ok": ok, "broken_at": bad}
