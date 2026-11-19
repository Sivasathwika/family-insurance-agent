"""LangGraph orchestration.

    classify ─┬─ structured ──────────► graph ──────────────┐
              ├─ comparative ─► graph ─► (rag if needed) ───┤
              ├─ clause ─────► [scope graph] ─► rag ────────┼─► generate ─► verify ─► risk_gate ─► audit
              └─ high_stakes ─► graph ─► rag ───────────────┘

Every node appends to `trace` and `timings`, so each answer records which agents ran and how long each took.
"""
from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import config, family
from .embedders import STOP, tokenize
from .family import Entities, inr
from .governance import AuditStore
from .graph_store import BaseGraph
from .retrieval import HybridRetriever, Retrieved

CITE = re.compile(r"\[(P\d+)§([\w.]+)\]")
RISK_KW = ["surgery", "death", " dies", " die ", "died", "cancer", "icu", "claim", "heart attack", "stroke",
           "transplant", "dialysis", "hospitali", "accident", "replacement", "emergency"]
CLAIM_DECISION = re.compile(r"(should (we|i) (file|claim|make)|file a claim|filing a claim|can (we|i) claim|claim for)")
COMPARATIVE = re.compile(r"\b(compare|comparison|versus|vs|shortest|longest|lowest|highest|cheapest|across|differ\w*)\b")


class State(TypedDict, total=False):
    query_id: str
    user: str
    query: str
    ts: str
    route: str
    entities: dict
    graph_runs: list[dict]
    graph_rows: dict
    scope_policy_ids: list[str] | None
    graph_lines: list[str]
    graph_answerable: bool
    retrieved: list[Retrieved]
    retrieval_timing: dict
    answer: str
    llm_part: str
    confidence: float
    verification: dict
    citations: list[str]
    review_reasons: list[str]
    needs_review: bool
    status: str
    timings: dict
    trace: list[str]


def timed(name):
    def deco(fn):
        @wraps(fn)
        def wrapper(state: State) -> dict:
            t0 = time.perf_counter()
            out = fn(state) or {}
            timings = dict(state.get("timings", {}))
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)
            return {**out, "timings": timings, "trace": [*state.get("trace", []), name]}
        return wrapper
    return deco


# ----------------------------------------------------------------------------- planning
def _days_in(q: str) -> int:
    m = re.search(r"(\d+)\s*(day|week|month)", q)
    if not m:
        return 30
    return int(m.group(1)) * {"day": 1, "week": 7, "month": 30}[m.group(2)]


def plan_graph(query: str, e: Entities) -> tuple[str, dict] | None:
    """Map a question to ONE named Cypher template (+params), or None if the graph cannot answer it."""
    ql = query.lower()
    members = e.member_ids
    if "nominee" in ql:
        return "nominees", {}
    if re.search(r"\b(renewals?|renew|due)\b", ql) and re.search(r"\b(due|upcoming|next|within|soon|coming)\b", ql) \
            and "grace" not in ql:
        today = config.today()
        return "renewals_due", {"today": today.isoformat(), "cutoff": (today + timedelta(days=_days_in(ql))).isoformat()}
    if re.search(r"(total|overall|altogether).*premium|premium.*(total|altogether|in all)|how much.*premium|annual premium", ql):
        return "total_premium", {}
    if ("waiting" in ql and not e.disease_terms) or (e.condition == "suicide" and COMPARATIVE.search(ql)):
        return "waiting_periods", {"condition": e.condition}
    if e.coverage and members and re.search(r"\bcover", ql):
        return "coverage_for_member", {"coverage": e.coverage}
    if e.coverage and re.search(r"which (of our |of the )?(polic|plan)", ql):
        return "coverage_any", {"coverage": e.coverage}
    if members and re.search(r"which polic|\bcover(s|ed|age)?\b|insured under|policies", ql):
        return "policies_for_member", {}
    if re.search(r"\b(list|all|every)\b.*polic|how many polic", ql):
        return "list_policies", {}
    return None


def classify_query(query: str, e: Entities) -> str:
    ql = " " + query.lower().replace("’", "'") + " "
    if CLAIM_DECISION.search(ql) or (e.member_ids and any(k in ql for k in RISK_KW)):
        return "high_stakes"
    if COMPARATIVE.search(ql):
        return "comparative"
    return "structured" if plan_graph(query, e) else "clause"


# ----------------------------------------------------------------------------- formatting
def _cite(pid: str, clause: str = "S") -> str:
    return f"[{pid}§{clause}]"


def format_graph(template: str, rows: list[dict], query: str, e: Entities) -> list[str]:
    ql = query.lower()
    if template == "total_premium":
        r = rows[0] if rows else {"total": 0, "n": 0, "breakdown": []}
        if not r["n"]:
            return []
        lines = [f"Total annual premium across {r['n']} polic{'y' if r['n'] == 1 else 'ies'}: **{inr(r['total'])}**."]
        lines += [f"- {b['policy']}: {inr(b['premium'])} {_cite(b['policy_id'])}" for b in r["breakdown"]]
        return lines
    if not rows:
        return []
    if template in ("policies_for_member", "coverage_for_member"):
        head = f"{rows[0]['member']} is covered by {len(rows)} polic{'y' if len(rows) == 1 else 'ies'}"
        head += f" with {e.coverage.replace('_', ' ')} cover:" if template == "coverage_for_member" else ":"
        return [head] + [f"- {r['policy']} ({r['type'].replace('_', ' ')}, {r['sum_insured']}) {_cite(r['policy_id'])}" for r in rows]
    if template in ("coverage_any", "list_policies"):
        head = (f"Policies with {e.coverage.replace('_', ' ')} cover:" if template == "coverage_any"
                else f"All {len(rows)} policies in the family portfolio:")
        return [head] + [f"- {r['policy']} ({r['sum_insured']}); covers: {', '.join(r['members'])} {_cite(r['policy_id'])}" for r in rows]
    if template == "renewals_due":
        today = config.today()
        lines = [f"{len(rows)} polic{'y is' if len(rows) == 1 else 'ies are'} due for renewal within the window "
                 f"(as of {today.isoformat()}):"]
        for r in rows:
            d = (datetime.fromisoformat(r["renewal_date"]).date() - today).days
            lines.append(f"- {r['policy']}: due {r['renewal_date']} (in {d} days), premium {inr(r['premium'])} {_cite(r['policy_id'])}")
        return lines
    if template == "nominees":
        return ["Nominees on record:"] + [f"- {r['policy']}: {r['nominee']} ({r['relation']}) {_cite(r['policy_id'])}" for r in rows]
    if template == "waiting_periods":
        cond = (e.condition or "all").replace("_", "-" if e.condition == "pre_existing" else " ")
        lines = []
        if COMPARATIVE.search(ql) and e.condition:
            if re.search(r"shortest|lowest|least|minimum", ql):
                best = [r for r in rows if r["days"] == min(x["days"] for x in rows)]
                tag = "Shortest"
            else:
                best = [r for r in rows if r["days"] == max(x["days"] for x in rows)]
                tag = "Longest"
            lines.append(f"{tag} {cond} waiting period: "
                         + "; ".join(f"**{r['policy']}** ({r['label']}) {_cite(r['policy_id'], r['clause'])}" for r in best) + ".")
            lines.append("Full comparison:")
        else:
            lines.append(f"Waiting periods ({cond}):")
        lines += [f"- {r['policy']}: {r['condition'].replace('_', '-' if r['condition'] == 'pre_existing' else ' ')} - {r['label']} {_cite(r['policy_id'], r['clause'])}" for r in rows]
        return lines
    return []


def _cites_in(lines: list[str]) -> list[str]:
    seen, out = set(), []
    for l in lines:
        for pid, cl in CITE.findall(l):
            if f"{pid}§{cl}" not in seen:
                seen.add(f"{pid}§{cl}")
                out.append(f"{pid}§{cl}")
    return out


# ----------------------------------------------------------------------------- agent
def build_agent(graph: BaseGraph, retriever: HybridRetriever, llm, store: AuditStore):
    def ents(state: State) -> Entities:
        return Entities(**state["entities"])

    @timed("classify")
    def classify(state: State):
        e = family.extract_entities(state["query"])
        return {"entities": e.__dict__, "route": classify_query(state["query"], e),
                "graph_runs": [], "graph_rows": {}, "graph_lines": [], "graph_answerable": False,
                "scope_policy_ids": None, "retrieved": [], "retrieval_timing": {}}

    @timed("graph")
    def graph_node(state: State):
        e, q = ents(state), state["query"]
        runs, rows_by = list(state["graph_runs"]), dict(state["graph_rows"])

        def run(template, **params):
            res = graph.run(template, **params)
            runs.append(res.audit())
            rows_by[template] = res.rows
            return res

        # 1) policies covering the named member(s) -> scope for later steps
        scope: list[str] | None = None
        if e.member_ids:
            ids: list[str] = []
            for m in e.member_ids:
                ids += [r["policy_id"] for r in run("policies_for_member", member_id=m).rows]
            scope = sorted(set(ids))
        if e.policy_ids:
            scope = sorted(set(scope) & set(e.policy_ids)) or sorted(e.policy_ids) if scope else sorted(e.policy_ids)

        # 2) the planned template, if the graph can answer the question itself
        planned = plan_graph(q, e)
        lines: list[str] = []
        answerable = False
        if planned:
            template, params = planned
            params = {**params}
            if template in ("nominees", "waiting_periods", "total_premium"):
                params["policy_ids"], params["ptypes"] = scope, e.ptypes
            if template in ("policies_for_member", "coverage_for_member"):
                lines, found = [], False
                for m in e.member_ids:
                    res = run(template, member_id=m, **params)
                    found = found or bool(res.rows)
                    lines += format_graph(template, res.rows, q, e)
                answerable = found
            else:
                res = run(template, **params)
                lines = format_graph(template, res.rows, q, e)
                answerable = bool(res.rows)
                if template in ("coverage_any", "list_policies", "renewals_due") and res.rows:
                    scope = scope or None
        return {"graph_runs": runs, "graph_rows": rows_by, "scope_policy_ids": scope,
                "graph_lines": lines, "graph_answerable": answerable}

    @timed("rag")
    def rag_node(state: State):
        e, q = ents(state), state["query"]
        scope = state.get("scope_policy_ids") or (e.policy_ids or None)
        if state["route"] == "comparative" and not e.policy_ids:
            pids = scope or [p["id"] for p in family.policies() if not e.ptypes or p["type"] in e.ptypes]
            merged: dict[str, Retrieved] = {}
            timing: dict = {}
            for pid in pids:
                res, timing = retriever.search(q, policy_ids=[pid], top_k=2)
                for r in res:
                    merged[r.chunk.id] = r
            allr = sorted(merged.values(), key=lambda r: -(r.rerank if r.rerank is not None else r.rrf))[:6]
            return {"retrieved": allr, "retrieval_timing": timing}
        res, timing = retriever.search(q, policy_ids=scope, ptypes=None if scope else e.ptypes)
        return {"retrieved": res, "retrieval_timing": timing}

    @timed("generate")
    def generate(state: State):
        lines, retrieved = state["graph_lines"], state["retrieved"]
        if state["route"] in ("structured", "comparative") and state["graph_answerable"] and not retrieved:
            return {"answer": "\n".join(lines), "llm_part": ""}
        parts, llm_part = [], ""
        if lines:
            parts.append("**From the family knowledge graph**\n" + "\n".join(lines))
        if retrieved:
            blocks = [(r.chunk.citation.strip("[]"), f"{r.chunk.policy_name} - {r.chunk.title}", r.chunk.text) for r in retrieved]
            llm_part = llm.answer(state["query"], blocks)
            parts.append(("**From the policy documents**\n" if lines else "") + llm_part)
        if not parts:
            return {"answer": "I could not find this in the family's policy documents or portfolio data.", "llm_part": ""}
        return {"answer": "\n\n".join(parts), "llm_part": llm_part}

    @timed("verify")
    def verify(state: State):
        """Citation check: every LLM claim needs a valid tag whose chunk lexically supports it."""
        retrieved = {r.chunk.id: r for r in state["retrieved"]}
        allowed = set(retrieved) | set(_cites_in(state["graph_lines"]))
        llm_part = state.get("llm_part", "")
        report = {"llm_lines": 0, "supported": 0, "unsupported": [], "uncited_dropped": [], "invalid_cites": []}
        kept = []
        for line in __import__("fia.llm", fromlist=["split_lines"]).split_lines(llm_part):
            tags = [f"{p}§{c}" for p, c in CITE.findall(line)]
            words = [t for t in tokenize(CITE.sub("", line)) if len(t) > 2]
            if not tags:
                if len(words) >= 6:
                    report["uncited_dropped"].append(line)
                    continue
                kept.append(line)
                continue
            report["llm_lines"] += 1
            bad = [t for t in tags if t not in allowed]
            if bad:
                report["invalid_cites"] += bad
                continue
            support = 0.0
            for t in tags:
                src = retrieved[t].chunk.indexed_text if t in retrieved else ""
                toks = set(tokenize(src))
                support = max(support, sum(w in toks for w in words) / (len(words) or 1))
            if support >= 0.5:
                report["supported"] += 1
                kept.append(line)
            else:
                report["unsupported"].append(line)
        answer = state["answer"]
        if llm_part and kept != split_lines_cached(llm_part):
            new_llm = "\n".join(kept) or "No claim in the generated text could be verified against the policy documents."
            answer = answer.replace(llm_part, new_llm)
            llm_part = new_llm
        # confidence
        if state["route"] in ("structured", "comparative") and state["graph_answerable"] and not state["retrieved"]:
            conf = 0.95
        elif state["retrieved"]:
            top = state["retrieved"][0]
            signal = 1.0 if (top.dense_rank and top.bm25_rank and top.dense_rank <= 5 and top.bm25_rank <= 5) \
                else 0.7 if (top.dense_rank or top.bm25_rank) else 0.4
            frac = report["supported"] / report["llm_lines"] if report["llm_lines"] else 0.0
            conf = round(0.5 * frac + 0.5 * signal, 3)
            if state["graph_answerable"]:
                conf = round(min(1.0, conf + 0.05), 3)
            # Evidence guard: dense search always returns *something*, so also require lexical evidence that the
            # retrieved text is actually about the question. No overlap at all => "no evidence" (route to a human).
            qtoks = {t for t in tokenize(state["query"]) if len(t) > 2}
            ctx = set().union(*(set(tokenize(r.chunk.indexed_text)) for r in state["retrieved"][:3]))
            coverage = len(qtoks & ctx) / (len(qtoks) or 1)
            report["evidence_coverage"] = round(coverage, 3)
            if coverage == 0:
                conf = min(conf, 0.2)
            elif coverage < 0.15:
                conf = min(conf, 0.6)
        else:
            conf = 0.3
        report["confidence"] = conf
        cites = _cites_in(answer.splitlines())
        return {"answer": answer, "llm_part": llm_part, "verification": report, "confidence": conf, "citations": cites}

    @timed("risk_gate")
    def risk_gate(state: State):
        reasons = []
        if state["route"] == "high_stakes":
            reasons.append("high-stakes question (named family member + medical/claim/death keyword, or claim decision)")
        if state["confidence"] < config.CONFIDENCE_THRESHOLD:
            reasons.append(f"low confidence ({state['confidence']:.2f} < {config.CONFIDENCE_THRESHOLD})")
        for amt in _amounts(state["query"]):
            if amt >= config.AMOUNT_THRESHOLD_INR:
                reasons.append(f"financial decision involving {inr(amt)}")
                break
        return {"review_reasons": reasons, "needs_review": bool(reasons),
                "status": "pending_review" if reasons else "auto_approved"}

    @timed("audit")
    def audit(state: State):
        rec = {
            "query_id": state["query_id"], "ts": state["ts"], "user": state["user"], "query": state["query"],
            "route": state["route"], "entities": state["entities"], "graph": state["graph_runs"],
            "retrieved": [r.audit() for r in state["retrieved"]], "retrieval_timing": state["retrieval_timing"],
            "backends": {"llm": llm.name, **retriever.backends, "graph": graph.backend},
            "answer": state["answer"], "citations": state["citations"], "confidence": state["confidence"],
            "verification": state["verification"], "needs_review": state["needs_review"],
            "review_reasons": state["review_reasons"], "status": state["status"],
            "timings": state["timings"], "trace": state["trace"] + ["audit"],
        }
        store.log_query(rec)
        return {}

    def after_classify(state: State) -> str:
        r = state["route"]
        if r in ("structured", "comparative", "high_stakes"):
            return "graph"
        return "graph" if ents(state).member_ids else "rag"

    def after_graph(state: State) -> str:
        r = state["route"]
        if r == "structured":
            return "generate"
        if r == "comparative" and state["graph_answerable"]:
            return "generate"
        return "rag"

    g = StateGraph(State)
    for name, fn in [("classify", classify), ("graph", graph_node), ("rag", rag_node), ("generate", generate),
                     ("verify", verify), ("risk_gate", risk_gate), ("audit", audit)]:
        g.add_node(name, fn)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", after_classify, {"graph": "graph", "rag": "rag"})
    g.add_conditional_edges("graph", after_graph, {"generate": "generate", "rag": "rag"})
    g.add_edge("rag", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", "risk_gate")
    g.add_edge("risk_gate", "audit")
    g.add_edge("audit", END)
    app = g.compile()

    def ask(user: str, query: str) -> State:
        t0 = time.perf_counter()
        out = app.invoke({"query_id": uuid.uuid4().hex[:12], "user": user, "query": query,
                          "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "timings": {}, "trace": []})
        out["timings"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    ask.compiled = app  # type: ignore[attr-defined]
    return ask


def split_lines_cached(text: str) -> list[str]:
    from .llm import split_lines
    return split_lines(text)


def _amounts(text: str) -> list[int]:
    t = text.lower().replace(",", "")
    out = [int(x) for x in re.findall(r"(?:₹|rs\.?|inr)\s*(\d+)", t)]
    out += [int(float(n) * 100000) for n in re.findall(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)", t)]
    out += [int(float(n) * 10000000) for n in re.findall(r"(\d+(?:\.\d+)?)\s*crore", t)]
    return out
