"""Evaluation harness. Each metric isolates one failure mode of the pipeline.

 retrieval   hit@5 / MRR for dense, bm25, hybrid, hybrid+rerank  (raw: NO graph scoping -> conservative)
 pipeline    hit@5 of the chunks the agent actually retrieved (with graph/entity scoping)
 routing     route accuracy; recall of high-stakes questions sent to review
 graph       policy recall/precision + exact-answer checks for graph questions
 citations   % answers with >=1 citation; % citing a gold source
 faithfulness (lexical proxy) share of generated claims supported by their cited chunk
"""
from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault("FIA_TODAY", "2026-11-07")

from . import config  # noqa: E402
from .governance import AuditStore  # noqa: E402
from .retrieval import MODES  # noqa: E402


def load_gold(path=config.GOLD_PATH) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def raw_retrieval(svc, gold: list[dict]) -> dict:
    qs = [g for g in gold if g["gold_chunks"]]
    out = {}
    for mode in MODES:
        hits, rr, by_tag = 0, [], {}
        for g in qs:
            res, _ = svc.retriever.search(g["question"], mode=mode, top_k=5)
            ids = [r.chunk.id for r in res]
            rank = next((i for i, c in enumerate(ids, 1) if c in g["gold_chunks"]), None)
            hits += rank is not None
            rr.append(1 / rank if rank else 0.0)
            by_tag.setdefault(g.get("tag", "easy"), []).append(rank is not None)
        out[mode] = {"hit@5": hits / len(qs), "mrr": sum(rr) / len(qs), "n": len(qs),
                     "by_tag": {t: (sum(v) / len(v), len(v)) for t, v in by_tag.items()}}
    return out


def run_eval(svc_factory=None) -> dict:
    from .service import FamilyInsuranceService
    tmp = tempfile.mkdtemp()
    svc = (svc_factory or FamilyInsuranceService)(store=AuditStore(os.path.join(tmp, "eval.sqlite"))) \
        if svc_factory is None else svc_factory()
    gold = load_gold()
    rows, lat = [], []
    for g in gold:
        r = svc.ask("eval", g["question"])
        st = r["_state"]
        cited = set(st["citations"])
        cited_policies = {c.split("§")[0] for c in cited}
        if g["gold_route"] == "comparative" and g["gold_policies"] and st["graph_lines"]:
            # score comparative answers on the headline line (winner); the full table below it is context
            cited_policies = {p for p, _ in __import__("fia.agents", fromlist=["CITE"]).CITE.findall(st["graph_lines"][0])}
        rows.append({"g": g, "route": st["route"], "needs_review": st["needs_review"], "answer": st["answer"],
                     "cited": cited, "cited_policies": cited_policies,
                     "retrieved": [x.chunk.id for x in st["retrieved"]], "ver": st["verification"],
                     "conf": st["confidence"], "ms": st["timings"]["total"], "reasons": st["review_reasons"]})
        lat.append(st["timings"]["total"])

    m: dict = {"n_questions": len(gold)}
    m["backends"] = {**svc.retriever.backends, "llm": svc.llm.name, "graph": svc.graph.backend}
    m["raw_retrieval"] = raw_retrieval(svc, gold)

    chunk_rows = [x for x in rows if x["g"]["gold_chunks"]]
    ran_rag = [x for x in chunk_rows if x["retrieved"]]
    m["pipeline_hit@5"] = {"hit": sum(any(c in x["g"]["gold_chunks"] for c in x["retrieved"]) for x in ran_rag) / len(ran_rag),
                           "n": len(ran_rag)}
    m["route_accuracy"] = sum(x["route"] == x["g"]["gold_route"] for x in rows) / len(rows)
    m["route_confusions"] = [(x["g"]["id"], x["g"]["gold_route"], x["route"]) for x in rows if x["route"] != x["g"]["gold_route"]]

    hs = [x for x in rows if x["g"]["gold_review"]]
    non = [x for x in rows if not x["g"]["gold_review"]]
    m["high_stakes_review_recall"] = sum(x["needs_review"] for x in hs) / len(hs)
    m["non_high_stakes_review_rate"] = sum(x["needs_review"] for x in non) / len(non)

    gp = [x for x in rows if x["g"]["gold_policies"]]
    rec = [len(set(x["g"]["gold_policies"]) & x["cited_policies"]) / len(x["g"]["gold_policies"]) for x in gp]
    prec = [len(set(x["g"]["gold_policies"]) & x["cited_policies"]) / (len(x["cited_policies"]) or 1) for x in gp]
    m["graph_policy_recall"] = statistics.mean(rec)
    m["graph_policy_precision"] = statistics.mean(prec)
    ac = [x for x in rows if x["g"]["answer_contains"]]
    m["answer_contains_pass"] = sum(all(s.lower() in x["answer"].lower() for s in x["g"]["answer_contains"]) for x in ac) / len(ac)

    m["answers_with_citation"] = sum(bool(x["cited"]) for x in rows) / len(rows)
    cc = [x for x in chunk_rows]
    m["citation_accuracy_chunks"] = sum(bool(x["cited"] & set(x["g"]["gold_chunks"])) for x in cc) / len(cc)
    llm_lines = sum(x["ver"]["llm_lines"] for x in rows)
    m["faithfulness_lexical"] = (sum(x["ver"]["supported"] for x in rows) / llm_lines) if llm_lines else None
    m["latency_ms"] = {"p50": round(statistics.median(lat), 1), "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)}
    m["failures"] = [{"id": x["g"]["id"], "q": x["g"]["question"], "route": x["route"], "retrieved": x["retrieved"],
                      "gold_chunks": x["g"]["gold_chunks"], "cited": sorted(x["cited"])}
                     for x in chunk_rows if x["retrieved"] and not any(c in x["g"]["gold_chunks"] for c in x["retrieved"])]
    return m


def to_markdown(m: dict) -> str:
    rr = m["raw_retrieval"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    b = m["backends"]
    L = ["# Evaluation report", "",
         f"Questions: **{m['n_questions']}**  |  embedder: `{b['embedder']}`  |  reranker: `{b['reranker']}`  |  LLM: `{b['llm']}`  |  graph: `{b['graph']}`", ""]
    if "fallback" in b["embedder"] + b["reranker"]:
        L += ["> ⚠️ Offline fallback models were used. Do NOT quote these numbers as your result - "
              "re-run with sentence-transformers installed.", ""]
    tags = sorted(rr["hybrid"]["by_tag"])
    L += ["## Retrieval (raw, no graph scoping)", "",
          "| Mode | hit@5 (all) | MRR | " + " | ".join(f"{t} (n={rr['hybrid']['by_tag'][t][1]})" for t in tags) + " |",
          "|---|---|---|" + "---|" * len(tags)]
    for mode in MODES:
        L.append(f"| {mode} | {pct(rr[mode]['hit@5'])} | {rr[mode]['mrr']:.3f} | "
                 + " | ".join(pct(rr[mode]['by_tag'][t][0]) for t in tags) + " |")
    L += ["", f"Questions with gold chunks: {rr['hybrid']['n']}", "",
          "## Pipeline", "",
          f"- Pipeline hit@5 (with graph/entity scoping): **{pct(m['pipeline_hit@5']['hit'])}** (n={m['pipeline_hit@5']['n']})",
          f"- Route accuracy: **{pct(m['route_accuracy'])}**  (confusions: {m['route_confusions'] or 'none'})",
          f"- High-stakes questions sent to human review: **{pct(m['high_stakes_review_recall'])}**",
          f"- Non-high-stakes questions sent to review (over-flagging): {pct(m['non_high_stakes_review_rate'])}",
          f"- Graph policy recall / precision: {pct(m['graph_policy_recall'])} / {pct(m['graph_policy_precision'])}",
          f"- Exact-answer checks (premium totals, nominees, waiting periods): {pct(m['answer_contains_pass'])}",
          f"- Answers with >=1 citation: **{pct(m['answers_with_citation'])}**",
          f"- Citation accuracy (cites a gold clause): {pct(m['citation_accuracy_chunks'])}",
          f"- Faithfulness (lexical support proxy, not RAGAS): {pct(m['faithfulness_lexical'])}",
          f"- Latency p50 / p95: {m['latency_ms']['p50']} ms / {m['latency_ms']['p95']} ms", ""]
    if m["failures"]:
        L += ["## Retrieval failures (pipeline)", ""]
        for f in m["failures"]:
            L.append(f"- **{f['id']}** {f['q']}  \n  gold: {f['gold_chunks']} | got: {f['retrieved']}")
    return "\n".join(L) + "\n"
