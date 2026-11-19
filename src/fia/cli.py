"""CLI:  python -m fia.cli {ingest|seed-neo4j|ask|eval|serve}"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    # Windows consoles default to cp1252/cp437, which cannot print the rupee sign or emoji used in answers/reports.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="fia")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="chunk policies and build Chroma + BM25 indexes")
    sub.add_parser("warmup", help="download models + build the index (run at Docker build time)")
    sub.add_parser("seed-neo4j", help="load data/family.json into Neo4j")
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--user", default="cli")
    e = sub.add_parser("eval"); e.add_argument("--gate", type=float, default=None, help="exit 1 if raw hybrid_rerank hit@5 < this")
    e.add_argument("--out", default="eval_report.md")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "ingest":
        from .ingest import ingest
        print(json.dumps(ingest(), indent=2))
    elif args.cmd == "warmup":
        from .embedders import get_embedder, get_reranker
        from .ingest import ingest
        emb = get_embedder()
        rer = get_reranker([["warm"]])
        rer.score("warm up", ["warm up document"])
        print(json.dumps({**ingest(emb), "reranker": rer.name}, indent=2))
    elif args.cmd == "seed-neo4j":
        from . import config
        from .graph_store import Neo4jGraph
        g = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
        g.seed()
        print("Neo4j seeded.")
    elif args.cmd == "ask":
        from .service import FamilyInsuranceService
        r = FamilyInsuranceService().ask(args.user, args.question)
        print(f"[route={r['route']} status={r['status']} confidence={r['confidence']}]")
        print(r["answer"] or r["message"])
        for c in r["citations"]:
            print(f"  [{c['id']}] {c['policy']} - {c['title']}")
    elif args.cmd == "eval":
        from .eval import run_eval, to_markdown
        m = run_eval()
        md = to_markdown(m)
        open(args.out, "w", encoding="utf-8").write(md)
        open(args.out.rsplit(".", 1)[0] + ".json", "w", encoding="utf-8").write(json.dumps(m, indent=2, default=list))
        print(md)
        if args.gate is not None and m["raw_retrieval"]["hybrid_rerank"]["hit@5"] < args.gate:
            sys.exit(1)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("fia.api:app", host="0.0.0.0", port=args.port, workers=1)


if __name__ == "__main__":
    main()