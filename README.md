# Family Insurance Agent

A governed, auditable AI assistant for a family's insurance portfolio. It combines a **Neo4j knowledge graph**
(who is covered by what), **hybrid RAG** (clause-level text), **LangGraph multi-agent routing**, and
**human-in-the-loop review** for high-stakes questions. Every answer carries a source clause and every query is logged.

> Stack: LangGraph · Neo4j · Chroma · BM25 · cross-encoder reranker · FastAPI · Docker / Render (LLM: Claude or any free OpenAI-compatible API)

## Architecture

```
                         ┌────────────┐
 question ─► classify ───┤ route      ├─ structured  ─► graph ───────────────────────┐
             + entities  │            ├─ comparative ─► graph ─► (rag if not in graph)┤
                         │            ├─ clause ─────► [scope graph] ─► rag ──────────┼─► generate ─► verify ─► risk_gate ─► audit
                         └────────────┘└ high_stakes ─► graph ─► rag ─────────────────┘      (cite check)  (HITL?)    (hash chain)

 rag = Chroma dense top-20 ─┐
                            ├─ Reciprocal Rank Fusion (k=60) ─► cross-encoder rerank ─► top-5
       BM25 top-20 ─────────┘
```

| Question type | Example | Path |
|---|---|---|
| Structured | "What's our total premium?" | Named Cypher template → exact answer, no LLM arithmetic |
| Clause-level | "What's the room rent cap?" | Hybrid RAG, scoped to the policies the graph says apply |
| Comparative | "Which policy has the shortest PED wait?" | Graph (structured dimension) or per-policy RAG |
| High-stakes | "Should we claim for Dad's surgery?" | Graph + RAG → **held for human approval** |

Design rules: **the LLM never writes Cypher** (fixed, parameterised templates → deterministic and auditable);
**no citation, no claim** (uncited or unsupported sentences are dropped); **every query is logged** with the exact
Cypher, retrieved chunks with dense/BM25/RRF/rerank scores, model names, confidence, review decision and per-node latency.
The audit table is a SHA-256 hash chain (`GET /audit-verify` detects edited or deleted rows).

Review is triggered by (a) a named family member + medical/death/claim keyword, or an explicit claim decision,
(b) confidence < 0.7, (c) amounts ≥ ₹1,00,000 in the question. Pending answers are **not shown** to the asker until a
reviewer approves / edits / rejects.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements-dev.txt                        # light profile (ONNX). For PyTorch models: requirements-torch.txt
cp .env.example .env                                       # auto-loaded; set the keys, pick a free LLM (or none)
export PYTHONPATH=src                                      # Windows PowerShell: $env:PYTHONPATH="src"

python -m fia.cli serve                                    # http://localhost:8000  (web UI at /, API docs at /docs)
python -m fia.cli ask "Which policies cover Mom?"          # or use the CLI
```
First start builds the retrieval index automatically and downloads two small ONNX models (~100 MB). Without
`NEO4J_URI` the graph runs in memory; to use real Neo4j: `docker compose up -d neo4j`, set `NEO4J_URI=bolt://localhost:7687`
and `NEO4J_PASSWORD=insurance123` - an empty database is seeded on startup (`python -m fia.cli seed-neo4j` re-seeds manually).

## Deploy on Render

One Docker web service serves the UI and API; models and the index are baked into the image at build time.

1. Push this folder to a GitHub repo, then Render > **New > Blueprint** > pick the repo (uses `render.yaml`).
2. Render generates `FIA_ACCESS_KEY` and `FIA_REVIEWER_KEY` (Dashboard > service > **Environment**). Open
   `https://<service>.onrender.com`, paste the access key (and reviewer key for the Reviewer/Audit tabs) in the header boxes.
3. In **Environment**, add an LLM (optional; without one you get the extractive fallback):

   | Provider | `FIA_LLM_BASE_URL` | `FIA_LLM_MODEL` |
   |---|---|---|
   | Groq (free tier) | `https://api.groq.com/openai/v1` | e.g. `llama-3.3-70b-versatile` |
   | Gemini (free tier) | `https://generativelanguage.googleapis.com/v1beta/openai` | e.g. `gemini-2.0-flash` |
   | Claude (paid) | set `ANTHROPIC_API_KEY` instead | `claude-sonnet-5-5` |

   plus `FIA_LLM_API_KEY`. Model names and free limits change - check the provider's docs. Local Ollama does **not**
   work on Render (no localhost).
4. Optional real graph: create a free **Neo4j AuraDB** instance, then set `NEO4J_URI` (`neo4j+s://<id>.databases.neo4j.io`),
   `NEO4J_USER`, `NEO4J_PASSWORD`. The empty database is seeded on first start. Skip this and the in-memory graph is used.
5. Verify: `GET /health` shows the embedder/reranker/LLM/graph actually in use. If you see `hash` / `lexical` /
   `extractive` there, you are on fallbacks; the Docker image sets `FIA_EMBEDDER=fastembed`, which **fails loudly**
   rather than silently downgrading.

**Things to know**
* **Disk:** the audit log and review queue live in SQLite on the `/data` disk. Disks need a paid instance type. The
  free-tier file `deploy/render.free.yaml` has no disk, so those records reset on every restart/redeploy and the
  service sleeps when idle: demo only, never real family data.
* **Memory:** the app itself used ~150 MB RSS in local tests with stand-in models, plus the two ONNX models, Chroma and
  LangGraph. I could not measure the real total on Render. If the instance is OOM-killed, move to a larger plan.
* **Single worker:** SQLite + embedded Chroma are single-process stores, so the service runs one uvicorn worker. To scale
  out, move the audit log to Postgres and Chroma to a server.
* **Security:** the keys are shared secrets over HTTPS, enough for a family tool, not for multi-tenant use. There is
  no rate limiting; keep the access key private if you attach a paid LLM key. Don't upload real policy numbers or
  health details to a public repo; keep real policies in a private repo.
* **Cold starts:** index and models are in the image, so start-up is fast; the free tier's idle sleep still adds a
  wake-up delay.

## Evaluation

```bash
python scripts/make_gold.py                              # regenerates data/eval/gold.jsonl (50 questions)
PYTHONPATH=src python -m fia.cli eval --gate 0.85        # writes eval_report.md/json, exits 1 if hit@5 < 85%
pytest -q tests                                          # 30 offline tests
NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=insurance123 pytest tests/test_neo4j_parity.py   # Cypher == in-memory twin
```

Reported: hit@5 / MRR for **dense, BM25, hybrid, hybrid+rerank** (ablation, split into easy / paraphrase / exact-term
questions), pipeline hit@5, route accuracy, high-stakes→review recall, graph policy recall/precision, exact-answer
checks, citation coverage and accuracy, a lexical faithfulness proxy, latency p50/p95.

### Be honest about the numbers
* The bundled 7 policies are **synthetic** (fictional insurers). The gold set was written by the system's author. Both
  make the test easier than reality. Before quoting any number: replace/add **your own family's real policies
  (redact personal data)**, add questions you did not write yourself, and run with real models.
* Runs with `hash-1024` / `lexical` / `extractive-fallback` print a warning banner in the report. Those exist only so
  tests run offline.
* The faithfulness number is a lexical-overlap proxy. For a RAGAS score, run RAGAS over the audit log records.

## Using your real policies
1. Convert each policy PDF to clause-structured Markdown in `data/policies/` (front matter `policy_id/name/type`,
   one `## <clause> <title>` heading per clause; `S` = schedule). Keep the clause numbers from the PDF so citations
   point to real sections.
2. Update `data/family.json` (members, coverage, nominees, waiting periods, premiums, renewal dates).
3. `python -m fia.cli seed-neo4j && python -m fia.cli ingest`, rewrite `scripts/make_gold.py` with your questions.

## Layout
```
src/fia/  static/index.html (web UI) · config · family (entities) · graph_store (Cypher templates, Neo4j + memory twin) · ingest · embedders
          retrieval (dense+BM25+RRF+rerank) · llm · agents (LangGraph) · governance (audit+review) · service · api · eval · cli
data/     family.json · policies/*.md · eval/gold.jsonl      tests/      scripts/make_gold.py
Dockerfile · render.yaml · deploy/render.free.yaml · requirements*.txt
```

## Known limitations / next steps
* Entity resolution and routing are rule-based (fast, auditable, but brittle on unusual phrasing); an LLM classifier
  with a rule-based fallback is the natural upgrade.
* The citation verifier is lexical; add an LLM or NLI entailment check for stricter faithfulness.
* Real PDFs need a layout-aware parser (tables, multi-column); only Markdown is ingested here.
* Untested here (sandbox had no Docker, Neo4j server or model downloads): the Docker build, fastembed models, Neo4j Cypher
  (run `tests/test_neo4j_parity.py`), and the real Render deploy. The Python code paths were tested with stand-in models.
* Not legal or financial advice; high-stakes answers are decision support for a human reviewer.
#   f a m i l y - i n s u r a n c e - a g e n t  
 