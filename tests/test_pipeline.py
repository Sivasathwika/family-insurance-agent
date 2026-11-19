import pytest

from fia import family
from fia.agents import classify_query, plan_graph, _amounts
from fia.graph_store import MemoryGraph
from fia.ingest import chunk_policy
from fia.retrieval import rrf_fuse


# ---------- units
def test_indian_grouping():
    assert family.inr(193750) == "₹1,93,750"
    assert family.inr(10000000) == "₹1,00,00,000"
    assert family.inr(950) == "₹950"


def test_rrf_formula():
    s = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["d"] == pytest.approx(1 / 63)
    assert s["a"] > s["c"]


def test_chunker_keeps_clause_ids():
    ch = chunk_policy("---\npolicy_id: P9\nname: X\ntype: health\n---\n## 2.2 Room Rent\nCap is 1%.\n\n## S. Schedule\nHello.\n")
    assert [c.id for c in ch] == ["P9§2.2", "P9§S"]


def test_amount_parsing():
    assert _amounts("claim of ₹5,00,000") == [500000]
    assert _amounts("a 2 lakh surgery") == [200000]


def test_entity_resolution():
    e = family.extract_entities("Is Mom's cancer covered?")
    assert e.member_ids == ["M2"] and e.coverage == "cancer"
    assert family.extract_entities("What is the family floater room rent?").policy_ids == ["P1"]


def test_graph_templates_memory():
    g = MemoryGraph()
    assert [r["policy_id"] for r in g.run("policies_for_member", member_id="M2").rows] == ["P1", "P4", "P7"]
    assert g.run("total_premium").rows[0]["total"] == 193750
    r = g.run("waiting_periods", condition="pre_existing", ptypes=["health"])
    assert [x["policy_id"] for x in r.rows] == ["P2", "P1"]
    assert "MATCH" in r.cypher and r.params["condition"] == "pre_existing"


# ---------- routing
@pytest.mark.parametrize("q,route", [
    ("Which policies cover Dad?", "structured"),
    ("What is our total annual premium?", "structured"),
    ("What is the room rent limit in the family floater?", "clause"),
    ("Which policy has the shortest waiting period for pre-existing diseases?", "comparative"),
    ("Should we file a claim for Dad's heart surgery?", "high_stakes"),
    ("Is Is Mom's cancer covered?".replace("Is Is", "Is"), "high_stakes"),
    ("Is the senior citizen policy renewable for life?", "clause"),  # 'renew' must not trigger renewal template
])
def test_routes(q, route):
    assert classify_query(q, family.extract_entities(q)) == route


# ---------- end-to-end
def test_structured_answer_is_exact_and_cited(svc):
    r = svc.ask("t", "What is our total annual premium?")
    assert "₹1,93,750" in r["answer"] and r["citations"] and not r["needs_review"]
    q = svc.store.get_query(r["query_id"])
    assert q["graph"][0]["template"] == "total_premium" and "sum(p.premium_annual)" in q["graph"][0]["cypher"]


def test_graph_scopes_rag_to_member_policies(svc):
    r = svc.ask("t", "What is the maturity benefit of Mom's endowment plan?")
    assert any(c["id"] == "P4§2.2" for c in r["citations"])
    st = r["_state"]
    assert {x.chunk.policy_id for x in st["retrieved"]} <= {"P1", "P4", "P7"}  # Mom's policies only


def test_high_stakes_is_held_for_review_and_answer_hidden(svc):
    r = svc.ask("Sathwika", "Is Mom's cancer covered?")
    assert r["status"] == "pending_review" and r["answer"] is None and r["needs_review"]
    assert svc.get_answer(r["query_id"])["answer"] is None
    pend = [p for p in svc.pending_reviews() if p["query_id"] == r["query_id"]][0]
    assert "P7§2.2" in pend["draft_answer"]
    svc.decide(pend["review_id"], "reviewer-1", "approve")
    final = svc.get_answer(r["query_id"])
    assert final["status"] == "approved" and "P7§2.2" in final["answer"]
    with pytest.raises(ValueError):  # cannot decide twice
        svc.decide(pend["review_id"], "reviewer-2", "reject")


def test_reviewer_edit_and_reject(svc):
    r1 = svc.ask("t", "Should we file a claim for Dad's heart surgery?")
    r2 = svc.ask("t", "Is Arjun's knee replacement covered in the floater?")
    ids = {p["query_id"]: p["review_id"] for p in svc.pending_reviews()}
    svc.decide(ids[r1["query_id"]], "rev", "edit", edited_answer="Edited by reviewer [P1§3.1]")
    svc.decide(ids[r2["query_id"]], "rev", "reject", note="needs the doctor's note")
    assert svc.get_answer(r1["query_id"])["answer"] == "Edited by reviewer [P1§3.1]"
    assert svc.get_answer(r2["query_id"])["status"] == "rejected"
    assert svc.get_answer(r2["query_id"])["answer"] is None


def test_every_answer_has_citation(svc):
    for q in ["Which policies cover Mom?", "How much does the car policy deduct per claim?",
              "Compare the room rent limits across our health policies", "Which policies are due for renewal in the next 30 days?"]:
        assert svc.ask("t", q)["citations"], q


def test_unknown_question_does_not_hallucinate(svc):
    r = svc.ask("t", "zzzz qqqq xxxx")
    assert r["confidence"] <= 0.2 and r["needs_review"]   # no lexical evidence -> human review, never auto-approved


def test_audit_chain_detects_tampering(svc):
    assert svc.store.verify_chain()["ok"]
    svc.store.db.execute("UPDATE audit_log SET record = record || ' ' WHERE seq = 1")
    assert not svc.store.verify_chain()["ok"]


def test_audit_record_has_traceability_fields(svc):
    rec = svc.store.get_query(svc.ask("t", "Which policies cover Dad?")["query_id"])
    for k in ("route", "graph", "retrieved", "backends", "answer", "citations", "confidence", "timings", "trace", "review_reasons"):
        assert k in rec
