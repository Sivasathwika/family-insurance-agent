from fastapi.testclient import TestClient


def test_api_flow(svc):
    from fia import api
    api.svc = svc
    with TestClient(api.app) as c:
        assert c.get("/health").json()["ok"]
        r = c.post("/ask", json={"user": "Sathwika", "question": "Should we file a claim for Dad's heart surgery?"}).json()
        assert r["status"] == "pending_review" and r["answer"] is None
        pend = [p for p in c.get("/reviews/pending").json() if p["query_id"] == r["query_id"]][0]
        d = c.post(f"/reviews/{pend['review_id']}/decision", json={"reviewer": "dr-x", "decision": "approve"})
        assert d.status_code == 200
        assert c.post(f"/reviews/{pend['review_id']}/decision", json={"reviewer": "dr-x", "decision": "approve"}).status_code == 409
        assert c.get(f"/queries/{r['query_id']}").json()["status"] == "approved"
        assert c.get("/queries/nope").status_code == 404
        assert c.get("/audit-verify").json()["ok"]
