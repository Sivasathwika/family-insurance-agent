import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(svc, monkeypatch):
    from fia import api, config
    api.svc = svc
    monkeypatch.setattr(config, "ACCESS_KEY", "acc-123")
    monkeypatch.setattr(config, "REVIEWER_KEY", "rev-456")
    with TestClient(api.app) as c:
        yield c


A, R = {"X-Access-Key": "acc-123"}, {"X-Access-Key": "acc-123", "X-Reviewer-Key": "rev-456"}


def test_health_is_open_and_reports_auth(client):
    j = client.get("/health").json()
    assert j["ok"] and j["auth"] == {"access_key": True, "reviewer_key": True}


def test_ui_served_and_js_is_valid(client, tmp_path):
    r = client.get("/")
    assert r.status_code == 200 and "Family Insurance Agent" in r.text
    js = re.search(r"<script>(.*)</script>", r.text, re.S).group(1)
    f = tmp_path / "ui.js"
    f.write_text(js, encoding="utf-8")  # Windows default (cp1252) cannot encode the emoji in the page
    node = shutil.which("node")  # optional: only checks the JS syntax when Node.js is installed
    if node:
        assert subprocess.run([node, "--check", str(f)], capture_output=True).returncode == 0


def test_ask_requires_access_key(client):
    body = {"user": "t", "question": "Which policies cover Dad?"}
    assert client.post("/ask", json=body).status_code == 401
    assert client.post("/ask", json=body, headers={"X-Access-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=body, headers=A).status_code == 200


def test_review_and_audit_need_reviewer_key(client):
    for path in ("/reviews/pending", "/audit", "/audit-verify"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers=A).status_code == 401          # access key alone is not enough
        assert client.get(path, headers=R).status_code == 200
    assert client.post("/reviews/1/decision", json={"reviewer": "x", "decision": "approve"}, headers=A).status_code == 401


def test_pending_answer_not_leaked_before_approval(client):
    r = client.post("/ask", json={"user": "t", "question": "Is Mom's cancer covered?"}, headers=A).json()
    assert r["answer"] is None
    assert client.get(f"/queries/{r['query_id']}", headers=A).json()["answer"] is None
    rid = [p for p in client.get("/reviews/pending", headers=R).json() if p["query_id"] == r["query_id"]][0]["review_id"]
    assert client.post(f"/reviews/{rid}/decision", json={"reviewer": "dr", "decision": "approve"}, headers=R).status_code == 200
    assert "P7" in client.get(f"/queries/{r['query_id']}", headers=A).json()["answer"]


def test_input_validation(client):
    assert client.post("/ask", json={"user": "t", "question": "x"}, headers=A).status_code == 422
    assert client.post("/ask", json={"user": "t", "question": "q" * 600}, headers=A).status_code == 422


def test_ensure_index_rebuilds_when_missing(tmp_path, monkeypatch):
    from fia import config, ingest
    from fia.embedders import HashEmbedder
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.json")
    first = ingest.ensure_index(HashEmbedder())
    second = ingest.ensure_index(HashEmbedder())
    assert first["rebuilt"] is True and second["rebuilt"] is False


def test_dotenv_loader(tmp_path, monkeypatch):
    from fia import config
    monkeypatch.delenv("FIA_TEST_X", raising=False)
    monkeypatch.delenv("FIA_TEST_Y", raising=False)
    monkeypatch.setenv("FIA_TEST_Y", "real")
    f = tmp_path / ".env"
    f.write_text("# c\nFIA_TEST_X=hello\nFIA_TEST_Y=fromfile\nEMPTY=\n")
    config._load_dotenv(f)
    import os
    assert os.environ["FIA_TEST_X"] == "hello" and os.environ["FIA_TEST_Y"] == "real"