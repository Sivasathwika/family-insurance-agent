"""Knowledge graph of the family portfolio.

Design rule: the LLM never writes Cypher. Every graph question maps to a *named,
parameterised template* below, so each answer is deterministic and the exact query
(+ params) can be written to the audit log.

Two interchangeable backends share the same templates:
  * Neo4jGraph  - executes the Cypher against a real Neo4j server.
  * MemoryGraph - pure-Python twin used for tests/offline demos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import config, family

# --------------------------------------------------------------------------- Cypher
_MEMBER_COLS = """p.id AS policy_id, p.name AS policy, p.type AS type,
       p.sum_insured_display AS sum_insured, p.renewal_date AS renewal_date"""

CYPHER: dict[str, str] = {
    "policies_for_member": f"""
MATCH (m:FamilyMember {{id: $member_id}})-[:COVERED_BY]->(p:Policy)
RETURN m.name AS member, {_MEMBER_COLS}
ORDER BY p.id""",
    "coverage_for_member": f"""
MATCH (m:FamilyMember {{id: $member_id}})-[:COVERED_BY]->(p:Policy)-[:COVERS]->(c:Coverage {{name: $coverage}})
RETURN m.name AS member, {_MEMBER_COLS}
ORDER BY p.id""",
    "coverage_any": """
MATCH (p:Policy)-[:COVERS]->(c:Coverage {name: $coverage})
OPTIONAL MATCH (m:FamilyMember)-[:COVERED_BY]->(p)
RETURN p.id AS policy_id, p.name AS policy, p.type AS type,
       p.sum_insured_display AS sum_insured, collect(m.name) AS members
ORDER BY p.id""",
    "waiting_periods": """
MATCH (p:Policy)-[:HAS_WAITING]->(w:Waiting)
WHERE ($condition IS NULL OR w.condition = $condition)
  AND ($ptypes IS NULL OR p.type IN $ptypes)
  AND ($policy_ids IS NULL OR p.id IN $policy_ids)
RETURN p.id AS policy_id, p.name AS policy, w.condition AS condition,
       w.days AS days, w.label AS label, w.clause AS clause
ORDER BY w.days, p.id""",
    "total_premium": """
MATCH (p:Policy)
WHERE ($ptypes IS NULL OR p.type IN $ptypes)
  AND ($policy_ids IS NULL OR p.id IN $policy_ids)
RETURN sum(p.premium_annual) AS total, count(p) AS n,
       collect({policy_id: p.id, policy: p.name, premium: p.premium_annual}) AS breakdown""",
    "renewals_due": """
MATCH (p:Policy)
WHERE date(p.renewal_date) >= date($today) AND date(p.renewal_date) <= date($cutoff)
RETURN p.id AS policy_id, p.name AS policy, p.renewal_date AS renewal_date,
       p.premium_annual AS premium
ORDER BY date(p.renewal_date)""",
    "nominees": """
MATCH (n:FamilyMember)-[:NOMINEE_OF]->(p:Policy)
WHERE $policy_ids IS NULL OR p.id IN $policy_ids
RETURN p.id AS policy_id, p.name AS policy, n.name AS nominee, n.relation AS relation
ORDER BY p.id""",
    "list_policies": """
MATCH (p:Policy)
OPTIONAL MATCH (m:FamilyMember)-[:COVERED_BY]->(p)
RETURN p.id AS policy_id, p.name AS policy, p.type AS type,
       p.sum_insured_display AS sum_insured, collect(m.name) AS members
ORDER BY p.id""",
}

SEED_CYPHER = {
    "constraints": [
        "CREATE CONSTRAINT member_id IF NOT EXISTS FOR (m:FamilyMember) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
    ],
    "member": "MERGE (m:FamilyMember {id: $id}) SET m.name = $name, m.relation = $relation, m.age = $age",
    "policy": """MERGE (p:Policy {id: $id})
SET p.name = $name, p.type = $type, p.insurer = $insurer, p.policy_no = $policy_no,
    p.sum_insured = $sum_insured, p.sum_insured_display = $sum_insured_display,
    p.premium_annual = $premium_annual, p.renewal_date = $renewal_date""",
    "covered": "MATCH (m:FamilyMember {id: $m}), (p:Policy {id: $p}) MERGE (m)-[:COVERED_BY]->(p)",
    "holder": "MATCH (m:FamilyMember {id: $m}), (p:Policy {id: $p}) MERGE (p)-[:OWNED_BY]->(m)",
    "nominee": "MATCH (m:FamilyMember {id: $m}), (p:Policy {id: $p}) MERGE (m)-[:NOMINEE_OF]->(p)",
    "coverage": "MATCH (p:Policy {id: $p}) MERGE (c:Coverage {name: $name}) MERGE (p)-[:COVERS]->(c)",
    "waiting": """MATCH (p:Policy {id: $p})
MERGE (p)-[:HAS_WAITING]->(w:Waiting {key: $key})
SET w.condition = $condition, w.days = $days, w.label = $label, w.clause = $clause""",
}


@dataclass
class GraphResult:
    template: str
    cypher: str
    params: dict[str, Any]
    rows: list[dict]
    backend: str
    latency_ms: float = 0.0

    def audit(self) -> dict:
        return {"template": self.template, "cypher": self.cypher.strip(), "params": self.params,
                "n_rows": len(self.rows), "backend": self.backend, "latency_ms": round(self.latency_ms, 2)}


class BaseGraph:
    backend = "base"

    def run(self, template: str, **params) -> GraphResult:
        assert template in CYPHER, f"unknown template {template}"
        full = {"member_id": None, "coverage": None, "condition": None, "ptypes": None,
                "policy_ids": None, "today": None, "cutoff": None}
        full.update(params)
        t0 = time.perf_counter()
        rows = self._execute(template, full)
        return GraphResult(template, CYPHER[template], {k: v for k, v in full.items() if v is not None},
                           rows, self.backend, (time.perf_counter() - t0) * 1000)

    def _execute(self, template: str, p: dict) -> list[dict]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- Neo4j
class Neo4jGraph(BaseGraph):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def _execute(self, template: str, p: dict) -> list[dict]:
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(CYPHER[template], **p)]

    def ensure_seeded(self) -> bool:
        """Seed an empty database (e.g. a fresh AuraDB instance). Never wipes existing data."""
        with self.driver.session(database=self.database) as s:
            n = s.run("MATCH (p:Policy) RETURN count(p) AS n").single()["n"]
        if n == 0:
            self.seed(wipe=False)
        return n == 0

    def seed(self, wipe: bool = True) -> None:
        """Load data/family.json into Neo4j (idempotent)."""
        fam = family.load_family()
        with self.driver.session(database=self.database) as s:
            if wipe:
                s.run("MATCH (n) DETACH DELETE n")
            for c in SEED_CYPHER["constraints"]:
                s.run(c)
            for m in fam["members"]:
                s.run(SEED_CYPHER["member"], id=m["id"], name=m["name"], relation=m["relation"], age=m["age"])
            for p in fam["policies"]:
                s.run(SEED_CYPHER["policy"], **{k: p[k] for k in (
                    "id", "name", "type", "insurer", "policy_no", "sum_insured",
                    "sum_insured_display", "premium_annual", "renewal_date")})
                s.run(SEED_CYPHER["holder"], m=p["policyholder"], p=p["id"])
                for m in p["covered"]:
                    s.run(SEED_CYPHER["covered"], m=m, p=p["id"])
                for m in p["nominees"]:
                    s.run(SEED_CYPHER["nominee"], m=m, p=p["id"])
                for c in p["coverages"]:
                    s.run(SEED_CYPHER["coverage"], p=p["id"], name=c)
                for w in p["waiting"]:
                    s.run(SEED_CYPHER["waiting"], p=p["id"], key=f'{p["id"]}:{w["condition"]}', **w)


# --------------------------------------------------------------------------- memory
class MemoryGraph(BaseGraph):
    """Python twin of the Cypher templates (same inputs -> same row shapes)."""
    backend = "memory"

    def __init__(self):
        self.fam = family.load_family()
        self.members = {m["id"]: m for m in self.fam["members"]}
        self.policies = {p["id"]: p for p in self.fam["policies"]}

    def _scope(self, p: dict, pol: dict) -> bool:
        return ((p.get("ptypes") is None or pol["type"] in p["ptypes"]) and
                (p.get("policy_ids") is None or pol["id"] in p["policy_ids"]))

    def _base(self, pol: dict) -> dict:
        return {"policy_id": pol["id"], "policy": pol["name"], "type": pol["type"],
                "sum_insured": pol["sum_insured_display"]}

    def _execute(self, t: str, p: dict) -> list[dict]:
        pols = sorted(self.policies.values(), key=lambda x: x["id"])
        if t in ("policies_for_member", "coverage_for_member"):
            m = self.members[p["member_id"]]
            rows = []
            for pol in pols:
                if m["id"] in pol["covered"] and (t == "policies_for_member" or p["coverage"] in pol["coverages"]):
                    rows.append({"member": m["name"], **self._base(pol), "renewal_date": pol["renewal_date"]})
            return rows
        if t == "coverage_any":
            return [{**self._base(pol), "members": [self.members[i]["name"] for i in pol["covered"]]}
                    for pol in pols if p["coverage"] in pol["coverages"]]
        if t == "waiting_periods":
            rows = []
            for pol in pols:
                if not self._scope(p, pol):
                    continue
                for w in pol["waiting"]:
                    if p["condition"] is None or w["condition"] == p["condition"]:
                        rows.append({"policy_id": pol["id"], "policy": pol["name"], "condition": w["condition"],
                                     "days": w["days"], "label": w["label"], "clause": w["clause"]})
            return sorted(rows, key=lambda r: (r["days"], r["policy_id"]))
        if t == "total_premium":
            sel = [pol for pol in pols if self._scope(p, pol)]
            return [{"total": sum(x["premium_annual"] for x in sel), "n": len(sel),
                     "breakdown": [{"policy_id": x["id"], "policy": x["name"], "premium": x["premium_annual"]} for x in sel]}]
        if t == "renewals_due":
            lo, hi = date.fromisoformat(p["today"]), date.fromisoformat(p["cutoff"])
            sel = [x for x in pols if lo <= date.fromisoformat(x["renewal_date"]) <= hi]
            sel.sort(key=lambda x: x["renewal_date"])
            return [{"policy_id": x["id"], "policy": x["name"], "renewal_date": x["renewal_date"],
                     "premium": x["premium_annual"]} for x in sel]
        if t == "nominees":
            rows = []
            for pol in pols:
                if p.get("policy_ids") is not None and pol["id"] not in p["policy_ids"]:
                    continue
                for nid in pol["nominees"]:
                    n = self.members[nid]
                    rows.append({"policy_id": pol["id"], "policy": pol["name"], "nominee": n["name"], "relation": n["relation"]})
            return rows
        if t == "list_policies":
            return [{**self._base(pol), "members": [self.members[i]["name"] for i in pol["covered"]]} for pol in pols]
        raise KeyError(t)


def get_graph() -> BaseGraph:
    if config.GRAPH_BACKEND == "neo4j":
        g = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
        if config.SEED_ON_START:
            g.ensure_seeded()
        return g
    return MemoryGraph()
