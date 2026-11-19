"""Runs only when a Neo4j server is available:
    docker compose up -d neo4j && export NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=insurance123
    python -m fia.cli seed-neo4j && pytest tests/test_neo4j_parity.py
Asserts the real Cypher returns exactly what the in-memory twin returns, for every template."""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("NEO4J_URI"), reason="NEO4J_URI not set")


@pytest.fixture(scope="module")
def graphs():
    from fia import config
    from fia.graph_store import MemoryGraph, Neo4jGraph
    return MemoryGraph(), Neo4jGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)


CASES = [
    ("policies_for_member", {"member_id": "M2"}),
    ("policies_for_member", {"member_id": "M1"}),
    ("coverage_for_member", {"member_id": "M2", "coverage": "cancer"}),
    ("coverage_any", {"coverage": "surgery"}),
    ("waiting_periods", {"condition": "pre_existing"}),
    ("waiting_periods", {"condition": "specific_disease", "ptypes": ["health"]}),
    ("waiting_periods", {"policy_ids": ["P1", "P4"]}),
    ("total_premium", {}),
    ("total_premium", {"ptypes": ["health"]}),
    ("renewals_due", {"today": "2026-11-07", "cutoff": "2026-12-07"}),
    ("nominees", {}),
    ("nominees", {"policy_ids": ["P3"]}),
    ("list_policies", {}),
]


def _norm(rows):
    def fix(v):
        if isinstance(v, list):
            return sorted((fix(x) for x in v), key=str)
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v
    return sorted((fix(r) for r in rows), key=str)


@pytest.mark.parametrize("template,params", CASES)
def test_parity(graphs, template, params):
    mem, neo = graphs
    assert _norm(mem.run(template, **params).rows) == _norm(neo.run(template, **params).rows)
