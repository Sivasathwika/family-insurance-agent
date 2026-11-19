"""Static family/policy metadata + entity resolution (who / which policy / which cover)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def load_family() -> dict:
    return json.loads(config.FAMILY_JSON.read_text(encoding="utf-8"))


def members() -> list[dict]:
    return load_family()["members"]


def policies() -> list[dict]:
    return load_family()["policies"]


def policy_by_id(pid: str) -> dict | None:
    return next((p for p in policies() if p["id"] == pid), None)


def member_by_id(mid: str) -> dict | None:
    return next((m for m in members() if m["id"] == mid), None)


# ---- vocab ---------------------------------------------------------------

COVERAGE_KEYWORDS = {
    "surgery": ["surgery", "surgical", "operation"],
    "hospitalisation": ["hospitalisation", "hospitalization", "inpatient", "admitted"],
    "maternity": ["maternity", "delivery", "pregnan"],
    "critical_illness": ["critical illness", "critical care"],
    "cancer": ["cancer"],
    "heart_attack": ["heart attack"],
    "death_benefit": ["death benefit", "death cover", "life cover", "pay a death"],
    "accidental_death": ["accidental death"],
    "day_care": ["day care", "daycare"],
    "ambulance": ["ambulance"],
    "own_damage": ["own damage"],
    "third_party": ["third party", "third-party"],
    "personal_accident": ["personal accident"],
    "trip_cancellation": ["trip cancellation"],
    "baggage_loss": ["baggage"],
    "overseas_medical": ["overseas medical"],
    "maturity_benefit": ["maturity benefit"],
}

WAITING_CONDITIONS = {
    "pre_existing": ["pre-existing", "pre existing", "preexisting"],
    "initial": ["initial waiting", "first 30 days"],
    "specific_disease": ["specific disease", "specific illness"],
    "maternity": ["maternity"],
    "critical_illness": ["critical illness"],
    "suicide": ["suicide"],
}

# Concrete disease names live in clause text, not in the graph -> clause-level RAG.
DISEASE_TERMS = [
    "cataract", "hernia", "diabetes", "hypertension", "knee replacement",
    "joint replacement", "piles", "kidney stone", "gall bladder", "hysterectomy",
]

TYPE_WORDS = {"health": ["health"], "life": ["term", "endowment"]}


@dataclass
class Entities:
    member_ids: list[str] = field(default_factory=list)
    policy_ids: list[str] = field(default_factory=list)  # explicitly named policies
    ptypes: list[str] | None = None
    coverage: str | None = None
    condition: str | None = None
    disease_terms: list[str] = field(default_factory=list)


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text) is not None


def extract_entities(query: str) -> Entities:
    q = query.lower().replace("’", "'")
    q = re.sub(r"'s\b", "", q)
    e = Entities()
    for m in members():
        if any(_has(q, a) for a in m["aliases"]):
            e.member_ids.append(m["id"])
    for p in policies():
        if any(_has(q, a) for a in p["aliases"]):
            e.policy_ids.append(p["id"])
    # "health policies" style type scoping, only when no specific policy was named
    if not e.policy_ids:
        for w, t in TYPE_WORDS.items():
            if _has(q, w):
                e.ptypes = t
    for cov, kws in COVERAGE_KEYWORDS.items():
        if any(_has(q, k) for k in kws):
            e.coverage = cov
            break
    for cond, kws in WAITING_CONDITIONS.items():
        if any(k in q for k in kws):
            e.condition = cond
            break
    e.disease_terms = [d for d in DISEASE_TERMS if d in q]
    return e


def inr(n: int | float) -> str:
    """Indian digit grouping: 193750 -> ₹1,93,750."""
    n = int(round(n))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + "₹" + s
