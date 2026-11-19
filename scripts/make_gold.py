"""Builds data/eval/gold.jsonl (50 questions). Edit this file, re-run, and ADD YOUR OWN questions:
a gold set written only by the system's author flatters the system."""
import json
from pathlib import Path

G = []
def add(qid, q, route, chunks=None, policies=None, contains=None, review=False, tag="easy"):
    G.append({"id": qid, "question": q, "tag": tag, "gold_route": route, "gold_chunks": chunks or [],
              "gold_policies": policies or [], "answer_contains": contains or [], "gold_review": review})

# ---- structured (graph-only) ----
add("S01", "Which policies cover Mom?", "structured", policies=["P1", "P4", "P7"])
add("S02", "Which policies cover Dad?", "structured", policies=["P1", "P3", "P5", "P7"])
add("S03", "Which policies cover my brother Arjun?", "structured", policies=["P1"])
add("S04", "What is our total annual premium?", "structured", contains=["1,93,750"])
add("S05", "What is the total premium for our health policies?", "structured", contains=["74,200"])
add("S06", "Which policies are due for renewal in the next 30 days?", "structured", policies=["P4", "P5"])
add("S07", "Which policies are due for renewal in the next 60 days?", "structured", policies=["P1", "P4", "P5"])
add("S08", "Who is the nominee on Dad's policies?", "structured", policies=["P3"], contains=["Lakshmi"])
add("S09", "Who is the nominee on Mom's policies?", "structured", policies=["P4"], contains=["Ravi"])
add("S10", "Which policies cover surgery?", "structured", policies=["P1", "P2"])
add("S11", "Which policies include critical illness cover?", "structured", policies=["P7"])
add("S12", "Which policies cover maternity?", "structured", policies=["P1"])
add("S13", "Which policies cover Grandpa?", "structured", policies=["P2"])
add("S14", "Which policies pay a death benefit?", "structured", policies=["P3", "P4"])
add("S15", "List all our insurance policies", "structured", policies=["P1", "P2", "P3", "P4", "P5", "P6", "P7"])
# ---- comparative ----
add("K01", "Which policy has the shortest waiting period for pre-existing diseases?", "comparative", policies=["P2"], contains=["24 months"])
add("K02", "Which policy has the longest waiting period for pre-existing diseases?", "comparative", policies=["P1", "P7"], contains=["36 months"])
add("K03", "Which health policy has the shortest waiting period for specific diseases?", "comparative", policies=["P2"], contains=["12 months"])
add("K04", "Compare the room rent limits across our health policies", "comparative", chunks=["P1§2.2", "P2§2.2"])
add("K05", "Compare the pre and post hospitalisation cover across our health policies", "comparative", chunks=["P1§2.3", "P2§2.3"])
add("K06", "Compare the suicide exclusion period across our life policies", "comparative", policies=["P3", "P4"], contains=["12 months"])
add("K07", "Which policy has the shortest waiting period for critical illness?", "comparative", policies=["P7"], contains=["90 days"])
# ---- clause-level (hybrid RAG) ----
add("C01", "What is the room rent limit in the family floater?", "clause", chunks=["P1§2.2"])
add("C02", "What is the room rent cap in Grandpa's senior citizen policy?", "clause", chunks=["P2§2.2"])
add("C03", "Is there a co-payment in Grandpa's health policy?", "clause", chunks=["P2§3.1"])
add("C04", "How long do I have to submit documents for a reimbursement claim under the family floater?", "clause", chunks=["P1§3.2"])
add("C05", "What is the payout limit if childbirth needs a surgical delivery?", "clause", chunks=["P1§2.5"], tag="paraphrase")
add("C06", "What is the sum insured and policy period for SHI/FF/2025/004512?", "clause", chunks=["P1§S"], tag="exact_term")
add("C07", "Will tooth treatment be paid for under the family floater?", "clause", chunks=["P1§6.1"], tag="paraphrase")
add("C08", "How long is the waiting period for cataract under the family floater?", "clause", chunks=["P1§5.3"])
add("C09", "Does the diabetes waiting period apply under the senior citizen policy?", "clause", chunks=["P2§5.2"])
add("C10", "How much extra is paid if death happens because of a mishap?", "clause", chunks=["P3§2.2"], tag="paraphrase")
add("C11", "Does the term plan pay out for suicide?", "clause", chunks=["P3§5.1"])
add("C12", "What does Section 45 of the Insurance Act mean for our term plan?", "clause", chunks=["P3§5.3"], tag="exact_term")
add("C13", "How long does the insurer have to cancel the term plan for giving false information?", "clause", chunks=["P3§5.3"], tag="paraphrase")
add("C14", "If I stop paying premiums on the endowment plan, when do I still keep reduced cover?", "clause", chunks=["P4§3.2"], tag="paraphrase")
add("C15", "When does policy JLI/EP/2018/003340 mature?", "clause", chunks=["P4§S"], tag="exact_term")
add("C16", "What is the maturity benefit of Mom's endowment plan?", "clause", chunks=["P4§2.2"])
add("C17", "What is the compulsory deductible in the car policy?", "clause", chunks=["P5§3.1"])
add("C18", "Will the car insurer pay if the driver had been drinking?", "clause", chunks=["P5§5.1"], tag="paraphrase")
add("C19", "What compensation do I get if my suitcase goes missing abroad?", "clause", chunks=["P6§2.4"], tag="paraphrase")
add("C20", "Does the travel plan pay for injuries while skiing?", "clause", chunks=["P6§4.2"], tag="paraphrase")
# ---- high-stakes (graph + RAG + human review) ----
add("H01", "Is Mom's cancer covered?", "high_stakes", chunks=["P7§2.2", "P7§1.1"], review=True)
add("H02", "Should we file a claim for Dad's heart surgery?", "high_stakes", chunks=["P1§2.1", "P7§1.1", "P1§3.1"], review=True)
add("H03", "Can we claim for Grandpa's cataract surgery?", "high_stakes", chunks=["P2§3.3", "P2§5.3"], review=True)
add("H04", "What is the death benefit under Mom's endowment policy?", "high_stakes", chunks=["P4§2.1"], review=True)
add("H05", "Will Dad's term policy pay if he dies in an accident?", "high_stakes", chunks=["P3§2.2", "P3§2.1"], review=True)
add("H06", "Is Arjun's knee replacement covered in the floater?", "high_stakes", chunks=["P1§5.3"], review=True)
add("H07", "Should we claim for Sathwika's hospitalisation in Germany?", "high_stakes", chunks=["P6§2.1", "P6§3.2"], review=True)
add("H08", "Is Dad's car accident repair covered?", "high_stakes", chunks=["P5§2.1"], review=True)

assert len(G) == 50, len(G)
out = Path(__file__).resolve().parents[1] / "data" / "eval" / "gold.jsonl"
out.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in G) + "\n", encoding="utf-8")
print("wrote", len(G), "questions ->", out)
