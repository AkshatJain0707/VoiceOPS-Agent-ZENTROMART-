"""No-LLM retrieval check: recall@k of gold KB sections on the practice questions."""
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from zentro.kb import KnowledgeBase, PINNED_RAG, PINNED_FULL
ds = Path(sys.argv[1]); col = sys.argv[2] if len(sys.argv) > 2 else "question"; PIN = (sys.argv[3] == "pin") if len(sys.argv) > 3 else False
kb = KnowledgeBase(ds / "kb", "none")
rows = list(csv.DictReader(open(ds / "practice/questions.csv", encoding="utf-8")))
print(len(kb.chunks), "chunks;", sum(c.superseded for c in kb.chunks), "superseded")
for k in (4, 6, 8):
    tot = hit = n = 0; miss = []
    for r in rows:
        gold = [g.strip() for g in re.split(r"\s*;\s*", r["gold_evidence"] or "") if g.strip() and ":" not in g]
        if not gold: continue
        got = {h.chunk.section_id for h in kb.retrieve(r[col], k=k, pinned=PINNED_RAG if PIN else [])}
        n += 1; tot += len(gold); hit += sum(g in got for g in gold)
        if k == 6 and any(g not in got for g in gold): miss.append((r["id"], r["language"], r[col][:60], gold, sorted(got)))
    print(f"recall@{k}: {hit/tot:.3f} over {n} questions ({tot} gold sections)")
    if k == 6:
        for m in miss[:40]:
            safe_text = m[2].encode("ascii", "replace").decode()
            print("  MISS", m[0], m[1], safe_text, "| gold", m[3], "| got", m[4][:8])
