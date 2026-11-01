"""Plumbing check on the REAL dataset with a scripted 'oracle' model: for every practice conversation turn it issues exactly the
tool calls the practice CSV expects, then compares what our code actually did (order-id normalisation, verification, retry
cap, tickets). No LLM or API key needed. Usage: python scripts_smoke_real.py <dataset_dir>"""
import csv, os, re, sys
from pathlib import Path
ds = Path(sys.argv[1]).resolve()
os.environ["DATASET_DIR"] = str(ds); os.environ["EMBED_BACKEND"] = "none"
sys.path.insert(0, str(Path(__file__).parent / "src"))
from zentro.agent import SupportAgent
from zentro.config import Config
from zentro.factory import Runtime
from zentro.llm import LLMResponse

rows = list(csv.DictReader(open(ds / "practice/conversations.csv", encoding="utf-8")))
rt = Runtime(Config())
print("order id format:", rt.order_fmt, "| chunks:", len(rt.kb.chunks), "| products:", len(rt.tools.products), "| sku prefixes:", rt.tools.sku_prefixes)

class Oracle:
    def __init__(self): self.q = []
    def chat(self, system, messages, tools=None, force_tool=False, max_tokens=0, fast=False):
        if fast: return LLMResponse(text="policy")
        return self.q.pop(0)

def parse(cell):
    out = []
    for m in re.finditer(r"(lookup_order|create_ticket)\(([^)]*)\)", cell or ""):
        kw = dict(re.findall(r"(\w+)=([^,\s]+)", m.group(2)))
        out.append((m.group(1), kw))
    return out

bad = tot = 0
convs = {}
for r in rows: convs.setdefault(r["conversation_id"], []).append(r)
for cid, turns in convs.items():
    turns.sort(key=lambda r: int(r["turn"]))
    llm = Oracle(); agent = SupportAgent(llm, rt.kb, rt.tools, rt.cfg, mode="full")
    for r in turns:
        exp = parse(r["expected_tool_calls"]); calls, steps = [], []
        for name, kw in exp:
            if name == "lookup_order": steps.append(LLMResponse(tool_calls=[{"id": f"l{len(steps)}", "name": "lookup_order", "input": kw}]))
        dec = {"escalate": "escalate"}.get(r["expected_decision"], r["expected_decision"])
        ids = [g.strip() for g in re.split(r"\s*;\s*", r["gold_evidence"] or "") if g.strip()]
        steps.append(LLMResponse(tool_calls=[{"id": "f", "name": "final_reply", "input": {
            "reply": "ok, a human colleague will contact you" if dec == "escalate" else "ok", "decision": dec, "used_sections": ids,
            "intent": r["scenario"], "escalation_reason": "other",
            "handover_summary": f"Customer scenario {r['scenario']}: needs a human colleague; checked the order record and policy."}}]))
        llm.q = steps
        res = agent.respond(r["customer_message"])
        got = [(c["tool"], c.get("order_id"), c.get("verification_value")) for c in res.tool_calls if c["tool"] in ("lookup_order", "create_ticket")]
        want = [(n, kw.get("order_id"), "verification_value" in kw) for n, kw in exp if n == "lookup_order"] + ([("create_ticket", None, None)] if any(n == "create_ticket" for n, _ in exp) else [])
        tot += 1
        ok = got == [(a, b, c) for a, b, c in want]
        ok = ok and (res.ticket_created == any(n == "create_ticket" for n, _ in exp) or not any(n == "create_ticket" for n, _ in exp))
        if not ok:
            bad += 1; print("MISMATCH", cid, r["turn"], "| want", want, "| got", got, "| msg:", r["customer_message"][:50])
print(f"\n{tot - bad}/{tot} turns: tool calls and ticket outcomes match the practice expectations")
tk = ds / "tools" / "tickets_created.csv"
if tk.exists(): print("tickets written by the real tool:", sum(1 for _ in open(tk)) - 1); tk.unlink()
