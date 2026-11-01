"""Own evaluation (10 marks): Step 1 vs Step 2 vs Step 3 on the practice sets.
Deterministic checks first (must_include / must_not_include / acceptable decisions / tool calls / retrieval recall of
gold_evidence), plus an optional LLM judge against reference_answer. Uses the practice CSV columns as shipped."""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from .factory import Runtime

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9.]+$")
JUDGE = """You grade a customer-support agent's reply against a reference answer. Be strict about facts (numbers, days, amounts,
conditions) and about invented policy; be lenient about wording and language (Hindi/Hinglish replies are fine).
Return JSON only: {"correct": true|false, "reason": "<=20 words"}"""


def split_items(cell: str) -> list[list[str]]:
    """' ; ' separates required items; ' / ' separates alternatives inside an item (any one is enough)."""
    return [[a.strip() for a in item.split(" / ") if a.strip()] for item in re.split(r"\s;\s", (cell or "").strip()) if item.strip()]


def check_includes(reply: str, must_include: str, must_not: str) -> tuple[bool, bool]:
    low = reply.lower().replace(",", "")
    inc = all(any(a.lower().replace(",", "") in low for a in alts) for alts in split_items(must_include))
    exc = not any(any(a.lower().replace(",", "") in low for a in alts) for alts in split_items(must_not))
    return inc, exc


def decision_ok(got: str, expected: str, acceptable: str) -> bool:
    ok = {x.strip() for x in re.split(r"[;,]", (acceptable or expected or "")) if x.strip()}
    return got in ok or got == expected


def gold_sections(cell: str) -> list[str]:
    return [g.strip() for g in re.split(r"\s*;\s*", cell or "") if ID_RE.match(g.strip()) and ":" not in g]


def recall(gold: list[str], retrieved: list[dict]) -> float | None:
    if not gold:
        return None
    got = {r["section_id"] for r in retrieved}
    return sum(g in got for g in gold) / len(gold)


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def judge(rt: Runtime, q: str, ref: str, reply: str) -> bool | None:
    if not ref:
        return None
    try:
        r = rt.llm.chat(JUDGE, [{"role": "user", "content": f"Customer: {q}\nReference answer: {ref}\nAgent reply: {reply}"}],
                        max_tokens=120, fast=True)
        m = re.search(r"\{.*\}", r.text, re.S)
        return bool(json.loads(m.group(0)).get("correct")) if m else None
    except Exception:
        return None


def run_questions(rt: Runtime, rows: list[dict], mode: str, col: str, use_judge: bool) -> list[dict]:
    out = []
    for row in rows:
        agent = rt.new_agent(mode, "text")
        r = agent.respond(row[col])
        inc, exc = check_includes(r.text, row.get("must_include", ""), row.get("must_not_include", ""))
        out.append({"id": row["id"], "mode": mode, "language": row["language"], "question": row[col], "reply": r.text,
                    "decision": r.decision, "decision_ok": decision_ok(r.decision, row["expected_decision"], row.get("acceptable_decisions", "")),
                    "includes": inc, "excludes_ok": exc, "recall": recall(gold_sections(row.get("gold_evidence", "")), r.retrieved) if mode != "baseline" else None,
                    "judge": judge(rt, row[col], row.get("reference_answer", ""), r.text) if use_judge else None,
                    "answerable": row.get("answerable")})
        print(f"  [{mode}] {row['id']}: decision={r.decision} includes={inc} excl_ok={exc}")
    return out


def expected_calls(cell: str) -> list[tuple]:
    out = []
    for m in re.finditer(r"(lookup_order|create_ticket)\(([^)]*)\)", cell or ""):
        kw = dict(re.findall(r"(\w+)=([^,\s]+)", m.group(2)))
        out.append(("create_ticket",) if m.group(1) == "create_ticket" else ("lookup_order", kw.get("order_id"), "verification_value" in kw))
    return out


def run_convs(rt: Runtime, rows: list[dict], mode: str, use_judge: bool) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["conversation_id"]].append(r)
    out = []
    for cid, turns in groups.items():
        turns.sort(key=lambda r: int(r["turn"]))
        agent = rt.new_agent(mode, "text")
        for row in turns:
            r = agent.respond(row["customer_message"])
            inc, exc = check_includes(r.text, row.get("must_include", ""), row.get("must_not_include", ""))
            got = [("lookup_order", c.get("order_id"), c.get("verification_value")) for c in r.tool_calls if c["tool"] == "lookup_order"]
            got += [("create_ticket",) for c in r.tool_calls if c["tool"] == "create_ticket" and c.get("created")]
            want = expected_calls(row.get("expected_tool_calls", ""))
            out.append({"conv": cid, "turn": row["turn"], "scenario": row["scenario"], "mode": mode, "message": row["customer_message"],
                        "reply": r.text, "decision": r.decision, "decision_ok": r.decision == row["expected_decision"],
                        "includes": inc, "excludes_ok": exc, "tools_ok": got == want if mode == "full" else None,
                        "recall": recall(gold_sections(row.get("gold_evidence", "")), r.retrieved)})
            print(f"  [{mode}] {cid} t{row['turn']}: decision={r.decision} (want {row['expected_decision']}) includes={inc} tools_ok={got == want}")
    return out


def rate(rs: list[dict], key: str):
    v = [r[key] for r in rs if r.get(key) is not None]
    return round(sum(map(float, v)) / len(v), 3) if v else None


def summarize(rs: list[dict]) -> dict:
    return {"n": len(rs), "decision_match": rate(rs, "decision_ok"), "must_include": rate(rs, "includes"),
            "must_not_include_ok": rate(rs, "excludes_ok"), "judge_correct": rate(rs, "judge"),
            "retrieval_recall": rate(rs, "recall"), "tool_calls_match": rate(rs, "tools_ok")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--limit", type=int, default=0, help="first N questions / 2N conversation rows, for a quick check")
    ap.add_argument("--col", default="question", help="question column: question or asr_text (speech-to-text version)")
    ap.add_argument("--judge", action="store_true", help="also grade against reference_answer with an LLM")
    a = ap.parse_args()
    rt = Runtime()
    qs, cs = read_csv(rt.cfg.practice_dir / "questions.csv"), read_csv(rt.cfg.practice_dir / "conversations.csv")
    if a.limit:
        qs, cs = qs[:a.limit], cs[:a.limit * 4]
    res, summary = {}, {}
    plan = [("questions", lambda m: run_questions(rt, qs, m, a.col, a.judge), ("baseline", "rag")),
            ("conversations", lambda m: run_convs(rt, cs, m, a.judge), ("rag", "full"))]
    for name, fn, modes in plan:
        for m in modes:
            print(f"\n== {name} / {m} ==")
            res[f"{name}/{m}"] = fn(m)
            summary[f"{name}/{m}"] = summarize(res[f"{name}/{m}"])
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, "details.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(a.out, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# Evaluation Results Summary\n",
          "| Mode | Decision Match | Must Include | Must Not Include OK | Retrieval Recall | Tool Calls Match |",
          "|---|---|---|---|---|---|"]
    for k, v in summary.items():
        dec = f"{v.get('decision_match', 0)*100:.1f}%" if v.get('decision_match') is not None else "-"
        inc = f"{v.get('must_include', 0)*100:.1f}%" if v.get('must_include') is not None else "-"
        exc = f"{v.get('must_not_include_ok', 0)*100:.1f}%" if v.get('must_not_include_ok') is not None else "-"
        rec = f"{v.get('retrieval_recall', 0)*100:.1f}%" if v.get('retrieval_recall') is not None else "-"
        tc = f"{v.get('tool_calls_match', 0)*100:.1f}%" if v.get('tool_calls_match') is not None else "-"
        md.append(f"| {k} | {dec} | {inc} | {exc} | {rec} | {tc} |")
    Path(a.out, "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
