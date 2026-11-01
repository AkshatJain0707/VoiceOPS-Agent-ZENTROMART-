import sys, os, csv, json
from pathlib import Path

sys.path.insert(0, "src")
from zentro.factory import Runtime
from zentro.voice import GnaniVoice, for_ear
from zentro.evaluate import check_includes, decision_ok, gold_sections, recall

rt = Runtime()
gv = GnaniVoice(rt.cfg)

test_qids = ["PQ0003", "PQ0002", "PQ0010"]
with open("Capstone_Project_Dataset_FTTT_2026/practice/questions.csv", encoding="utf-8") as f:
    all_qs = {r["id"]: r for r in csv.DictReader(f)}

print("=" * 70)
print("EVALUATION OF 3 USER QUESTIONS ACROSS FULL RAG + AGENT + GNANI PIPELINE")
print("=" * 70 + "\n")

results = []

for qid in test_qids:
    row = all_qs[qid]
    q_text = row["question"]
    exp_dec = row["expected_decision"]
    gold = gold_sections(row.get("gold_evidence", ""))
    must_inc = row.get("must_include", "")
    must_not = row.get("must_not_include", "")

    print(f">>> [{qid}] Language: {row['language']} | Category: {row.get('category', 'general')}")
    print(f"Customer: \"{q_text}\"")

    agent = rt.new_agent(mode="full", channel="voice")
    res = agent.respond(q_text)

    ret_ids = [r["section_id"] for r in res.retrieved]
    rec_val = recall(gold, res.retrieved)

    inc_ok, exc_ok = check_includes(res.text, must_inc, must_not)
    dec_match = decision_ok(res.decision, exp_dec, row.get("acceptable_decisions", ""))

    spoken = for_ear(res.text, rt.cfg.today)
    wav_out = f"eval_{qid}.wav"
    tts_res = gv.synthesize(spoken, wav_out)

    print(f"Decision: {res.decision} (Expected: {exp_dec}) -> Decision Match: {'PASS' if dec_match else 'FAIL'}")
    print(f"Gold Evidence: {gold}")
    print(f"Retrieved Sections: {ret_ids}")
    print(f"Recall: {rec_val}")
    print(f"Cited Sections: {res.used_sections}")
    print(f"Tool Calls: {res.tool_calls}")
    print(f"Agent Reply: \"{res.text}\"")
    print(f"Must Include: {'PASS' if inc_ok else 'FAIL'} (Expected: '{must_inc}')")
    print(f"Must NOT Include: {'PASS' if exc_ok else 'FAIL'}")
    print(f"Spoken (for ear): \"{spoken}\"")
    print(f"Gnani TTS: ok={tts_res.get('ok')}, voice={tts_res.get('voice')}, bytes={tts_res.get('bytes')}")
    print("-" * 70 + "\n")

    results.append({
        "id": qid,
        "question": q_text,
        "decision": res.decision,
        "expected_decision": exp_dec,
        "decision_match": dec_match,
        "recall": rec_val,
        "includes_ok": inc_ok,
        "excludes_ok": exc_ok,
        "reply": res.text,
        "spoken": spoken,
        "tts_ok": tts_res.get("ok", False)
    })

print(f"Summary: {sum(r['decision_match'] for r in results)}/3 decisions matched, {sum(r['includes_ok'] for r in results)}/3 includes matched.")
