import io
import sys
import wave
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIX = ROOT / "tests" / "fixtures" / "mini"

from zentro.agent import SupportAgent  # noqa: E402
from zentro.config import Config  # noqa: E402
from zentro.dates import DateCalc, load_holidays  # noqa: E402
from zentro.factory import Runtime  # noqa: E402
from zentro.kb import PINNED_FULL, KnowledgeBase  # noqa: E402
from zentro.lang import is_hinglish, opening_line  # noqa: E402
from zentro.llm import LLMResponse  # noqa: E402
from zentro.speech_norm import canonical_order_id, infer_id_format, normalize_utterance  # noqa: E402
from zentro.voice import concat_wavs, detect_lang, for_ear, is_unclear, split_sentences, split_wav  # noqa: E402

FMT = ("", "", 8)


@pytest.fixture()
def cfg(monkeypatch):
    monkeypatch.setenv("DATASET_DIR", str(FIX))
    monkeypatch.setenv("EMBED_BACKEND", "none")
    return Config()


# ---------------- knowledge base ----------------
def test_kb_parses_real_format_and_prefers_current_policy(cfg):
    kb = KnowledgeBase(cfg.kb_dir, "none")
    ids = {c.section_id: c for c in kb.chunks}
    assert ids["RET-2"].doc_type == "policy" and ids["RET-2"].status == "current" and ids["RET-2"].effective_from == "2026-08-01"
    assert ids["RETv3-2"].superseded and ids["MKT-1"].doc_type == "marketing"
    hits = kb.search("how many days to return electronics", k=4)
    order = [h.chunk.section_id for h in hits]
    assert order[0] == "RET-2" and order.index("RET-2") < order.index("MKT-1") if "MKT-1" in order else True
    assert "RETv3-2" in [h.chunk.section_id for h in kb.search("My smartwatch was delivered on 20 July. Can I return it?", k=6)]
    assert "RETv3-2" not in [h.chunk.section_id for h in kb.search("electronics return window", k=3)]


def test_kb_crossref_and_pinned(cfg):
    kb = KnowledgeBase(cfg.kb_dir, "none")
    hits = kb.search("cancel shipped order", k=1)
    assert any(h.via == "xref" and h.chunk.section_id == "RET-2" for h in hits)  # bare 'RET-2' mention in CAN-2
    got = {h.chunk.section_id for h in kb.retrieve("anything about returns", k=2, pinned=PINNED_FULL)}
    assert {"ESC-1", "ESC-4"} <= got


def test_kb_vocabulary_bridge_and_coverage(cfg):
    kb = KnowledgeBase(cfg.kb_dir, "none")
    assert "RET-6" in [h.chunk.section_id for h in kb.search("is the collection free if it came torn", k=3)]
    assert "ESC-1" in [h.chunk.section_id for h in kb.search("क्या मैं किसी इंसान से बात कर सकती हूँ", k=3)]
    good = kb.search("fee for clothing return", k=4)
    bad = kb.search("cryptocurrency mining rigs", k=4)
    assert kb.coverage("fee for clothing return", good) > kb.coverage("cryptocurrency mining rigs", bad)


# ---------------- dates ----------------
def test_business_days_skip_sunday_and_holiday():
    calc = DateCalc(date(2026, 10, 20), {date(2026, 10, 21)})
    assert calc.add_business_days(date(2026, 10, 20), 3) == date(2026, 10, 24)
    assert calc.add_business_days(date(2026, 10, 20), 4) == date(2026, 10, 26)  # Sunday 25 skipped
    assert calc.business_days_between(date(2026, 10, 20), date(2026, 10, 26)) == 4
    assert calc.run("add_calendar_days", "2026-10-20", days=7)["result"] == "2026-10-27"


def test_holidays_loaded(cfg):
    assert date(2026, 10, 21) in load_holidays(cfg.data_dir)


# ---------------- speech normalisation ----------------
def test_order_id_and_sku_normalisation():
    assert "89782837" in normalize_utterance("Order number hai 8978 2837.", FMT)
    assert "89782837" in normalize_utterance("ऑर्डर नंबर है 8978 2837।", FMT)
    assert "23554022" in normalize_utterance("the order number is two three five five four zero two two", FMT)
    assert "ELE-0012 pe" in normalize_utterance("so sku ele-zero zero one to pe zentromart", FMT, ["ELE"])
    assert "ELE-0008" in normalize_utterance("sku ele zero zero zero eight", FMT, ["ELE"])
    assert infer_id_format(["89782837", "18599253"]) == FMT and canonical_order_id("8978 2837", FMT) == "89782837"
    assert "4533" in normalize_utterance("Last four digits 4533 hain.", FMT)  # a lone 4-digit value is untouched


def test_language_helpers():
    assert is_hinglish("Mujhe order ke baare mein help chahiye") and not is_hinglish("I need help with my order")
    assert "एआई" in opening_line("नमस्ते") and "AI assistant" in opening_line("Namaste, mera order kab aayega") and "AI" in opening_line("hi")


# ---------------- voice text and audio ----------------
def test_for_ear_dates_ids_and_tickets():
    t = for_ear("**Refund** by 3 November. Order 89782837. Your ticket number is TKT" + "0" * 26 + "a1b2c3. [POL-RET-02]", date(2026, 10, 20))
    assert "3 November 2026" in t and "*" not in t and "POL-RET" not in t
    assert "8978 2837" in t and "a1b2c3".upper().replace("", " ").strip() in t.replace("  ", " ")
    assert "20 October 2026" in for_ear("on 20 October 2026", date(2026, 10, 20))
    assert "₹4,999" in for_ear("Refund of ₹4,999 is due.", date(2026, 10, 20))  # money is not an ID


def test_unclear_and_lang_detection():
    assert is_unclear("") and is_unclear(" ,. ") and not is_unclear("where is my order")
    assert detect_lang("आपका ऑर्डर कल पहुँचेगा") == "hi-IN" and detect_lang("hello") == "en-IN"
    assert detect_lang("Haan, aapka refund kal tak aa jayega") == "hi-en"
    assert len(split_sentences("One. " * 200, 100)) > 3


def _wav(seconds, rate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * rate * seconds)
    return buf.getvalue()


def test_split_and_concat_wav(tmp_path):
    p = tmp_path / "long.wav"
    p.write_bytes(_wav(130))
    parts = split_wav(p, 55)
    assert len(parts) == 3
    with wave.open(io.BytesIO(concat_wavs(parts))) as w:
        assert abs(w.getnframes() / w.getframerate() - 130) < 0.01


# ---------------- agent flows with a scripted LLM (tests OUR code, not a model) ----------------
class Scripted:
    def __init__(self, steps):
        self.steps, self.i, self.states = steps, 0, []

    def chat(self, system, messages, tools=None, force_tool=False, max_tokens=0, fast=False):
        if fast:
            return LLMResponse(text="refund timeline returns")
        self.states.append(system)
        calls = self.steps[self.i]
        self.i += 1
        return LLMResponse(tool_calls=[{"id": f"t{self.i}_{j}", "name": n, "input": a} for j, (n, a) in enumerate(calls)])


def make_agent(cfg, steps, mode="full"):
    rt = Runtime(cfg)
    llm = Scripted(steps)
    return SupportAgent(llm, rt.kb, rt.tools, cfg, mode=mode, channel="text"), rt, llm


def fin(reply, decision="answer", **kw):
    return ("final_reply", {"reply": reply, "decision": decision, **kw})


def test_verification_flow_then_refusal_without_ticket(cfg):
    agent, rt, llm = make_agent(cfg, [
        [fin("Hello! I'm the AI assistant. What is your order number?", "clarify")],
        [("lookup_order", {"order_id": "8978 2837"})],
        [fin("Thanks. Please share the last 4 digits of your phone or the delivery pincode.", "clarify")],
        [("lookup_order", {"order_id": "89782837", "verification_value": "4533"})],
        [fin("Thank you, you're verified. How can I help?", "answer", used_sections=["ESC-1"])],
        [fin("Sorry, I can't add a voucher. Policy does not allow compensation.", "refuse", used_sections=["ESC-4"])],
    ])
    r1 = agent.respond("Hello, mujhe order ke baare mein help chahiye.")
    assert r1.decision == "clarify" and "AI" in r1.text
    r2 = agent.respond("Order number hai 8978 2837.")
    assert agent.state.order_id == "89782837" and not agent.state.verified and "private" not in r2.text
    assert r2.tool_calls[0]["verified"] is False and r2.tool_calls[0]["verification_value"] is False
    r3 = agent.respond("Last four digits 4533 hain.")
    assert agent.state.verified and "delivery_pincode" not in agent.state.order     # pincode never shown to the model
    r4 = agent.respond("Order late aaya, mujhe 300 rupaye ka voucher chahiye.")
    assert r4.decision == "refuse" and not r4.ticket_created
    assert "4533" not in str(r3.tool_calls) and "delivery_pincode" not in llm.states[-1]


def test_failed_verification_allows_one_retry_then_blocks(cfg):
    agent, _, _ = make_agent(cfg, [
        [("lookup_order", {"order_id": "89782837", "verification_value": "1266"})], [fin("That didn't match. Could you try again?", "clarify")],
        [("lookup_order", {"order_id": "89782837", "verification_value": "1266"})], [fin("I can't share details. Registered phone needed.", "refuse", used_sections=["ESC-4"])],
        [("lookup_order", {"order_id": "89782837", "verification_value": "4533"})], [fin("Sorry, I can't verify this order.", "refuse", used_sections=["ESC-4"])],
    ])
    agent.respond("order 8978 2837 last four 1266"); assert agent.state.fails["89782837"] == 1
    agent.respond("1266 again"); assert agent.state.fails["89782837"] == 2 and not agent.state.verified
    agent.respond("ok it is 4533")                       # a third try, even with the right value, is blocked
    assert not agent.state.verified and agent.state.order == {}


def test_switching_orders_resets_verification(cfg):
    agent, _, _ = make_agent(cfg, [
        [("lookup_order", {"order_id": "89782837", "verification_value": "4533"})], [fin("Verified.", "answer", used_sections=["ESC-1"])],
        [("lookup_order", {"order_id": "18599253"})], [fin("Please share last 4 digits or pincode for that order.", "clarify")],
    ])
    agent.respond("order 8978 2837 last four 4533"); assert agent.state.verified
    agent.respond("different order 1859 9253"); assert not agent.state.verified and agent.state.order_id == "18599253"


def test_repeat_contact_flag_and_escalation_creates_real_ticket(cfg):
    import tools
    tools.tools.TICKETS.clear()
    agent, rt, llm = make_agent(cfg, [
        [("lookup_order", {"order_id": "89782837", "verification_value": "4533"})], [fin("Verified.", "answer", used_sections=["ESC-1"])],
        [fin("A human colleague will contact you.", "escalate", used_sections=["ESC-1"], intent="repeat_contact",
             escalation_reason="repeat_contact", handover_summary="Third contact this week about a late order; customer wants status.")],
    ])
    agent.respond("order 89782837 last four 4533")
    r = agent.respond("Main is hafte teesri baar call kar rahi hoon isi baare mein.")
    assert "FLAG: ESC-1 rule 4" in llm.states[-1]                      # code detected contacts_last_7_days >= 2
    assert r.decision == "escalate" and r.ticket_created and r.ticket_id.startswith("TKT") and r.ticket_id in r.text
    t = tools.tools.TICKETS[-1]
    assert t["order_id"] == "89782837" and t["intent"] == "repeat_contact" and "4533" not in t["summary"] and len(t["summary"]) >= 20


def test_never_claims_ticket_that_was_not_created(cfg):
    import tools
    agent, _, _ = make_agent(cfg, [[fin("I've raised a ticket for you.", "answer", used_sections=["ESC-1"])]])
    tools.tools.FAIL_TICKETS = True
    try:
        r = agent.respond("my parcel is damaged")
    finally:
        tools.tools.FAIL_TICKETS = False
    assert not r.ticket_created and "haven't handed it over" in r.text and "raised a ticket" not in r.text


def test_claimed_action_without_escalation_words_is_removed(cfg):
    agent, _, _ = make_agent(cfg, [[fin("I've approved your refund of 500 rupees.", "answer")]])
    r = agent.respond("refund please")
    assert "approved" not in r.text.lower().replace("can't do that myself", "")


def test_ungrounded_answer_is_bounced_once_and_dont_know_maps_to_answer(cfg):
    agent, _, _ = make_agent(cfg, [[fin("Returns take 90 days.", "answer", used_sections=["MADE-UP"])],
                                   [fin("Sorry, I don't have that information. I can connect you to a colleague.", "dont_know")]])
    r = agent.respond("what is the return window for sarees on mars")
    assert "90 days" not in r.text and r.decision == "answer" and r.internal_decision == "dont_know"


def test_data_tools_product_and_pincode(cfg):
    agent, rt, _ = make_agent(cfg, [
        [("lookup_product", {"query": "ELE-0012"}), ],
        [("check_pincode", {"pincode": "194101"})],
        [fin("Warranty is 12 months; Leh is not serviceable.", "answer", used_sections=["RET-2"])],
    ], mode="rag")
    r = agent.respond("warranty for sku ele zero zero one two, and do you deliver to 194101?")
    assert [c["tool"] for c in r.tool_calls] == ["lookup_product", "check_pincode"]
    assert rt.tools.lookup_product("Rangoli kurta")["matches"][0]["sku"] == "FSH-0038"
    assert rt.tools.check_pincode("194101")["serviceable"] == "False"


def test_opening_line_added_in_customer_language(cfg):
    agent, _, _ = make_agent(cfg, [[fin("Aapka order kal aayega.", "answer", used_sections=["CAN-2"])]])
    assert agent.respond("Mera order kab aayega, mujhe batao").text.startswith("Namaste, main ZentroMart ki AI assistant hoon.")


# ---------------- voice turn plumbing (no network) ----------------
def test_voice_turn_recovers_from_unclear_audio_then_answers(cfg, tmp_path):
    from zentro.voice import UNCLEAR, VoiceAgent

    class FakeVoice:
        def __init__(self, transcripts): self.t = list(transcripts); self.spoken = []
        def transcribe(self, p, language=None): return {"transcript": self.t.pop(0), "error": ""}
        def synthesize(self, text, out, lang="en-IN", voice=None, speed=1.0):
            self.spoken.append(text); return {"ok": True, "lang": lang}

    agent, _, _ = make_agent(cfg, [[fin("Refunds take 5 business days.", "answer", used_sections=["RET-2"])]])
    fv = FakeVoice(["", "", "uh", "uh", "how long does a refund take"])   # each unclear turn also retries Hindi
    va = VoiceAgent(agent, fv, cfg)
    r1 = va.turn("a.wav", tmp_path / "1.wav"); r2 = va.turn("b.wav", tmp_path / "2.wav"); r3 = va.turn("c.wav", tmp_path / "3.wav")
    assert r1["reply"] == UNCLEAR[0] and r2["reply"] == UNCLEAR[1] and r1["unclear"]
    assert r3["decision"] == "answer" and va.unclear_streak == 0 and len(fv.spoken) == 3


def test_batch_grouping():
    from zentro.batch import group_files
    g = group_files([Path("c1_turn2.wav"), Path("c1_turn1.wav"), Path("solo.wav"), Path("c2_t1.wav")])
    assert [p.name for p in g["c1"]] == ["c1_turn1.wav", "c1_turn2.wav"] and "solo" in g and "c2" in g


# ---------------- evaluator helpers ----------------
def test_eval_must_include_semantics_and_decisions():
    from zentro.evaluate import check_includes, decision_ok, expected_calls, gold_sections
    assert check_includes("Refunds take 5 business days", "business days / working days ; refund", "") == (True, True)
    assert check_includes("Refund is quick", "5 business days", "")[0] is False
    assert check_includes("₹300 has been credited", "sorry / apolog", "₹300 has been ; voucher added")[1] is False
    assert decision_ok("answer", "answer", "answer ; escalate") and decision_ok("escalate", "answer", "answer ; escalate")
    assert not decision_ok("refuse", "answer", "answer")
    assert gold_sections("RET-2 ; products.csv:ELE-0012 ; KB-00.3 ; RETv3-2") == ["RET-2", "KB-00.3", "RETv3-2"]
    assert expected_calls("lookup_order(order_id=89782837, verification_value=4533)\ncreate_ticket()") == [
        ("lookup_order", "89782837", True), ("create_ticket",)]


def test_spoken_verification_digits_collapse():
    assert "4400" in normalize_utterance("last four digits double four double zero", FMT)
    assert "560001" in normalize_utterance("pincode is five six zero zero zero one", FMT)
    assert "ZentroMart" in normalize_utterance("i bought this on zentro mart", FMT)


def test_smalltalk_and_polite_closings(cfg):
    agent, _, _ = make_agent(cfg, [[fin("You are welcome! Have a great day.", "answer")]])
    r = agent.respond("Thank you for your help!")
    assert "welcome" in r.text.lower()


def test_for_ear_rupees_formatting():
    t = for_ear("The fee is Rs. 500 for reverse pickup.", date(2026, 10, 20))
    assert "rupees" in t and "Rs." not in t


def test_policy_bilingual_cross_references(cfg):
    kb = KnowledgeBase(cfg.kb_dir, "none")
    assert "FAQH-1" in kb.by_section or "CAN-2" in kb.by_section
    expanded = kb.search("cancel shipped order", k=2, expand=True)
    assert len(expanded) >= 1
