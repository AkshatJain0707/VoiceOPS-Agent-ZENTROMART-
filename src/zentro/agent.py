"""The agent brain. One class, three modes: baseline (Step 1), rag (Step 2), full (Step 3)."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from .dates import fmt as fmt_date
from .kb import PINNED_FULL, PINNED_RAG, KnowledgeBase
from .lang import AI_WORDS, has_devanagari, is_hinglish, opening_line
from .prompts import BASELINE, CONDENSE, build_system
from .speech_norm import normalize_utterance
from .tools_adapter import MAX_VERIFY_ATTEMPTS, ToolBox, escalation_flags, looks_verified, slim_order

DECISIONS = ["answer", "clarify", "refuse", "escalate", "dont_know"]
ESC_REASONS = ["high_value_damage", "unauthorised_activity", "legal_threat", "repeat_contact", "refund_overdue",
               "human_requested", "safety_risk", "other"]

T_FINAL = {"name": "final_reply", "description": "Send your final reply to the customer. Always finish with this.",
           "parameters": {"type": "object", "properties": {
               "reply": {"type": "string", "description": "What the customer will read or hear, in their language."},
               "decision": {"type": "string", "enum": DECISIONS},
               "used_sections": {"type": "array", "items": {"type": "string"},
                                 "description": "Section ids from the passages that support the reply."},
               "intent": {"type": "string", "description": "Short intent label, e.g. return_window, refund_status, cancel_order."},
               "escalation_reason": {"type": "string", "enum": ESC_REASONS},
               "handover_summary": {"type": "string", "description": "Required when decision=escalate."}},
               "required": ["reply", "decision"]}}
T_SEARCH = {"name": "search_policy", "description": "Search the policy documents for another topic (English query).",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
T_PRODUCT = {"name": "lookup_product", "description": "Look up the product catalog by SKU (e.g. ELE-0012) or product name. "
             "Returns return_category, price_inr, warranty_months, assured, fulfilled_by.",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
T_PIN = {"name": "check_pincode", "description": "Check delivery coverage, cash on delivery and Express availability for a 6-digit pincode.",
         "parameters": {"type": "object", "properties": {"pincode": {"type": "string"}}, "required": ["pincode"]}}
T_DATE = {"name": "date_math", "description": "Exact date arithmetic. Dates are YYYY-MM-DD or 'today'.",
          "parameters": {"type": "object", "properties": {
              "op": {"type": "string", "enum": ["add_business_days", "add_calendar_days", "business_days_between",
                                                 "calendar_days_between", "weekday"]},
              "date_a": {"type": "string"}, "date_b": {"type": "string"}, "days": {"type": "integer"}},
              "required": ["op"]}}
T_LOOKUP = {"name": "lookup_order", "description": "Look up an order. Call with order_id alone first (returns only a verification prompt). "
            "Call again with verification_value (the customer's last 4 phone digits OR 6-digit delivery pincode) to verify.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "verification_value": {"type": "string"}},
                           "required": ["order_id"]}}

CLAIM_RE = re.compile(r"\b(i\s*(?:have|'ve|’ve)|i\s*am|i'm|i’m|we\s*(?:have|'ve|’ve))\s+(?:just\s+|now\s+)?"
                      r"(created|raised|logged|filed|opened|escalated|forwarded|handed|passed|transferred|cancelled|canceled|"
                      r"refunded|processed|approved|credited)\b", re.I)
TICKET_RE = re.compile(r"\b(?:ticket|टिकट)\s*(?:number|id|no\.?|नंबर)?\s*(?:is|hai|है|:)?\s*[A-Za-z0-9\-]{6,}[.,]?", re.I)
OUT_DECISION = {"dont_know": "answer"}  # the practice set expects "answer" with a polite no-info reply


@dataclass
class State:
    turn: int = 0
    order_id: str = ""
    verified: bool = False
    order: dict = field(default_factory=dict)
    fails: dict = field(default_factory=dict)  # order_id -> failed verification attempts
    secrets: set = field(default_factory=set)
    ticket_id: str = ""
    ticket_created: bool = False
    history: list[dict] = field(default_factory=list)


@dataclass
class AgentReply:
    text: str
    decision: str = "answer"
    internal_decision: str = "answer"
    used_sections: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # {section_id, doc, score, via}
    tool_calls: list[dict] = field(default_factory=list)
    query: str = ""
    ticket_id: str = ""
    ticket_created: bool = False


class SupportAgent:
    def __init__(self, llm, kb: KnowledgeBase | None, toolbox: ToolBox | None, cfg, mode: str = "full",
                 channel: str = "text", top_k: int = 6):
        assert mode in ("baseline", "rag", "full")
        self.llm, self.kb, self.tb, self.cfg = llm, kb, toolbox, cfg
        self.mode, self.channel, self.top_k = mode, channel, top_k
        self.use_rag = mode in ("rag", "full") and kb is not None
        self.use_data = self.use_rag and toolbox is not None       # product / pincode / date tools
        self.use_orders = mode == "full" and toolbox is not None   # order lookup + tickets
        self.pinned = PINNED_FULL if self.use_orders else PINNED_RAG
        self.system_base = BASELINE if mode == "baseline" else build_system(self.use_rag, self.use_orders, channel)
        self.state = State()

    # ------------------------------------------------------------------ helpers
    def _state_block(self) -> str:
        s = self.state
        lines = ["CONVERSATION STATE", f"today: {fmt_date(self.cfg.today)} ({self.cfg.today.isoformat()})", f"turn: {s.turn}"]
        if self.use_orders:
            lines.append(f"order_id_given: {s.order_id or 'none yet'}")
            lines.append(f"identity_verified: {'yes' if s.verified else 'no'}")
            if s.order_id and not s.verified:
                lines.append(f"failed_verifications_for_this_order: {s.fails.get(s.order_id, 0)} of {MAX_VERIFY_ATTEMPTS}")
            if s.verified and s.order:
                lines.append("order_record: " + json.dumps(s.order, ensure_ascii=False)[:3500])
                for f in escalation_flags(s.order):
                    lines.append("FLAG: " + f)
            lines.append(f"handover_ticket: {'created ' + s.ticket_id if s.ticket_created else 'none'}")
        return "\n".join(lines)

    def _needs_rewrite(self, text: str) -> bool:
        return self.state.turn > 1 or has_devanagari(text) or is_hinglish(text)

    def _condense(self, text: str) -> str:
        s = self.state
        facts = ""
        if s.verified and s.order:
            facts = " ".join(f"{k}={s.order[k]}" for k in ("status", "payment_method", "delivery_type") if k in s.order)
        if not self._needs_rewrite(text) and not facts:
            return text
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in s.history[-6:])
        try:
            r = self.llm.chat(CONDENSE, [{"role": "user", "content":
                f"Conversation:\n{convo or '(none)'}\n\nKnown order facts: {facts or 'none'}\nLatest customer message: {text}"}],
                max_tokens=80, fast=True)
            q = r.text.strip().splitlines()[0] if r.text.strip() else ""
            return f"{q} {text}" if q else text  # keep the original words too: it also matches the Hindi FAQ
        except Exception:
            prev = next((m["content"] for m in reversed(s.history) if m["role"] == "user"), "")
            return f"{prev} {text}".strip()

    def _retrieve(self, query: str):
        hits = self.kb.retrieve(query, k=self.top_k, pinned=self.pinned)
        main = [h for h in hits if h.via != "pinned"]
        cov = self.kb.coverage(query, main)
        hint = "good" if cov >= 0.5 else ("partial" if cov >= 0.25 else "weak: the documents may not cover this")
        return hits, hint

    def _scrub(self, text: str) -> str:
        for sec in self.state.secrets:
            if len(sec) >= 4:
                text = text.replace(sec, "")
        return text

    def _handover(self, summary: str, intent: str, reason: str) -> tuple[bool, str]:
        s = self.state
        if s.ticket_created:
            return True, s.ticket_id
        parts = [summary.strip()] if summary.strip() else []
        if s.order_id and f"{s.order_id}" not in summary:
            parts.append(f"Order ID: {s.order_id}.")
        if intent:
            parts.append(f"Intent: {intent}.")
        if reason:
            parts.append(f"Escalation reason: {reason}.")
        if not summary.strip():
            last = next((m["content"] for m in reversed(s.history) if m["role"] == "user"), "")
            parts.insert(0, f"Customer needs a human colleague. Last message: {last[:200]}.")
        text = self._scrub(" ".join(parts))[:900]
        res = self.tb.create_ticket(text, order_id=s.order_id or None, reason=reason or None, intent=intent or None)
        s.ticket_created = res.created
        if res.created:
            s.ticket_id = res.ticket_id
        return res.created, res.ticket_id

    # ------------------------------------------------------------------ main loop
    def respond(self, user_text: str) -> AgentReply:
        s = self.state
        s.turn += 1
        text = normalize_utterance(user_text, getattr(self.tb, "order_fmt", None) if self.tb else None,
                                   getattr(self.tb, "sku_prefixes", None) if self.tb else None)
        out = AgentReply(text="")
        hits, hint, passages = [], "", ""
        if self.use_rag:
            out.query = self._condense(text)
            hits, hint = self._retrieve(out.query)
            out.retrieved = [{"section_id": h.chunk.section_id, "doc": h.chunk.doc, "score": round(h.score, 5),
                              "via": h.via, "status": h.chunk.status} for h in hits]
            passages = f'<passages retrieval="{hint}">\n{self.kb.format_context(hits)}\n</passages>\n\n'
        known = {h.chunk.section_id for h in hits}

        tools = [T_FINAL]
        if self.use_rag:
            tools.append(T_SEARCH)
        if self.use_data:
            tools += [T_PRODUCT, T_PIN, T_DATE]
        if self.use_orders:
            tools.append(T_LOOKUP)
        system = self.system_base + "\n\n" + self._state_block()
        msgs = list(s.history) + [{"role": "user", "content": f"{passages}Customer says: {text}"}]
        final: dict | None = None
        repaired = False

        for _ in range(10):
            r = self.llm.chat(system, msgs, tools=tools, force_tool=True, max_tokens=1000)
            if not r.tool_calls:
                final = {"reply": r.text.strip(), "decision": "answer", "used_sections": []}
                break
            msgs.append({"role": "assistant", "content": r.text, "tool_calls": r.tool_calls})
            fin = next((c for c in r.tool_calls if c["name"] == "final_reply"), None)
            others = [c for c in r.tool_calls if c["name"] != "final_reply"]
            for c in others:
                result = self._run_tool(c, out, known)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                             "content": json.dumps(result, ensure_ascii=False, default=str)[:6000]})
            if fin and others:
                msgs.append({"role": "tool", "tool_call_id": fin["id"], "name": "final_reply",
                             "content": "NOT SENT: call final_reply again now that you have the tool results."})
                system = self.system_base + "\n\n" + self._state_block()  # state may have changed (verification)
                continue
            if fin:
                cand = fin["input"]
                secs = [x for x in cand.get("used_sections", []) if x in known]
                needs_grounding = (self.use_rag and cand.get("decision") in ("answer", "refuse") and not secs
                                   and not (s.verified and s.order) and "weak" not in hint and s.turn > 0
                                   and not self._smalltalk_ok(cand))
                if needs_grounding and not repaired:
                    repaired = True
                    msgs.append({"role": "tool", "tool_call_id": fin["id"], "name": "final_reply",
                                 "content": "REJECTED: no valid section id cited. Cite section ids from the passages, or use "
                                            "decision=dont_know if they do not cover this. Then call final_reply."})
                    continue
                final = cand
                break
            system = self.system_base + "\n\n" + self._state_block()
        if final is None:
            final = {"reply": "I'm sorry, I ran into a problem. Could you say that again?", "decision": "clarify"}

        reply = (final.get("reply") or "").strip()
        decision = final.get("decision") if final.get("decision") in DECISIONS else "answer"
        out.used_sections = [x for x in final.get("used_sections", []) if x in known]

        # Never claim an action the code did not do: a claimed escalation becomes a real ticket, other claims are removed.
        if self.use_orders and decision != "escalate" and CLAIM_RE.search(reply):
            if re.search(r"ticket|escalat|colleague|team|handed|forward", reply, re.I):
                decision = "escalate"
            else:
                reply = CLAIM_RE.sub("I can't do that myself, but", reply)
        if self.use_orders and decision == "escalate":
            ok, tid = self._handover(final.get("handover_summary", ""), final.get("intent", ""), final.get("escalation_reason", ""))
            out.tool_calls.append({"tool": "create_ticket", "created": ok, "ticket_id": tid})
            if ok:
                reply = TICKET_RE.sub("", reply).strip()
                reply += f" Your ticket number is {tid}."
            else:
                reply = ("I'm sorry, I couldn't reach our team to log this just now, so I haven't handed it over. "
                         "Please try again in a few minutes.")
                decision = "answer"
        out.ticket_created, out.ticket_id = s.ticket_created, s.ticket_id

        reply = self._scrub(reply)
        if s.turn == 1 and not AI_WORDS.search(reply):
            reply = opening_line(user_text) + reply
        out.text, out.internal_decision = reply, decision
        out.decision = OUT_DECISION.get(decision, decision)
        s.history.append({"role": "user", "content": text})
        s.history.append({"role": "assistant", "content": reply})
        return out

    def _smalltalk_ok(self, cand: dict) -> bool:
        """Greetings, clarifying questions, and polite conversational phrases need no citation."""
        if cand.get("decision") == "clarify":
            return True
        r = (cand.get("reply") or "").lower()
        return any(w in r for w in ("welcome", "thank you", "thanks", "hello", "hi", "namaste", "goodbye", "bye", "have a great day"))

    def _run_tool(self, call: dict, out: AgentReply, known: set) -> dict:
        name, a, s = call["name"], call["input"], self.state
        if name == "search_policy" and self.kb:
            q = str(a.get("query", ""))
            hits, hint = self._retrieve(q)
            for h in hits:
                known.add(h.chunk.section_id)
                out.retrieved.append({"section_id": h.chunk.section_id, "doc": h.chunk.doc, "score": round(h.score, 5),
                                      "via": "tool:" + h.via, "status": h.chunk.status})
            out.tool_calls.append({"tool": name, "query": q})
            return {"retrieval": hint, "passages": self.kb.format_context(hits)}
        if name == "lookup_product" and self.use_data:
            res = self.tb.lookup_product(str(a.get("query", "")))
            out.tool_calls.append({"tool": name, "query": a.get("query"), "n": len(res.get("matches", []))})
            return res
        if name == "check_pincode" and self.use_data:
            res = self.tb.check_pincode(str(a.get("pincode", "")))
            out.tool_calls.append({"tool": name, "listed": res.get("listed")})
            return res
        if name == "date_math" and self.use_data:
            args = {k: a[k] for k in ("op", "date_a", "date_b", "days") if k in a}
            res = self.tb.date_math(**args)
            out.tool_calls.append({"tool": name, **args, "result": res})
            return res
        if name == "lookup_order" and self.use_orders:
            from .speech_norm import canonical_order_id
            oid = canonical_order_id(str(a.get("order_id", "")), self.tb.order_fmt)
            val = "".join(ch for ch in str(a.get("verification_value") or "") if ch.isdigit()) or None
            if oid != s.order_id:  # new order in focus: verification does not carry over
                s.order_id, s.verified, s.order = oid, False, {}
            if val and s.fails.get(oid, 0) >= MAX_VERIFY_ATTEMPTS:
                res = {"verified": False, "blocked": True,
                       "message": "Verification failed twice for this order. Share no order details; offer general policy help only."}
            else:
                res = self.tb.lookup_order(oid, val)
                if val:
                    s.secrets.add(val)
                if looks_verified(res):
                    s.verified, s.order = True, slim_order(res, self.tb.calc)
                    res = {"verified": True, "note": "Verified. The record is now in CONVERSATION STATE.", **s.order}
                elif val:
                    s.fails[oid] = s.fails.get(oid, 0) + 1
                    res = {"verified": False, "message": "Verification failed. Ask once more if a retry remains."}
                else:
                    res = {"verification_required": True, "message": "Ask for the registered phone last 4 digits OR the delivery pincode."}
            out.tool_calls.append({"tool": name, "order_id": oid, "verification_value": bool(val), "verified": s.verified})
            return res
        return {"error": f"tool {name} not available"}
