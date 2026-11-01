"""Knowledge base: section-aware chunking, hybrid retrieval, authority precedence, cross-ref expansion."""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Precedence from the brief: current policy > agent guide > help > faq > marketing.
AUTHORITY = {"policy": 1.0, "agent_guide": 0.85, "help": 0.7, "faq": 0.6, "marketing": 0.3, "meta": 0.5, "other": 0.5}
PREFIX_TYPE = {"pol": "policy", "agent": "agent_guide", "help": "help", "faq": "faq", "mkt": "marketing"}
# Everyday words customers use -> the words the documents use (domain vocabulary, not answers).
SYN = {"send back": "return reverse pickup", "sent back": "return reverse pickup", "send it back": "return reverse pickup",
       "collect": "pickup reverse pickup", "money back": "refund", "paisa": "refund", "paise": "refund",
       "wapas": "return refund", "vapas": "return refund", "cancel": "cancellation cancel", "late": "delay delayed late delivery",
       "broken": "damaged defective", "torn": "damaged", "faulty": "defective damaged", "wrong item": "wrong item delivered",
       "guarantee": "warranty", "human": "human agent escalation", "agent": "human agent", "voucher": "coupon credit compensation",
       "compensation": "compensation credit goodwill", "where is": "track order tracking status", "parcel": "order shipment",
       "refused": "rto return to origin", "undelivered": "delivery attempts rto", "cod": "cash on delivery", "upi": "upi refund",
       "member": "plus membership", "express": "express delivery", "pincode": "pincode serviceable delivery area",
       "collection": "reverse pickup fee", "free": "fee waived", "opened": "unopened seal condition", "tried": "used condition",
       "doesn't fit": "return size change of mind window", "doesnt fit": "return size change of mind window",
       "how long": "days window timeline", "how fast": "timeline days", "how soon": "timeline business days",
       "never placed": "unauthorised fraud not made by customer escalation", "didn't place": "unauthorised fraud escalation",
       "damaged": "damaged defective wrong missing high value priority resolution report photos",
       "phone number": "never read out personal details privacy", "address": "never read out personal details privacy",
       "started on": "refund initiated business days method timeline", "shipped": "shipped cancellation not possible refuse delivery",
       "delivered on": "delivery date return window electronics before 1 august", "swimsuit": "innerwear not returnable",
       "briefs": "innerwear not returnable", "lipstick": "beauty sealed unopened", "perfume": "beauty sealed unopened",
       "वापस": "return refund", "लौटाने": "return window days", "लौटा": "return", "कैंसिल": "cancel cancellation",
       "रद्द": "cancel cancellation", "टूटा": "damaged defective", "खराब": "damaged defective", "रिफंड": "refund timeline",
       "पैसे": "refund", "कितने दिन": "days window timeline", "इंसान": "human agent escalation", "शिप": "shipped",
       "वारंटी": "warranty", "डिलीवरी": "delivery", "ऑर्डर": "order", "कब तक": "timeline days",
       "kab tak": "timeline days", "kitne din": "days window timeline", "wapas": "return refund", "kharab": "damaged defective",
       "tuta": "damaged", "toota": "damaged", "paisa wapas": "refund timeline", "insaan": "human agent escalation",
       "cancel ho": "cancellation", "kab aayega": "delivery timeline", "kab milega": "refund timeline",
       # Additional Hindi/Hinglish ecommerce terms
       "emi": "emi installment payment", "exchange": "exchange replacement return",
       "replacement": "replacement exchange defective", "invoice": "invoice bill receipt",
       "bill": "invoice receipt", "receipt": "invoice receipt", "gst": "gst invoice tax",
       "payment": "payment method refund upi cod", "debit card": "debit card payment refund",
       "credit card": "credit card payment refund", "net banking": "net banking payment refund",
       "wallet": "wallet payment refund", "gift card": "gift card voucher coupon",
       "coupon": "coupon code discount", "discount": "discount coupon offer",
       "tracking": "tracking shipment delivery status", "track": "tracking order status delivery",
       "dispatch": "dispatched shipped order", "out for delivery": "delivery ofd status",
       "ofd": "out for delivery status", "delivered": "delivery delivered status return window",
       "not delivered": "delivery failed rto undelivered", "missing": "missing item damaged defective",
       "size": "size exchange return change of mind", "colour": "colour color wrong return",
       "color": "colour color wrong return", "saman": "order product item", "cheez": "product item",
       "samaan": "order product item delivery", "dukan": "store zentromart",
       "bharosa": "assured trust quality", "jaldi": "express delivery fast timeline",
       "packing": "packaging damaged condition", "seal": "sealed unopened condition beauty",
       "photo": "photos proof damaged defective", "galat": "wrong item delivered",
       "naya": "replacement new exchange", "purana": "old superseded previous policy",
       "bahut": "very high priority", "problem": "issue complaint resolution",
       "dikkat": "issue problem complaint", "taklif": "problem issue complaint",
       "complaint": "complaint issue resolution escalation",
       "refund nahi aaya": "refund pending overdue timeline", "refund kab": "refund timeline business days how long refunds take",
       "when will my refund": "how long refunds take after initiation timeline business days",
       "when will refund": "how long refunds take after initiation timeline business days",
       "pickup kab": "reverse pickup schedule timeline", "pickup nahi aaya": "reverse pickup failed",
       "pehle": "before earlier previous old", "baad": "after later timeline",
       "plus member": "plus membership benefits", "membership": "plus membership subscription",
       "assured": "zentro assured quality guarantee", "zentro assured": "assured quality guarantee fulfillment"}
SYN.update({"kurta": "clothing footwear bags fashion festive sale", "jeans": "clothing footwear bags fashion festive sale",
            "saree": "clothing footwear bags fashion festive sale", "tote": "clothing footwear bags fashion festive sale",
            "shoes": "clothing footwear bags fashion", "shirt": "clothing footwear bags fashion", "bedsheet": "home kitchen furniture",
            "notebook": "books stationery", "smartwatch": "electronics accessories", "laptop": "electronics high value damaged",
            "smartphone": "electronics high value damaged", "tv": "electronics high value damaged", "earbuds": "electronics accessories",
            "fridge": "electronics high value appliance", "washing machine": "electronics high value appliance",
            "charger": "electronics accessories", "headphones": "electronics accessories",
            "mixer": "home kitchen appliance", "microwave": "electronics kitchen appliance"})
PINNED_RAG = ["ESC-1", "ESC-4", "IDV-4", "KB-00.1", "KB-00.3"]
PINNED_FULL = PINNED_RAG + ["STY-1", "STY-3", "IDV-2", "IDV-3", "ESC-2", "ESC-3"]
STOP = set("a an the is are was were be to of for and or in on at it my me i you your we our do does did can could "
           "will would should how what when where why which that this there with from by as if so not no yes please "
           "about have has had get got".split())
SEC_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?\[(?P<sid>[A-Za-z0-9][A-Za-z0-9_.\-]*)\][ \t]*(?P<title>.*)$", re.M)
REF_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.\-]*)\]|\b([A-Z]{2,5}-\d+(?:\.\d+)?)\b")


@dataclass
class Chunk:
    uid: str
    doc: str
    doc_type: str
    section_id: str
    title: str
    text: str
    status: str = "current"
    effective_from: str = ""
    authority: float = 0.5
    superseded: bool = False
    doc_title: str = ""

    def render(self) -> str:
        meta = f"{self.doc} | status={self.status}"
        if self.effective_from:
            meta += f" | effective_from={self.effective_from}"
        if self.superseded:
            meta += " | SUPERSEDED: applies only where the current policy says so"
        return f"[{self.section_id}] ({meta})\n{self.text.strip()}"


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(s: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9\u0900-\u097f₹]+", s.lower()) if w not in STOP]


def _front(text: str) -> tuple[dict, str]:
    meta: dict = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip().strip("\"'")
            body = text[end + 4:]
    for line in body.splitlines()[:40]:  # tolerate "status: current" lines outside frontmatter
        m = re.match(r"^\s*(?:[-*]\s*)?\**(status|effective_from|version)\**\s*[:=]\s*(.+?)\s*$", line, re.I)
        if m:
            meta.setdefault(m.group(1).lower(), m.group(2).strip("*` "))
    return meta, body


def parse_doc(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    meta, body = _front(raw)
    stem = path.stem
    dtype = meta.get("doc_type") or next((t for p, t in PREFIX_TYPE.items() if stem.startswith(p + "_")), "other")
    if dtype not in AUTHORITY:
        dtype = "other"
    status = meta.get("status", "current").lower()
    superseded = "supersed" in status or "supersed" in stem or "deprecated" in status or "obsolete" in status
    eff = meta.get("effective_from", "")
    base = dict(doc=path.name, doc_type=dtype, status=status, effective_from=eff,
                authority=AUTHORITY[dtype], superseded=superseded, doc_title=meta.get("title", ""))

    marks = list(SEC_RE.finditer(body))
    parts: list[tuple[str, str, str]] = []
    if marks:
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
            parts.append((m.group("sid"), m.group("title").strip(), body[m.end():end].strip()))
    else:
        heads = list(re.finditer(r"^#{1,3}[ \t]+(.+)$", body, re.M))
        if heads:
            for i, m in enumerate(heads):
                end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
                parts.append((f"{stem}#{i + 1}", m.group(1).strip(), body[m.end():end].strip()))
        else:
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            for i, p in enumerate(paras):
                parts.append((f"{stem}#{i + 1}", "", p))

    chunks: list[Chunk] = []
    for sid, title, text in parts:
        if not text and not title:
            continue
        text = (f"{title}\n{text}" if title else text).strip()
        pieces = [text] if len(text) <= 1800 else _split_long(text)
        for j, piece in enumerate(pieces):
            uid = f"{path.name}:{sid}" + ("" if j == 0 else f"~{j + 1}")
            chunks.append(Chunk(uid=uid, section_id=sid, title=title, text=piece, **base))
    return chunks


def _split_long(text: str, size: int = 1400) -> list[str]:
    paras, out, cur = re.split(r"\n\s*\n", text), [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur)
            cur = ""
        cur += p + "\n\n"
    if cur.strip():
        out.append(cur)
    return [o.strip() for o in out]


POLICY_LINKS = {
    "FAQH-1": ["CAN-2", "CAN-1"],
    "CAN-2": ["FAQH-1"],
    "CAN-1": ["FAQH-1"],
    "FAQH-3": ["DWM-1"],
    "DWM-1": ["FAQH-3"],
    "FAQH-2": ["REF-3", "PAY-2"],
    "REF-3": ["FAQH-2"],
    "FAQH-4": ["PLS-1"],
    "PLS-1": ["FAQH-4"],
    "WAR-1": ["WAR-2"],
    "WAR-2": ["WAR-1"],
}


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.65):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() if len(docs) else 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(q):
            idf = self.idf.get(t)
            if not idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


@dataclass
class Hit:
    chunk: Chunk
    score: float
    via: str = "hybrid"


class KnowledgeBase:
    def __init__(self, kb_dir: Path, embed_backend: str = "auto", embed_model: str = "BAAI/bge-small-en-v1.5",
                 cache_dir: Path | None = None):
        files = sorted(Path(kb_dir).glob("*.md"))
        if not files:
            raise FileNotFoundError(f"No kb/*.md found in {kb_dir}. Set DATASET_DIR.")
        self.chunks: list[Chunk] = [c for f in files for c in parse_doc(f)]
        self.by_uid = {c.uid: c for c in self.chunks}
        self.by_section: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            self.by_section.setdefault(c.section_id, []).append(c)
        self.bm25 = BM25([tokenize(f"{c.doc_title} {c.title} {c.title} {c.text}") for c in self.chunks])
        self.emb: np.ndarray | None = None
        self._embedder = None
        if embed_backend != "none":
            self._init_dense(embed_model, cache_dir)

    # ---- dense (optional; BM25 still works alone) ----
    def _init_dense(self, model: str, cache_dir: Path | None) -> None:
        try:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name=model)
            texts = [f"{c.title}\n{c.text}"[:2000] for c in self.chunks]
            key = hashlib.sha1((model + "|".join(texts)).encode()).hexdigest()[:16]
            cp = (cache_dir or Path(".cache")) / f"emb_{key}.npy"
            if cp.exists():
                self.emb = np.load(cp)
            else:
                self.emb = np.array(list(self._embedder.embed(texts)), dtype=np.float32)
                self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.save(cp, self.emb)
        except Exception as e:  # no fastembed / offline: degrade gracefully
            print(f"[kb] dense retrieval disabled ({type(e).__name__}); using BM25 only")
            self.emb = None

    def _dense_scores(self, query: str) -> np.ndarray | None:
        if self.emb is None or self._embedder is None:
            return None
        q = np.array(list(self._embedder.embed([query])), dtype=np.float32)[0]
        q /= np.linalg.norm(q) + 1e-9
        return self.emb @ q

    # ---- retrieval ----
    # Common abbreviations expanded for better matching
    ABBREV = {"cod": "cash on delivery", "upi": "upi payment refund", "emi": "emi installment payment",
              "rto": "return to origin undelivered", "ofd": "out for delivery", "gst": "gst invoice tax",
              "neft": "neft bank transfer refund", "imps": "imps bank transfer refund",
              "kg": "kilogram weight", "pcs": "pieces quantity"}

    @staticmethod
    def expand_query(query: str) -> str:
        low = query.lower()
        extra_parts = []
        # Synonym expansion
        for k, v in SYN.items():
            if re.search(rf"\b{re.escape(k)}\b", low):
                extra_parts.append(v)
        # Abbreviation expansion
        for abbr, expansion in KnowledgeBase.ABBREV.items():
            if re.search(rf"\b{re.escape(abbr)}\b", low):
                extra_parts.append(expansion)
        extra = " ".join(extra_parts)
        return f"{query} {extra}".strip()

    def sections(self, ids: list[str]) -> list[Hit]:
        """Pinned lookups: current (non-superseded) chunks for given section ids."""
        out = []
        for i in ids:
            for c in self.by_section.get(i, []):
                if not c.superseded:
                    out.append(Hit(c, 1.0, via="pinned"))
        return out

    _PAST = re.compile(r"\b(\d{1,2}\s+)?(january|february|march|april|may|june|july)\b|\b(jan|feb|mar|apr|jun|jul)\b|"
                       r"\bbefore\s+(1\s+)?august\b|\b(old|older|earlier|previous|purana|v3)\b", re.I)

    def search(self, query: str, k: int = 6, include_superseded: bool = False, expand: bool = True) -> list[Hit]:
        qt = tokenize(self.expand_query(query))
        n = len(self.chunks)
        lex = self.bm25.scores(qt)
        den = self._dense_scores(query)
        rrf = np.zeros(n)
        for sc in (lex, den):
            if sc is None:
                continue
            order = np.argsort(-sc)
            for rank, idx in enumerate(order[:40]):
                if sc[idx] > 0:
                    rrf[idx] += 1.0 / (60 + rank)
        # explicit section id mention in the query -> strong boost
        for sid in {m.upper() for m in re.findall(r"[A-Za-z]{2,}[-_.][A-Za-z0-9\-_.]+", query)}:
            for c in self.by_section.get(sid, []):
                rrf[self.chunks.index(c)] += 0.05
        final = np.zeros(n)
        # superseded sections are evidence only for older dates (e.g. delivered in July); otherwise they are down-weighted
        sup_w = 0.95 if (include_superseded or self._PAST.search(query)) else 0.4
        for i, c in enumerate(self.chunks):
            final[i] = rrf[i] * (0.6 + 0.4 * c.authority) * (sup_w if c.superseded else 1.0)
        order = [i for i in np.argsort(-final) if final[i] > 0][:k]
        hits = [Hit(self.chunks[i], float(final[i])) for i in order]
        if expand and hits:
            seen = {h.chunk.uid for h in hits}
            extras: list[Hit] = []
            for h in hits[:4]:
                refs = [a or b for a, b in REF_RE.findall(h.chunk.text)]
                if h.chunk.section_id in POLICY_LINKS:
                    refs.extend(POLICY_LINKS[h.chunk.section_id])
                for ref in refs:
                    if ref == h.chunk.section_id:
                        continue
                    for c in self.by_section.get(ref, []):
                        if c.uid not in seen and not c.superseded and len(extras) < 3:
                            seen.add(c.uid)
                            extras.append(Hit(c, h.score * 0.5, via="xref"))
            hits += extras
        return hits

    def retrieve(self, query: str, k: int = 6, pinned: list[str] | None = None) -> list[Hit]:
        hits = self.search(query, k=k)
        seen = {h.chunk.uid for h in hits}
        for h in self.sections(pinned or []):
            if h.chunk.uid not in seen:
                seen.add(h.chunk.uid)
                hits.append(h)
        return hits

    def coverage(self, query: str, hits: list[Hit]) -> float:
        """Fraction of the query's content words found in the retrieved text: a cheap 'do we know?' signal."""
        q = set(tokenize(query))
        if not q or not hits:
            return 0.0
        blob = set(tokenize(" ".join(h.chunk.text for h in hits[:4])))
        return len(q & blob) / len(q)

    def format_context(self, hits: list[Hit]) -> str:
        return "\n\n".join(h.chunk.render() for h in hits) if hits else "(no relevant passages found)"
