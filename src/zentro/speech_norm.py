"""Fix the things ASR gets wrong: spoken order IDs and SKUs, spaced digits."""
from __future__ import annotations
import re

DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
          "six": "6", "seven": "7", "eight": "8", "nine": "9",
          "shunya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4", "paanch": "5", "panch": "5",
          "chhe": "6", "chheh": "6", "saat": "7", "aath": "8", "nau": "9"}
SKU_HOMOPHONES = {"to": "2", "too": "2", "for": "4", "fore": "4", "won": "1", "ate": "8", "tree": "3"}
MULT = {"double": 2, "triple": 3}
SEP = {"dash": "-", "hyphen": "-", "minus": "-"}

_TRIGGER = re.compile(r"(order\s*(?:id|number|no\.?|नंबर)?\s*(?:is|hai|है|:|=)?\s*)", re.I)
_VERIFY_TRIGGER = re.compile(r"((?:last\s*(?:four|4)\s*(?:digits?|numbers?)?|phone|mobile|pin\s*code|pincode)\s*(?:is|are|hai|है|:|=)?\s*)", re.I)


def _is_tok(t: str) -> bool:
    tl = t.lower().strip(",.।")
    return bool(re.fullmatch(r"[a-z]", tl) or re.fullmatch(r"\d+", tl) or tl in DIGITS or tl in MULT or tl in SEP
                or re.fullmatch(r"[a-z]{1,4}-?\d{2,}", tl))


def collapse_spoken_series(text: str, trigger_re: re.Pattern) -> str:
    out, pos = [], 0
    for m in trigger_re.finditer(text):
        if m.start() < pos:
            continue
        toks = re.findall(r"\S+", text[m.end():])
        taken, mult, used = [], 1, 0
        for t in toks[:20]:
            tl = t.lower().strip(",.।")
            if not _is_tok(t):
                break
            used += 1
            if tl in MULT:
                mult = MULT[tl]
                continue
            piece = DIGITS.get(tl, SEP.get(tl, t.strip(",.।")))
            taken.append(piece * mult if mult > 1 else piece)
            mult = 1
        if len("".join(taken)) < 3:
            continue
        span_end = m.end()
        for t in toks[:used]:
            span_end = text.index(t, span_end) + len(t)
        out.append(text[pos:m.end()])
        out.append("".join(taken).upper())
        pos = span_end
    out.append(text[pos:])
    return "".join(out)


def collapse_spoken_ids(text: str) -> str:
    """'order id Z O dash one two three' -> 'order id ZO-123'. Only fires right after an 'order' trigger."""
    return collapse_spoken_series(text, _TRIGGER)


def infer_id_format(ids: list[str]):
    """Learn the order-ID shape from data: (prefix, separator, digit count). Pure-digit IDs give ('', '', n)."""
    pats = set()
    for i in ids[:200]:
        m = re.fullmatch(r"([A-Za-z]*)([-_]?)(\d+)", i.strip())
        if not m:
            return None
        pats.add((m.group(1).upper(), m.group(2) if m.group(1) else "", len(m.group(3))))
    return pats.pop() if len(pats) == 1 else None


def canonical_order_id(raw: str, fmt) -> str:
    if fmt and not fmt[0]:
        return "".join(ch for ch in (raw or "") if "0" <= ch <= "9")
    raw = re.sub(r"\s+", "", raw or "").upper()
    if not fmt:
        return raw
    prefix, sep, n = fmt
    m = re.fullmatch(rf"{prefix}[-_]?(\d{{{n}}})", raw)
    return f"{prefix}{sep}{m.group(1)}" if m else raw


def normalize_skus(text: str, prefixes: list[str]) -> str:
    """'sku ele zero zero zero eight' / 'ele-zero zero one to' -> 'ELE-0012'-style. Prefixes come from products.csv."""
    if not prefixes:
        return text
    pre = "|".join(sorted(set(prefixes)))
    pat = re.compile(rf"\b({pre})[\s\-]*((?:(?:\d|zero|oh|one|two|three|four|five|six|seven|eight|nine|to|too|for|won|ate)[\s\-]*){{2,6}})", re.I)

    def fix(m):
        digits = ""
        for tok in re.findall(r"\d|[a-z]+", m.group(2).lower()):
            digits += tok if tok.isdigit() else DIGITS.get(tok) or SKU_HOMOPHONES.get(tok, "")
        digits = digits.zfill(4)[-4:] if len(digits) <= 4 else digits
        tail = m.group(0)[len(m.group(0).rstrip()):]
        return f"{m.group(1).upper()}-{digits}{tail}"
    return pat.sub(fix, text)


def normalize_utterance(text: str, fmt=None, sku_prefixes: list[str] | None = None) -> str:
    text = collapse_spoken_ids(text or "")
    text = collapse_spoken_series(text, _VERIFY_TRIGGER)
    text = re.sub(r"\bzentro\s*mart\b", "ZentroMart", text, flags=re.I)
    if fmt:
        prefix, sep, n = fmt
        if prefix:
            text = re.sub(rf"\b{prefix}[\s-]*(\d{{{n}}})\b", lambda m: f"{prefix}{sep}{m.group(1)}", text, flags=re.I)
        elif n == 8:  # '8978 2837' or '8978-2837' -> 89782837 (only two adjacent 4-digit groups)
            text = re.sub(r"(?<!\d)(\d{4})[\s-](\d{4})(?!\d)", r"\1\2", text)
    return normalize_skus(text, sku_prefixes or [])
