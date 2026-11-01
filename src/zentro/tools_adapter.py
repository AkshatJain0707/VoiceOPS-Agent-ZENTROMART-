"""Wraps the dataset's mock tools and adds product / pincode lookups.
Rules enforced here or in the agent (from README and agent_identity_verification): verify before sharing, one value at a
time, one retry, tickets only reported when created=True, verification secrets never repeated."""
from __future__ import annotations
import csv
import dataclasses
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .dates import DateCalc
from .kb import BM25, tokenize
from .speech_norm import canonical_order_id

MAX_VERIFY_ATTEMPTS = 2  # first try plus one retry, per order
HIDE_FROM_MODEL = {"delivery_pincode"}  # a verification value must never be echoed back


def _jsonable(obj: Any) -> dict:
    if isinstance(obj, dict):
        return json.loads(json.dumps(obj, default=str))
    if dataclasses.is_dataclass(obj):
        return json.loads(json.dumps(dataclasses.asdict(obj), default=str))
    return {"result": str(obj)}


def looks_verified(res: dict) -> bool:
    return isinstance(res, dict) and res.get("found") is True and res.get("verified") is True


def slim_order(res: dict, calc: DateCalc | None = None) -> dict:
    """What the model sees after verification: no nulls, no pincode, plus a few derived facts computed by code."""
    o = {k: v for k, v in res.items() if v not in (None, "", []) and k not in HIDE_FROM_MODEL}
    if calc:
        try:
            if o.get("delivered_on"):
                o["days_since_delivery"] = (calc.today - date.fromisoformat(str(o["delivered_on"])[:10])).days
        except ValueError:
            pass
    return o


def escalation_flags(order: dict) -> list[str]:
    """ESC-1 rules that can be decided from the record alone."""
    flags = []
    try:
        if int(order.get("contacts_last_7_days") or 0) >= 2:
            flags.append("ESC-1 rule 4: repeat contact (contacts_last_7_days >= 2) -> escalate")
    except (TypeError, ValueError):
        pass
    sku = order.get("damage_sku")
    if sku:
        for it in order.get("items", []):
            if it.get("sku") == sku:
                try:
                    if float(it.get("unit_price_inr") or 0) > 10000:
                        flags.append(f"ESC-1 rule 1: damaged/wrong item {sku} is worth more than 10000 rupees -> escalate")
                except (TypeError, ValueError):
                    pass
    return flags


@dataclass
class TicketResult:
    created: bool
    ticket_id: str = ""
    raw: dict = field(default_factory=dict)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ToolBox:
    def __init__(self, cfg, calc: DateCalc, order_fmt=None):
        self.calc, self.order_fmt = calc, order_fmt
        self.lookup_fn = self.ticket_fn = None
        self.import_error = ""
        try:
            p = str(cfg.dataset_dir.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
            from tools import lookup_order, create_ticket  # provided by the organisers' dataset
            self.lookup_fn, self.ticket_fn = lookup_order, create_ticket
        except Exception as e:
            self.import_error = f"{type(e).__name__}: {e}"
        d = cfg.data_dir
        self.products = _read(d / "products.csv") if (d / "products.csv").exists() else []
        self.pincodes = {r["pincode"]: r for r in _read(d / "pincodes.csv")} if (d / "pincodes.csv").exists() else {}
        self._pbm25 = BM25([tokenize(f"{p['title']} {p['brand']} {p['sku']}") for p in self.products]) if self.products else None
        self.sku_prefixes = sorted({p["sku"].split("-")[0] for p in self.products})

    # ---- orders ----
    def lookup_order(self, order_id: str, verification_value: str | None = None) -> dict:
        if not self.lookup_fn:
            return {"error": f"order system unavailable ({self.import_error})"}
        oid = canonical_order_id(order_id, self.order_fmt)
        v = "".join(ch for ch in (verification_value or "") if ch.isdigit()) or None
        try:
            return _jsonable(self.lookup_fn(oid, verification_value=v) if v else self.lookup_fn(oid))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def create_ticket(self, summary: str, order_id: str | None = None, reason: str | None = None,
                      intent: str | None = None) -> TicketResult:
        if not self.ticket_fn:
            return TicketResult(False, raw={"error": f"ticket system unavailable ({self.import_error})"})
        try:
            raw = _jsonable(self.ticket_fn(summary, order_id=order_id or None, reason=reason or None, intent=intent or None))
        except Exception as e:
            return TicketResult(False, raw={"error": f"{type(e).__name__}: {e}"})
        return TicketResult(bool(raw.get("created", False)), str(raw.get("ticket_id") or raw.get("id") or ""), raw)

    # ---- catalog and coverage ----
    def lookup_product(self, query: str) -> dict:
        q = (query or "").strip()
        m = re.search(r"\b([A-Za-z]{3})[-\s]?(\d{4})\b", q)
        if m:
            sku = f"{m.group(1).upper()}-{m.group(2)}"
            rows = [p for p in self.products if p["sku"] == sku]
            if rows:
                return {"matches": [self._prod(p) for p in rows]}
        if not self._pbm25:
            return {"matches": [], "note": "catalog unavailable"}
        sc = self._pbm25.scores(tokenize(q))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:3]
        return {"matches": [self._prod(self.products[i]) for i in order if sc[i] > 0]}

    @staticmethod
    def _prod(p: dict) -> dict:
        keys = ("sku", "title", "brand", "return_category", "price_inr", "fulfilled_by", "assured", "warranty_months")
        return {k: p[k] for k in keys if k in p}

    def check_pincode(self, pincode: str) -> dict:
        pin = "".join(ch for ch in (pincode or "") if ch.isdigit())
        r = self.pincodes.get(pin)
        if not r:
            return {"pincode": pin, "listed": False, "note": "not in the delivery coverage list"}
        return {"pincode": pin, "listed": True, **{k: r[k] for k in ("city", "state", "serviceable", "cod_available", "express_available")}}

    def date_math(self, **kw) -> dict:
        try:
            return self.calc.run(**kw)
        except Exception as e:
            return {"error": str(e)}
